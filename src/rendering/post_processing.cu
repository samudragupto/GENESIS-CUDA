#include "post_processing.cuh"
#include "../core/cuda_utils.cuh"

void allocatePostProcessBuffers(PostProcessBuffers& pp, int width, int height) {
    pp.width = width;
    pp.height = height;
    CUDA_CHECK(cudaMalloc(&pp.d_bright_pass, width * height * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&pp.d_bloom_buffer, width * height * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&pp.d_temp_buffer, width * height * sizeof(float4)));
}

void freePostProcessBuffers(PostProcessBuffers& pp) {
    cudaFree(pp.d_bright_pass);
    cudaFree(pp.d_bloom_buffer);
    cudaFree(pp.d_temp_buffer);
}

__global__ void brightPassKernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    float threshold,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 c = input[idx];
    float luminance = c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;

    if (luminance > threshold) {
        float scale = (luminance - threshold) / luminance;
        output[idx] = make_float4(c.x * scale, c.y * scale, c.z * scale, 1.0f);
    } else {
        output[idx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
}

__global__ void bloomBlurHKernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float weights[5] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};

    int idx = y * width + x;
    float4 sum = make_float4(
        input[idx].x * weights[0],
        input[idx].y * weights[0],
        input[idx].z * weights[0],
        1.0f
    );

    for (int i = 1; i < 5; i++) {
        int lx = max(x - i, 0);
        int rx = min(x + i, width - 1);

        float4 cl = input[y * width + lx];
        float4 cr = input[y * width + rx];

        sum.x += (cl.x + cr.x) * weights[i];
        sum.y += (cl.y + cr.y) * weights[i];
        sum.z += (cl.z + cr.z) * weights[i];
    }

    output[idx] = sum;
}

__global__ void bloomBlurVKernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float weights[5] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};

    int idx = y * width + x;
    float4 sum = make_float4(
        input[idx].x * weights[0],
        input[idx].y * weights[0],
        input[idx].z * weights[0],
        1.0f
    );

    for (int i = 1; i < 5; i++) {
        int ly = max(y - i, 0);
        int ry = min(y + i, height - 1);

        float4 cl = input[ly * width + x];
        float4 cr = input[ry * width + x];

        sum.x += (cl.x + cr.x) * weights[i];
        sum.y += (cl.y + cr.y) * weights[i];
        sum.z += (cl.z + cr.z) * weights[i];
    }

    output[idx] = sum;
}

__global__ void bloomCompositeKernel(
    float4* __restrict__ color,
    const float4* __restrict__ bloom,
    float intensity,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 c = color[idx];
    float4 b = bloom[idx];

    color[idx] = make_float4(
        c.x + b.x * intensity,
        c.y + b.y * intensity,
        c.z + b.z * intensity,
        1.0f
    );
}

__global__ void toneMappingKernel(
    float4* __restrict__ color,
    float exposure,
    float gamma,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 c = color[idx];

    c.x *= exposure;
    c.y *= exposure;
    c.z *= exposure;

    c.x = c.x * (2.51f * c.x + 0.03f) / (c.x * (2.43f * c.x + 0.59f) + 0.14f);
    c.y = c.y * (2.51f * c.y + 0.03f) / (c.y * (2.43f * c.y + 0.59f) + 0.14f);
    c.z = c.z * (2.51f * c.z + 0.03f) / (c.z * (2.43f * c.z + 0.59f) + 0.14f);

    float inv_gamma = 1.0f / gamma;
    c.x = powf(fmaxf(c.x, 0.0f), inv_gamma);
    c.y = powf(fmaxf(c.y, 0.0f), inv_gamma);
    c.z = powf(fmaxf(c.z, 0.0f), inv_gamma);

    c.x = fminf(c.x, 1.0f);
    c.y = fminf(c.y, 1.0f);
    c.z = fminf(c.z, 1.0f);

    color[idx] = c;
}

__global__ void fxaaKernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) output[y * width + x] = input[y * width + x];
        return;
    }

    float4 cc = input[y * width + x];
    float4 cn = input[(y - 1) * width + x];
    float4 cs = input[(y + 1) * width + x];
    float4 ce = input[y * width + (x + 1)];
    float4 cw = input[y * width + (x - 1)];

    float lc = cc.x * 0.299f + cc.y * 0.587f + cc.z * 0.114f;
    float ln = cn.x * 0.299f + cn.y * 0.587f + cn.z * 0.114f;
    float ls = cs.x * 0.299f + cs.y * 0.587f + cs.z * 0.114f;
    float le = ce.x * 0.299f + ce.y * 0.587f + ce.z * 0.114f;
    float lw = cw.x * 0.299f + cw.y * 0.587f + cw.z * 0.114f;

    float l_min = fminf(fminf(fminf(ln, ls), fminf(le, lw)), lc);
    float l_max = fmaxf(fmaxf(fmaxf(ln, ls), fmaxf(le, lw)), lc);
    float l_range = l_max - l_min;

    float threshold = 0.0625f;
    if (l_range < fmaxf(threshold, l_max * 0.125f)) {
        output[y * width + x] = cc;
        return;
    }

    float4 cnw = input[(y - 1) * width + (x - 1)];
    float4 cne = input[(y - 1) * width + (x + 1)];
    float4 csw = input[(y + 1) * width + (x - 1)];
    float4 cse = input[(y + 1) * width + (x + 1)];

    output[y * width + x] = make_float4(
        (cc.x * 4.0f + cn.x + cs.x + ce.x + cw.x + (cnw.x + cne.x + csw.x + cse.x) * 0.5f) / 10.0f,
        (cc.y * 4.0f + cn.y + cs.y + ce.y + cw.y + (cnw.y + cne.y + csw.y + cse.y) * 0.5f) / 10.0f,
        (cc.z * 4.0f + cn.z + cs.z + ce.z + cw.z + (cnw.z + cne.z + csw.z + cse.z) * 0.5f) / 10.0f,
        1.0f
    );
}

__global__ void vignetteKernel(
    float4* __restrict__ color,
    float strength,
    float radius,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ((float)x / (float)width) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    float dist = sqrtf(u * u + v * v);
    float vignette = 1.0f - fminf(fmaxf((dist - radius) / (1.0f - radius) * strength, 0.0f), 1.0f);

    int idx = y * width + x;
    float4 c = color[idx];
    color[idx] = make_float4(c.x * vignette, c.y * vignette, c.z * vignette, c.w);
}

void launchBrightPass(
    const FrameBuffer& fb,
    PostProcessBuffers& pp,
    float threshold,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    brightPassKernel<<<grid, block, 0, stream>>>(
        fb.d_color, pp.d_bright_pass, threshold, fb.width, fb.height
    );
}

void launchBloomBlur(
    PostProcessBuffers& pp,
    int passes,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((pp.width + 15) / 16, (pp.height + 15) / 16);

    CUDA_CHECK(cudaMemcpyAsync(pp.d_bloom_buffer, pp.d_bright_pass,
        pp.width * pp.height * sizeof(float4), cudaMemcpyDeviceToDevice, stream));

    for (int i = 0; i < passes; i++) {
        bloomBlurHKernel<<<grid, block, 0, stream>>>(
            pp.d_bloom_buffer, pp.d_temp_buffer, pp.width, pp.height
        );
        bloomBlurVKernel<<<grid, block, 0, stream>>>(
            pp.d_temp_buffer, pp.d_bloom_buffer, pp.width, pp.height
        );
    }
}

void launchBloomComposite(
    FrameBuffer& fb,
    const PostProcessBuffers& pp,
    float intensity,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    bloomCompositeKernel<<<grid, block, 0, stream>>>(
        fb.d_color, pp.d_bloom_buffer, intensity, fb.width, fb.height
    );
}

void launchToneMapping(
    FrameBuffer& fb,
    float exposure,
    float gamma,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    toneMappingKernel<<<grid, block, 0, stream>>>(
        fb.d_color, exposure, gamma, fb.width, fb.height
    );
}

void launchFXAA(
    FrameBuffer& fb,
    PostProcessBuffers& pp,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);

    fxaaKernel<<<grid, block, 0, stream>>>(
        fb.d_color, pp.d_temp_buffer, fb.width, fb.height
    );

    CUDA_CHECK(cudaMemcpyAsync(fb.d_color, pp.d_temp_buffer,
        fb.width * fb.height * sizeof(float4), cudaMemcpyDeviceToDevice, stream));
}

void launchVignette(
    FrameBuffer& fb,
    float strength,
    float radius,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    vignetteKernel<<<grid, block, 0, stream>>>(
        fb.d_color, strength, radius, fb.width, fb.height
    );
}

void launchFullPostProcess(
    FrameBuffer& fb,
    PostProcessBuffers& pp,
    const PostProcessParams& params,
    cudaStream_t stream
) {
    launchBrightPass(fb, pp, params.bloom_threshold, stream);
    launchBloomBlur(pp, params.bloom_blur_passes, stream);
    launchBloomComposite(fb, pp, params.bloom_intensity, stream);
    launchToneMapping(fb, params.exposure, params.gamma, stream);

    if (params.fxaa_enabled) {
        launchFXAA(fb, pp, stream);
    }

    if (params.vignette_strength > 0.0f) {
        launchVignette(fb, params.vignette_strength, params.vignette_radius, stream);
    }
}