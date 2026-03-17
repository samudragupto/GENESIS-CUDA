#include "heatmap_generator.cuh"
#include "../core/cuda_utils.cuh"

void allocateHeatmapData(HeatmapData& hm, int width, int height) {
    hm.width = width;
    hm.height = height;
    CUDA_CHECK(cudaMalloc(&hm.d_color_output, width * height * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&hm.d_scalar_buffer, width * height * sizeof(float)));
    CUDA_CHECK(cudaMemset(hm.d_color_output, 0, width * height * sizeof(float4)));
    CUDA_CHECK(cudaMemset(hm.d_scalar_buffer, 0, width * height * sizeof(float)));
}

void freeHeatmapData(HeatmapData& hm) {
    cudaFree(hm.d_color_output);
    cudaFree(hm.d_scalar_buffer);
}

__device__ float4 viridisColormap(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);

    float4 c0 = make_float4(0.267004f, 0.004874f, 0.329415f, 1.0f);
    float4 c1 = make_float4(0.282327f, 0.140926f, 0.457517f, 1.0f);
    float4 c2 = make_float4(0.253935f, 0.265254f, 0.529983f, 1.0f);
    float4 c3 = make_float4(0.206756f, 0.371758f, 0.553117f, 1.0f);
    float4 c4 = make_float4(0.163625f, 0.471133f, 0.558148f, 1.0f);
    float4 c5 = make_float4(0.127568f, 0.566949f, 0.550556f, 1.0f);
    float4 c6 = make_float4(0.134692f, 0.658636f, 0.517649f, 1.0f);
    float4 c7 = make_float4(0.266941f, 0.748751f, 0.440573f, 1.0f);
    float4 c8 = make_float4(0.477504f, 0.821444f, 0.318195f, 1.0f);
    float4 c9 = make_float4(0.741388f, 0.873449f, 0.149561f, 1.0f);

    float4 colors[10] = {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9};

    float scaled = t * 9.0f;
    int idx = (int)scaled;
    idx = min(max(idx, 0), 8);
    float frac = scaled - (float)idx;

    float4 a = colors[idx];
    float4 b = colors[idx + 1];

    return make_float4(
        a.x + (b.x - a.x) * frac,
        a.y + (b.y - a.y) * frac,
        a.z + (b.z - a.z) * frac,
        1.0f
    );
}

__device__ float4 heatColormap(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    float r = fminf(t * 3.0f, 1.0f);
    float g = fminf(fmaxf((t - 0.33f) * 3.0f, 0.0f), 1.0f);
    float b = fminf(fmaxf((t - 0.66f) * 3.0f, 0.0f), 1.0f);
    return make_float4(r, g, b, 1.0f);
}

__device__ float4 terrainColormap(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    if (t < 0.3f) {
        float f = t / 0.3f;
        return make_float4(0.0f, 0.2f * f, 0.5f + 0.3f * f, 1.0f);
    } else if (t < 0.5f) {
        float f = (t - 0.3f) / 0.2f;
        return make_float4(0.76f * f, 0.7f * f + 0.2f * (1.0f - f), 0.1f, 1.0f);
    } else if (t < 0.7f) {
        float f = (t - 0.5f) / 0.2f;
        return make_float4(0.2f + 0.3f * f, 0.5f + 0.2f * f, 0.1f, 1.0f);
    } else {
        float f = (t - 0.7f) / 0.3f;
        return make_float4(0.5f + 0.5f * f, 0.5f + 0.5f * f, 0.5f + 0.5f * f, 1.0f);
    }
}

__global__ void scalarToHeatmapKernel(
    float4* __restrict__ output,
    const float* __restrict__ scalar,
    int width,
    int height,
    int field_width,
    int field_height,
    float min_val,
    float max_val,
    int colormap_type
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int fx = (int)((float)x / (float)width * (float)field_width);
    int fy = (int)((float)y / (float)height * (float)field_height);
    fx = min(max(fx, 0), field_width - 1);
    fy = min(max(fy, 0), field_height - 1);

    float val = scalar[fy * field_width + fx];
    float range = max_val - min_val;
    float t = (range > 0.0f) ? (val - min_val) / range : 0.0f;
    t = fminf(fmaxf(t, 0.0f), 1.0f);

    float4 color;
    switch (colormap_type) {
        case HEATMAP_DENSITY:
        case HEATMAP_ENERGY:
            color = viridisColormap(t);
            break;
        case HEATMAP_TEMPERATURE:
            color = heatColormap(t);
            break;
        case HEATMAP_ELEVATION:
            color = terrainColormap(t);
            break;
        default:
            color = viridisColormap(t);
            break;
    }

    output[y * width + x] = color;
}

__global__ void gaussianBlurHKernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int width,
    int height,
    int radius,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int dx = -radius; dx <= radius; dx++) {
        int sx = min(max(x + dx, 0), width - 1);
        float w = expf(-(float)(dx * dx) * inv_2sigma2);
        float4 c = input[y * width + sx];
        sum.x += c.x * w;
        sum.y += c.y * w;
        sum.z += c.z * w;
        weight_sum += w;
    }

    sum.x /= weight_sum;
    sum.y /= weight_sum;
    sum.z /= weight_sum;
    sum.w = 1.0f;
    output[y * width + x] = sum;
}

__global__ void gaussianBlurVKernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int width,
    int height,
    int radius,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int dy = -radius; dy <= radius; dy++) {
        int sy = min(max(y + dy, 0), height - 1);
        float w = expf(-(float)(dy * dy) * inv_2sigma2);
        float4 c = input[sy * width + x];
        sum.x += c.x * w;
        sum.y += c.y * w;
        sum.z += c.z * w;
        weight_sum += w;
    }

    sum.x /= weight_sum;
    sum.y /= weight_sum;
    sum.z /= weight_sum;
    sum.w = 1.0f;
    output[y * width + x] = sum;
}

__global__ void overlayHeatmapKernel(
    float4* __restrict__ framebuffer,
    const float4* __restrict__ heatmap,
    int width,
    int height,
    float alpha
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 fb = framebuffer[idx];
    float4 hm = heatmap[idx];

    framebuffer[idx] = make_float4(
        fb.x * (1.0f - alpha) + hm.x * alpha,
        fb.y * (1.0f - alpha) + hm.y * alpha,
        fb.z * (1.0f - alpha) + hm.z * alpha,
        1.0f
    );
}

void launchScalarToHeatmap(
    HeatmapData& hm,
    const float* d_scalar_field,
    int field_width,
    int field_height,
    float min_val,
    float max_val,
    HeatmapType type,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((hm.width + 15) / 16, (hm.height + 15) / 16);
    scalarToHeatmapKernel<<<grid, block, 0, stream>>>(
        hm.d_color_output, d_scalar_field,
        hm.width, hm.height, field_width, field_height,
        min_val, max_val, (int)type
    );
}

void launchGaussianBlurHeatmap(
    HeatmapData& hm,
    int radius,
    float sigma,
    cudaStream_t stream
) {
    float4* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, hm.width * hm.height * sizeof(float4)));

    dim3 block(16, 16);
    dim3 grid((hm.width + 15) / 16, (hm.height + 15) / 16);

    gaussianBlurHKernel<<<grid, block, 0, stream>>>(
        hm.d_color_output, d_temp, hm.width, hm.height, radius, sigma
    );
    gaussianBlurVKernel<<<grid, block, 0, stream>>>(
        d_temp, hm.d_color_output, hm.width, hm.height, radius, sigma
    );

    cudaFree(d_temp);
}

void launchOverlayHeatmap(
    float4* d_framebuffer,
    const float4* d_heatmap,
    int width,
    int height,
    float alpha,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    overlayHeatmapKernel<<<grid, block, 0, stream>>>(
        d_framebuffer, d_heatmap, width, height, alpha
    );
}