#include "cloud_raymarcher.cuh"
#include "../core/cuda_utils.cuh"

__device__ float hash3d(float x, float y, float z) {
    float n = sinf(x * 127.1f + y * 311.7f + z * 74.7f);
    return (n * 43758.5453f) - floorf(n * 43758.5453f);
}

__device__ float noise3d(float x, float y, float z) {
    float ix = floorf(x); float iy = floorf(y); float iz = floorf(z);
    float fx = x - ix; float fy = y - iy; float fz = z - iz;

    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    fz = fz * fz * (3.0f - 2.0f * fz);

    float a = hash3d(ix, iy, iz);
    float b = hash3d(ix + 1.0f, iy, iz);
    float c = hash3d(ix, iy + 1.0f, iz);
    float d = hash3d(ix + 1.0f, iy + 1.0f, iz);
    float e = hash3d(ix, iy, iz + 1.0f);
    float f = hash3d(ix + 1.0f, iy, iz + 1.0f);
    float g = hash3d(ix, iy + 1.0f, iz + 1.0f);
    float h = hash3d(ix + 1.0f, iy + 1.0f, iz + 1.0f);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return k0 + k1 * fx + k2 * fy + k3 * fz + k4 * fx * fy +
           k5 * fy * fz + k6 * fz * fx + k7 * fx * fy * fz;
}

__device__ float fbm3d(float x, float y, float z, int octaves) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise3d(x * frequency, y * frequency, z * frequency);
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return value;
}

__device__ float cloudDensity(float3 pos, float time, float scale, float coverage) {
    float x = pos.x * scale + time * 0.01f;
    float y = pos.y * scale * 2.0f;
    float z = pos.z * scale + time * 0.005f;

    float base = fbm3d(x, y, z, 5);
    float detail = fbm3d(x * 3.0f, y * 3.0f, z * 3.0f, 3) * 0.3f;

    float density = base + detail - (1.0f - coverage);
    return fmaxf(density, 0.0f);
}

__global__ void cloudRaymarchKernel(
    float4* __restrict__ color,
    float* __restrict__ depth,
    int width,
    int height,
    float cam_px, float cam_py, float cam_pz,
    float cam_lx, float cam_ly, float cam_lz,
    float cam_ux, float cam_uy, float cam_uz,
    float fov, float aspect,
    float sun_dx, float sun_dy, float sun_dz,
    float sun_r, float sun_g, float sun_b,
    float cloud_base, float cloud_top,
    float cloud_coverage, float cloud_density_scale,
    float cloud_scale, float time,
    int num_steps, int num_light_steps,
    float light_absorption, float ambient_light
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ((float)x / (float)width) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    float tan_fov = tanf(fov * 0.5f * 3.14159265f / 180.0f);

    float3 fwd = make_float3(cam_lx - cam_px, cam_ly - cam_py, cam_lz - cam_pz);
    float fwd_len = sqrtf(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z);
    fwd.x /= fwd_len; fwd.y /= fwd_len; fwd.z /= fwd_len;

    float3 up_v = make_float3(cam_ux, cam_uy, cam_uz);
    float3 right_v = make_float3(
        fwd.y * up_v.z - fwd.z * up_v.y,
        fwd.z * up_v.x - fwd.x * up_v.z,
        fwd.x * up_v.y - fwd.y * up_v.x
    );
    float right_len = sqrtf(right_v.x * right_v.x + right_v.y * right_v.y + right_v.z * right_v.z);
    right_v.x /= right_len; right_v.y /= right_len; right_v.z /= right_len;

    up_v = make_float3(
        right_v.y * fwd.z - right_v.z * fwd.y,
        right_v.z * fwd.x - right_v.x * fwd.z,
        right_v.x * fwd.y - right_v.y * fwd.x
    );

    float3 ray_dir;
    ray_dir.x = fwd.x + right_v.x * u * tan_fov * aspect + up_v.x * v * tan_fov;
    ray_dir.y = fwd.y + right_v.y * u * tan_fov * aspect + up_v.y * v * tan_fov;
    ray_dir.z = fwd.z + right_v.z * u * tan_fov * aspect + up_v.z * v * tan_fov;
    float rd_len = sqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z);
    ray_dir.x /= rd_len; ray_dir.y /= rd_len; ray_dir.z /= rd_len;

    if (ray_dir.y < 0.001f) return;

    float t_base = (cloud_base - cam_py) / ray_dir.y;
    float t_top = (cloud_top - cam_py) / ray_dir.y;

    if (t_base < 0.0f && t_top < 0.0f) return;
    if (t_base > t_top) { float tmp = t_base; t_base = t_top; t_top = tmp; }
    t_base = fmaxf(t_base, 0.0f);

    float step_size = (t_top - t_base) / (float)num_steps;

    float transmittance = 1.0f;
    float3 light_energy = make_float3(0.0f, 0.0f, 0.0f);

    float3 sun_dir = make_float3(sun_dx, sun_dy, sun_dz);
    float sun_len = sqrtf(sun_dir.x * sun_dir.x + sun_dir.y * sun_dir.y + sun_dir.z * sun_dir.z);
    sun_dir.x /= sun_len; sun_dir.y /= sun_len; sun_dir.z /= sun_len;

    for (int i = 0; i < num_steps; i++) {
        if (transmittance < 0.01f) break;

        float t = t_base + ((float)i + 0.5f) * step_size;
        float3 pos = make_float3(
            cam_px + ray_dir.x * t,
            cam_py + ray_dir.y * t,
            cam_pz + ray_dir.z * t
        );

        float height_frac = (pos.y - cloud_base) / (cloud_top - cloud_base);
        float height_atten = 4.0f * height_frac * (1.0f - height_frac);

        float density = cloudDensity(pos, time, cloud_scale, cloud_coverage) * cloud_density_scale * height_atten;
        if (density < 0.001f) continue;

        float light_od = 0.0f;
        float light_step = (cloud_top - pos.y) / fmaxf(sun_dir.y, 0.001f) / (float)num_light_steps;

        for (int j = 0; j < num_light_steps; j++) {
            float lt = ((float)j + 0.5f) * light_step;
            float3 light_pos = make_float3(
                pos.x + sun_dir.x * lt,
                pos.y + sun_dir.y * lt,
                pos.z + sun_dir.z * lt
            );
            if (light_pos.y > cloud_top || light_pos.y < cloud_base) break;

            float lh = (light_pos.y - cloud_base) / (cloud_top - cloud_base);
            float lha = 4.0f * lh * (1.0f - lh);
            light_od += cloudDensity(light_pos, time, cloud_scale, cloud_coverage) * cloud_density_scale * lha * light_step;
        }

        float light_transmittance = expf(-light_od * light_absorption);

        float3 incoming;
        incoming.x = (light_transmittance * sun_r + ambient_light) * density;
        incoming.y = (light_transmittance * sun_g + ambient_light) * density;
        incoming.z = (light_transmittance * sun_b + ambient_light) * density;

        light_energy.x += incoming.x * transmittance * step_size;
        light_energy.y += incoming.y * transmittance * step_size;
        light_energy.z += incoming.z * transmittance * step_size;

        transmittance *= expf(-density * light_absorption * step_size);
    }

    int idx = y * width + x;
    float4 existing = color[idx];

    color[idx] = make_float4(
        existing.x * transmittance + light_energy.x,
        existing.y * transmittance + light_energy.y,
        existing.z * transmittance + light_energy.z,
        1.0f
    );
}

void launchCloudRaymarching(
    FrameBuffer& fb,
    const Camera& camera,
    const LightParams& light,
    const CloudParams& clouds,
    cudaStream_t stream
) {
    dim3 block(8, 8);
    dim3 grid((fb.width + 7) / 8, (fb.height + 7) / 8);

    cloudRaymarchKernel<<<grid, block, 0, stream>>>(
        fb.d_color, fb.d_depth,
        fb.width, fb.height,
        camera.pos_x, camera.pos_y, camera.pos_z,
        camera.look_x, camera.look_y, camera.look_z,
        camera.up_x, camera.up_y, camera.up_z,
        camera.fov, camera.aspect,
        light.sun_dir_x, light.sun_dir_y, light.sun_dir_z,
        light.sun_color_r, light.sun_color_g, light.sun_color_b,
        clouds.cloud_base_height, clouds.cloud_top_height,
        clouds.cloud_coverage, clouds.cloud_density,
        clouds.cloud_scale, clouds.time,
        clouds.num_steps, clouds.num_light_steps,
        clouds.light_absorption, clouds.ambient_light
    );
}