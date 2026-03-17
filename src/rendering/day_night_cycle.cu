#include "day_night_cycle.cuh"
#include "../core/cuda_utils.cuh"
#include <cmath>

void updateDayNightCycle(
    DayNightState& state,
    LightParams& light,
    float dt
) {
    state.time_of_day += dt / state.day_length;
    if (state.time_of_day > 1.0f) state.time_of_day -= 1.0f;

    state.sun_angle = state.time_of_day * 2.0f * 3.14159265f - 3.14159265f * 0.5f;

    light.sun_dir_x = cosf(state.sun_angle);
    light.sun_dir_y = sinf(state.sun_angle);
    light.sun_dir_z = 0.3f;

    float len = sqrtf(light.sun_dir_x * light.sun_dir_x +
                      light.sun_dir_y * light.sun_dir_y +
                      light.sun_dir_z * light.sun_dir_z);
    light.sun_dir_x /= len;
    light.sun_dir_y /= len;
    light.sun_dir_z /= len;

    float day_factor = fmaxf(light.sun_dir_y, 0.0f);
    float sunset_factor = fmaxf(1.0f - fabsf(light.sun_dir_y) * 5.0f, 0.0f);

    light.sun_color_r = 1.0f + sunset_factor * 0.5f;
    light.sun_color_g = 0.9f - sunset_factor * 0.3f;
    light.sun_color_b = 0.8f - sunset_factor * 0.5f;

    light.sun_intensity = day_factor;

    light.ambient_r = 0.05f + day_factor * 0.15f + sunset_factor * 0.1f;
    light.ambient_g = 0.05f + day_factor * 0.15f;
    light.ambient_b = 0.08f + day_factor * 0.12f;
    light.ambient_intensity = 0.2f + day_factor * 0.3f;

    light.time_of_day = state.time_of_day;
}

__global__ void dayNightLightingKernel(
    float4* __restrict__ color,
    int width,
    int height,
    float ambient_r, float ambient_g, float ambient_b,
    float sun_intensity,
    float time_of_day
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 c = color[idx];

    float night_factor = fmaxf(1.0f - sun_intensity * 2.0f, 0.0f);
    float night_blue_shift = night_factor * 0.3f;

    c.x *= (1.0f - night_blue_shift);
    c.y *= (1.0f - night_blue_shift * 0.5f);
    c.z *= (1.0f + night_blue_shift * 0.5f);

    float brightness = sun_intensity * 0.7f + 0.3f;
    c.x *= brightness;
    c.y *= brightness;
    c.z *= brightness;

    c.x += ambient_r * 0.1f;
    c.y += ambient_g * 0.1f;
    c.z += ambient_b * 0.1f;

    color[idx] = c;
}

__global__ void starFieldKernel(
    float4* __restrict__ color,
    int width,
    int height,
    float cam_yaw, float cam_pitch,
    float night_factor,
    unsigned int seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (night_factor < 0.1f) return;

    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    if (v < 0.3f) return;

    float sx = u * 200.0f + cam_yaw * 10.0f;
    float sy = v * 200.0f + cam_pitch * 10.0f;

    float star_hash = sinf(floorf(sx) * 127.1f + floorf(sy) * 311.7f + (float)seed * 0.01f) * 43758.5453f;
    star_hash = star_hash - floorf(star_hash);

    if (star_hash > 0.997f) {
        float brightness = (star_hash - 0.997f) / 0.003f * night_factor;
        float twinkle = 0.7f + 0.3f * sinf(star_hash * 100.0f + (float)seed * 0.1f);
        brightness *= twinkle;

        int idx = y * width + x;
        float4 c = color[idx];
        c.x += brightness;
        c.y += brightness;
        c.z += brightness * 1.2f;
        color[idx] = c;
    }
}

void launchApplyDayNightLighting(
    FrameBuffer& fb,
    const LightParams& light,
    const DayNightState& state,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    dayNightLightingKernel<<<grid, block, 0, stream>>>(
        fb.d_color, fb.width, fb.height,
        light.ambient_r, light.ambient_g, light.ambient_b,
        light.sun_intensity, light.time_of_day
    );
}

void launchStarField(
    FrameBuffer& fb,
    const Camera& camera,
    float night_factor,
    unsigned int seed,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    starFieldKernel<<<grid, block, 0, stream>>>(
        fb.d_color, fb.width, fb.height,
        camera.yaw, camera.pitch, night_factor, seed
    );
}