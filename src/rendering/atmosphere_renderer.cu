#include "atmosphere_renderer.cuh"
#include "../core/cuda_utils.cuh"

__device__ float3 normalize3(float3 v) {
    float inv = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z + 1e-10f);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float2 raySphereIntersect(float3 origin, float3 dir, float radius) {
    float b = 2.0f * dot3(origin, dir);
    float c = dot3(origin, origin) - radius * radius;
    float disc = b * b - 4.0f * c;
    if (disc < 0.0f) return make_float2(-1.0f, -1.0f);
    disc = sqrtf(disc);
    return make_float2((-b - disc) * 0.5f, (-b + disc) * 0.5f);
}

__device__ float rayleighPhase(float cos_angle) {
    return 0.75f * (1.0f + cos_angle * cos_angle);
}

__device__ float miePhase(float cos_angle, float g) {
    float g2 = g * g;
    float num = 1.5f * (1.0f - g2) * (1.0f + cos_angle * cos_angle);
    float denom = (2.0f + g2) * powf(1.0f + g2 - 2.0f * g * cos_angle, 1.5f);
    return num / denom;
}

__global__ void atmosphericScatteringKernel(
    float4* __restrict__ color,
    int width,
    int height,
    float cam_px, float cam_py, float cam_pz,
    float cam_lx, float cam_ly, float cam_lz,
    float cam_ux, float cam_uy, float cam_uz,
    float fov, float aspect,
    float sun_dx, float sun_dy, float sun_dz,
    float sun_r, float sun_g, float sun_b,
    float planet_radius,
    float atmo_radius,
    float rayleigh_sh,
    float mie_sh,
    float mie_g,
    float3 rayleigh_coeff,
    float mie_coeff_val,
    int num_samples,
    int num_light_samples
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ((float)x / (float)width) * 2.0f - 1.0f;
    float v = ((float)y / (float)height) * 2.0f - 1.0f;

    float tan_fov = tanf(fov * 0.5f * 3.14159265f / 180.0f);

    float3 forward = normalize3(make_float3(cam_lx - cam_px, cam_ly - cam_py, cam_lz - cam_pz));
    float3 up = make_float3(cam_ux, cam_uy, cam_uz);
    float3 right = normalize3(make_float3(
        forward.y * up.z - forward.z * up.y,
        forward.z * up.x - forward.x * up.z,
        forward.x * up.y - forward.y * up.x
    ));
    up = make_float3(
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x
    );

    float3 ray_dir = normalize3(make_float3(
        forward.x + right.x * u * tan_fov * aspect + up.x * v * tan_fov,
        forward.y + right.y * u * tan_fov * aspect + up.y * v * tan_fov,
        forward.z + right.z * u * tan_fov * aspect + up.z * v * tan_fov
    ));

    float3 origin = make_float3(0.0f, planet_radius + cam_py, 0.0f);
    float3 sun_dir = normalize3(make_float3(sun_dx, sun_dy, sun_dz));

    float2 atmo_hit = raySphereIntersect(origin, ray_dir, atmo_radius);
    if (atmo_hit.y < 0.0f) {
        color[y * width + x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    float t_start = fmaxf(atmo_hit.x, 0.0f);
    float t_end = atmo_hit.y;

    float2 planet_hit = raySphereIntersect(origin, ray_dir, planet_radius);
    if (planet_hit.x > 0.0f) {
        t_end = planet_hit.x;
    }

    float segment_length = (t_end - t_start) / (float)num_samples;

    float3 rayleigh_sum = make_float3(0.0f, 0.0f, 0.0f);
    float mie_sum = 0.0f;
    float optical_depth_r = 0.0f;
    float optical_depth_m = 0.0f;

    for (int i = 0; i < num_samples; i++) {
        float t = t_start + ((float)i + 0.5f) * segment_length;
        float3 sample_pos = make_float3(
            origin.x + ray_dir.x * t,
            origin.y + ray_dir.y * t,
            origin.z + ray_dir.z * t
        );

        float height = length3(sample_pos) - planet_radius;
        float hr = expf(-height / rayleigh_sh) * segment_length;
        float hm = expf(-height / mie_sh) * segment_length;

        optical_depth_r += hr;
        optical_depth_m += hm;

        float2 light_hit = raySphereIntersect(sample_pos, sun_dir, atmo_radius);
        float light_segment = light_hit.y / (float)num_light_samples;
        float light_od_r = 0.0f;
        float light_od_m = 0.0f;

        int j;
        for (j = 0; j < num_light_samples; j++) {
            float lt = ((float)j + 0.5f) * light_segment;
            float3 light_pos = make_float3(
                sample_pos.x + sun_dir.x * lt,
                sample_pos.y + sun_dir.y * lt,
                sample_pos.z + sun_dir.z * lt
            );
            float lh = length3(light_pos) - planet_radius;
            if (lh < 0.0f) break;
            light_od_r += expf(-lh / rayleigh_sh) * light_segment;
            light_od_m += expf(-lh / mie_sh) * light_segment;
        }
        if (j < num_light_samples) continue;

        float3 tau = make_float3(
            rayleigh_coeff.x * (optical_depth_r + light_od_r) + mie_coeff_val * (optical_depth_m + light_od_m),
            rayleigh_coeff.y * (optical_depth_r + light_od_r) + mie_coeff_val * (optical_depth_m + light_od_m),
            rayleigh_coeff.z * (optical_depth_r + light_od_r) + mie_coeff_val * (optical_depth_m + light_od_m)
        );

        float3 attenuation = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));

        rayleigh_sum.x += hr * attenuation.x;
        rayleigh_sum.y += hr * attenuation.y;
        rayleigh_sum.z += hr * attenuation.z;
        mie_sum += hm * attenuation.x;
    }

    float cos_angle = dot3(ray_dir, sun_dir);
    float rp = rayleighPhase(cos_angle);
    float mp = miePhase(cos_angle, mie_g);

    float3 result = make_float3(
        (rayleigh_sum.x * rayleigh_coeff.x * rp + mie_sum * mie_coeff_val * mp) * sun_r,
        (rayleigh_sum.y * rayleigh_coeff.y * rp + mie_sum * mie_coeff_val * mp) * sun_g,
        (rayleigh_sum.z * rayleigh_coeff.z * rp + mie_sum * mie_coeff_val * mp) * sun_b
    );

    result.x = 1.0f - expf(-result.x * 2.0f);
    result.y = 1.0f - expf(-result.y * 2.0f);
    result.z = 1.0f - expf(-result.z * 2.0f);

    color[y * width + x] = make_float4(result.x, result.y, result.z, 1.0f);
}

__global__ void skyDomeKernel(
    float4* __restrict__ color,
    int width,
    int height,
    float time_of_day,
    float sun_dx, float sun_dy, float sun_dz,
    float sun_r, float sun_g, float sun_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    float day = fmaxf(sun_dy, 0.0f);
    float night = fmaxf(-sun_dy, 0.0f);

    float3 day_top = make_float3(0.1f, 0.3f, 0.8f);
    float3 day_bottom = make_float3(0.5f, 0.7f, 1.0f);
    float3 night_top = make_float3(0.01f, 0.01f, 0.05f);
    float3 night_bottom = make_float3(0.02f, 0.02f, 0.08f);
    float3 sunset_color = make_float3(1.0f, 0.4f, 0.1f);

    float3 sky;
    sky.x = day_top.x * v + day_bottom.x * (1.0f - v);
    sky.y = day_top.y * v + day_bottom.y * (1.0f - v);
    sky.z = day_top.z * v + day_bottom.z * (1.0f - v);

    float3 night_sky;
    night_sky.x = night_top.x * v + night_bottom.x * (1.0f - v);
    night_sky.y = night_top.y * v + night_bottom.y * (1.0f - v);
    night_sky.z = night_top.z * v + night_bottom.z * (1.0f - v);

    float blend = day / (day + night + 0.001f);

    float sunset_factor = fmaxf(1.0f - fabsf(sun_dy) * 5.0f, 0.0f);
    sunset_factor *= fmaxf(1.0f - v, 0.0f);

    float3 result;
    result.x = sky.x * blend + night_sky.x * (1.0f - blend) + sunset_color.x * sunset_factor * 0.5f;
    result.y = sky.y * blend + night_sky.y * (1.0f - blend) + sunset_color.y * sunset_factor * 0.5f;
    result.z = sky.z * blend + night_sky.z * (1.0f - blend) + sunset_color.z * sunset_factor * 0.5f;

    int idx = y * width + x;
    float existing_depth = 1e30f;
    if (color[idx].w < 0.5f || existing_depth > 1e20f) {
        color[idx] = make_float4(result.x, result.y, result.z, 1.0f);
    }
}

void launchAtmosphericScattering(
    FrameBuffer& fb,
    const Camera& camera,
    const LightParams& light,
    const AtmosphereParams& atmo,
    cudaStream_t stream
) {
    dim3 block(8, 8);
    dim3 grid((fb.width + 7) / 8, (fb.height + 7) / 8);

    atmosphericScatteringKernel<<<grid, block, 0, stream>>>(
        fb.d_color, fb.width, fb.height,
        camera.pos_x, camera.pos_y, camera.pos_z,
        camera.look_x, camera.look_y, camera.look_z,
        camera.up_x, camera.up_y, camera.up_z,
        camera.fov, camera.aspect,
        light.sun_dir_x, light.sun_dir_y, light.sun_dir_z,
        light.sun_color_r, light.sun_color_g, light.sun_color_b,
        atmo.planet_radius, atmo.atmosphere_radius,
        atmo.rayleigh_scale_height, atmo.mie_scale_height, atmo.mie_g,
        atmo.rayleigh_coeff, atmo.mie_coeff,
        atmo.num_samples, atmo.num_light_samples
    );
}

void launchSkyDome(
    FrameBuffer& fb,
    const Camera& camera,
    const LightParams& light,
    float time_of_day,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);

    skyDomeKernel<<<grid, block, 0, stream>>>(
        fb.d_color, fb.width, fb.height,
        time_of_day,
        light.sun_dir_x, light.sun_dir_y, light.sun_dir_z,
        light.sun_color_r, light.sun_color_g, light.sun_color_b
    );
}