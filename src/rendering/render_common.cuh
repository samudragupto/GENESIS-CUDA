#ifndef RENDER_COMMON_CUH
#define RENDER_COMMON_CUH

#include <cuda_runtime.h>

struct Camera {
    float pos_x, pos_y, pos_z;
    float look_x, look_y, look_z;
    float up_x, up_y, up_z;
    float fov;
    float near_plane;
    float far_plane;
    float aspect;
    float yaw, pitch;
    float speed;
    float sensitivity;
};

struct FrameBuffer {
    float4* d_color;
    float*  d_depth;
    int     width;
    int     height;
};

struct LightParams {
    float sun_dir_x, sun_dir_y, sun_dir_z;
    float sun_color_r, sun_color_g, sun_color_b;
    float sun_intensity;
    float ambient_r, ambient_g, ambient_b;
    float ambient_intensity;
    float time_of_day;
};

void allocateFrameBuffer(FrameBuffer& fb, int width, int height);
void freeFrameBuffer(FrameBuffer& fb);
void clearFrameBuffer(FrameBuffer& fb, float4 clear_color, cudaStream_t stream = 0);

#endif