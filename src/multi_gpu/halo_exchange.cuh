#ifndef HALO_EXCHANGE_CUH
#define HALO_EXCHANGE_CUH

#include <cuda_runtime.h>

struct HaloDescriptor {
    int src_device;
    int dst_device;
    int halo_width;
    int field_height;
    int world_size;
    int direction;
};

void launchPackHaloRegion(
    const float* d_field,
    float* d_halo_buffer,
    int start_col, int num_cols,
    int world_size,
    cudaStream_t stream = 0
);

void launchUnpackHaloRegion(
    float* d_field,
    const float* d_halo_buffer,
    int start_col, int num_cols,
    int world_size,
    cudaStream_t stream = 0
);

void performHaloExchange(
    const HaloDescriptor& desc,
    float* d_src_field,
    float* d_dst_field,
    float* d_src_send_buf,
    float* d_dst_recv_buf,
    cudaStream_t src_stream,
    cudaStream_t dst_stream
);

void launchMultiFieldHaloExchange(
    float** d_fields,
    int num_fields,
    float** d_send_bufs,
    float** d_recv_bufs,
    const HaloDescriptor& desc,
    cudaStream_t stream = 0
);

#endif