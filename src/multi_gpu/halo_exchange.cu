#include "halo_exchange.cuh"
#include "../core/cuda_utils.cuh"

__global__ void packHaloColumnsKernel(
    const float* __restrict__ field,
    float* __restrict__ buffer,
    int start_col,
    int num_cols,
    int world_size
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= world_size) return;

    for (int c = 0; c < num_cols; c++) {
        buffer[y * num_cols + c] = field[y * world_size + start_col + c];
    }
}

__global__ void unpackHaloColumnsKernel(
    float* __restrict__ field,
    const float* __restrict__ buffer,
    int start_col,
    int num_cols,
    int world_size
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= world_size) return;

    for (int c = 0; c < num_cols; c++) {
        field[y * world_size + start_col + c] = buffer[y * num_cols + c];
    }
}

__global__ void packHaloRowsKernel(
    const float* __restrict__ field,
    float* __restrict__ buffer,
    int start_row,
    int num_rows,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= world_size) return;

    for (int r = 0; r < num_rows; r++) {
        buffer[r * world_size + x] = field[(start_row + r) * world_size + x];
    }
}

__global__ void unpackHaloRowsKernel(
    float* __restrict__ field,
    const float* __restrict__ buffer,
    int start_row,
    int num_rows,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= world_size) return;

    for (int r = 0; r < num_rows; r++) {
        field[(start_row + r) * world_size + x] = buffer[r * world_size + x];
    }
}

void launchPackHaloRegion(
    const float* d_field,
    float* d_halo_buffer,
    int start_col, int num_cols,
    int world_size,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (world_size + block - 1) / block;
    packHaloColumnsKernel<<<grid, block, 0, stream>>>(
        d_field, d_halo_buffer, start_col, num_cols, world_size
    );
}

void launchUnpackHaloRegion(
    float* d_field,
    const float* d_halo_buffer,
    int start_col, int num_cols,
    int world_size,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (world_size + block - 1) / block;
    unpackHaloColumnsKernel<<<grid, block, 0, stream>>>(
        d_field, d_halo_buffer, start_col, num_cols, world_size
    );
}

void performHaloExchange(
    const HaloDescriptor& desc,
    float* d_src_field,
    float* d_dst_field,
    float* d_src_send_buf,
    float* d_dst_recv_buf,
    cudaStream_t src_stream,
    cudaStream_t dst_stream
) {
    int halo_size = desc.field_height * desc.halo_width;

    cudaSetDevice(desc.src_device);

    int block = 256;
    int grid = (desc.field_height + block - 1) / block;

    int pack_col = (desc.direction == 0) ?
        desc.world_size - desc.halo_width * 2 : 0;

    packHaloColumnsKernel<<<grid, block, 0, src_stream>>>(
        d_src_field, d_src_send_buf, pack_col, desc.halo_width, desc.world_size
    );

    cudaEvent_t pack_done;
    CUDA_CHECK(cudaEventCreate(&pack_done));
    CUDA_CHECK(cudaEventRecord(pack_done, src_stream));
    CUDA_CHECK(cudaStreamWaitEvent(dst_stream, pack_done, 0));

    CUDA_CHECK(cudaMemcpyPeerAsync(
        d_dst_recv_buf, desc.dst_device,
        d_src_send_buf, desc.src_device,
        halo_size * sizeof(float), dst_stream
    ));

    cudaSetDevice(desc.dst_device);

    int unpack_col = (desc.direction == 0) ? 0 : desc.world_size - desc.halo_width;

    unpackHaloColumnsKernel<<<grid, block, 0, dst_stream>>>(
        d_dst_field, d_dst_recv_buf, unpack_col, desc.halo_width, desc.world_size
    );

    cudaEventDestroy(pack_done);
}

void launchMultiFieldHaloExchange(
    float** d_fields,
    int num_fields,
    float** d_send_bufs,
    float** d_recv_bufs,
    const HaloDescriptor& desc,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (desc.field_height + block - 1) / block;

    for (int f = 0; f < num_fields; f++) {
        int pack_col = (desc.direction == 0) ?
            desc.world_size - desc.halo_width * 2 : 0;

        packHaloColumnsKernel<<<grid, block, 0, stream>>>(
            d_fields[f], d_send_bufs[f], pack_col, desc.halo_width, desc.world_size
        );
    }
}