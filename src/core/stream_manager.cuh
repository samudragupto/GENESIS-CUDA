#ifndef STREAM_MANAGER_CUH
#define STREAM_MANAGER_CUH

#include <cuda_runtime.h>
#include "constants.cuh"

class StreamManager {
public:
    cudaStream_t streams[MAX_STREAMS];
    cudaEvent_t events[MAX_STREAMS];
    int num_streams;

    void initialize(int numStreams = MAX_STREAMS);
    void destroy();
    cudaStream_t getStream(int index);
    cudaEvent_t getEvent(int index);
    void synchronizeAll();
};

#endif