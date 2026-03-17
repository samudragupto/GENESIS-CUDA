#include "stream_manager.cuh"
#include "cuda_utils.cuh"

void StreamManager::initialize(int numStreams) {
    num_streams = (numStreams > MAX_STREAMS) ? MAX_STREAMS : numStreams;
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    }
}

void StreamManager::destroy() {
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
    num_streams = 0;
}

cudaStream_t StreamManager::getStream(int index) {
    if (index < 0 || index >= num_streams) return 0;
    return streams[index];
}

cudaEvent_t StreamManager::getEvent(int index) {
    if (index < 0 || index >= num_streams) return nullptr;
    return events[index];
}

void StreamManager::synchronizeAll() {
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
}