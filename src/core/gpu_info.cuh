#pragma once
#include <string>

namespace genesis {

struct DeviceCapabilities {
    int             deviceId;
    std::string     name;
    int             smCount;
    int             maxThreadsPerSM;
    int             maxThreadsPerBlock;
    int             warpSize;
    size_t          totalGlobalMem;
    size_t          sharedMemPerBlock;
    size_t          sharedMemPerSM;
    int             maxBlocksPerSM;
    int             computeMajor;
    int             computeMinor;
    bool            concurrentKernels;
    bool            unifiedMemory;
    int             asyncEngineCount;
    int             l2CacheSize;
    int             memBusWidth;
    float           memBandwidthGB;
    int             clockRateMHz;
    int             memClockMHz;
    bool            canMapHostMemory;
    bool            p2pSupported;
    int             optimalBlockSize1D;
    int             optimalBlockSize2D;
};

class GPUInfo {
public:
    static int              getDeviceCount();
    static DeviceCapabilities queryDevice(int deviceId = 0);
    static void             printDeviceInfo(int deviceId = 0);
    static void             selectBestDevice();
    static int              getOptimalBlockSize(int deviceId = 0);
};

} // namespace genesis