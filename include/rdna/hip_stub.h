#ifndef RDNA_HIP_STUB_H
#define RDNA_HIP_STUB_H

// HIP stub definitions for development without ROCm installation
// These are placeholder definitions that allow compilation without ROCm

#include <cstdint>
#include <string>

// HIP error codes (simplified)
typedef int hipError_t;
#define hipSuccess 0
#define hipErrorInvalidValue 1
#define hipErrorMemoryAllocation 2
#define hipErrorNotInitialized 3

// HIP device properties structure (simplified)
struct hipDeviceProp_t {
    char name[256];
    char gcnArchName[256];
    size_t totalGlobalMem;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int warpSize;
    int pciBusID;
    int pciDeviceID;
    int arch;
};

// HIP stream type (placeholder)
typedef void* hipStream_t;

// HIP runtime functions (stubs)
inline const char* hipGetErrorString(hipError_t error) {
    static const char* error_strings[] = {
        "hipSuccess",
        "hipErrorInvalidValue", 
        "hipErrorMemoryAllocation",
        "hipErrorNotInitialized"
    };
    if (error >= 0 && error < 4) return error_strings[error];
    return "Unknown hipError_t value";
}

inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipGetDeviceCount(int* count) { *count = 1; return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    // Return mock properties for RDNA device
    snprintf(prop->name, 256, "AMD Radeon RX 6800 XT (Stub)");
    snprintf(prop->gcnArchName, 256, "gfx1030");
    prop->totalGlobalMem = 16ULL * 1024 * 1024 * 1024; // 16GB
    prop->multiProcessorCount = 72;
    prop->maxThreadsPerBlock = 1024;
    prop->warpSize = 64;
    prop->pciBusID = 1;
    prop->pciDeviceID = 0;
    prop->arch = 803; // RDNA2
    return hipSuccess;
}
inline hipError_t hipSetDevice(int deviceId) { return hipSuccess; }
inline hipError_t hipGetDevice(int* deviceId) { *deviceId = 0; return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipStreamCreate(hipStream_t* stream) { *stream = nullptr; return hipSuccess; }
inline hipError_t hipStreamDestroy(hipStream_t stream) { return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t stream) { return hipSuccess; }
inline hipError_t hipMemcpy(void* dst, const void* src, size_t size, int kind) { return hipSuccess; }
inline hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size, int kind, hipStream_t stream) { return hipSuccess; }
inline hipError_t hipMalloc(void** ptr, size_t size) { *ptr = malloc(size); return hipSuccess; }
inline hipError_t hipFree(void* ptr) { free(ptr); return hipSuccess; }
inline hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) { *ptr = malloc(size); return hipSuccess; }
inline hipError_t hipHostFree(void* ptr) { free(ptr); return hipSuccess; }
inline hipError_t hipMallocManaged(void** ptr, size_t size) { *ptr = malloc(size); return hipSuccess; }
inline hipError_t hipMemset(void* ptr, int value, size_t size) { memset(ptr, value, size); return hipSuccess; }
inline hipError_t hipMemsetAsync(void* ptr, int value, size_t size, hipStream_t stream) { memset(ptr, value, size); return hipSuccess; }
inline hipError_t hipMemGetInfo(size_t* free, size_t* total) { 
    *free = 16ULL * 1024 * 1024 * 1024; 
    *total = 16ULL * 1024 * 1024 * 1024;
    return hipSuccess; 
}
inline hipError_t hipRuntimeGetVersion(int* version) { *version = 60000; return hipSuccess; }
inline hipError_t hipDriverGetVersion(int* version) { *version = 60000; return hipSuccess; }
inline hipError_t hipPointerGetAttributes(void* attributes, const void* ptr) { 
    // Simplified implementation
    return hipSuccess; 
}

// MIOpen and rocBLAS stubs (simplified)
typedef void* miopenHandle_t;
typedef void* rocblas_handle;

inline int miopenCreate(miopenHandle_t* handle) { *handle = nullptr; return 0; }
inline int miopenDestroy(miopenHandle_t handle) { return 0; }

inline int rocblas_create_handle(rocblas_handle* handle) { *handle = nullptr; return 0; }
inline int rocblas_destroy_handle(rocblas_handle handle) { return 0; }

#endif // RDNA_HIP_STUB_H