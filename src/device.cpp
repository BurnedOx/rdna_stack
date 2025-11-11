#include "rdna/device.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>

namespace rdna {

// DeviceProperties implementation
DeviceProperties::DeviceProperties()
    : device_id(-1), total_memory(0), free_memory(0), compute_units(0),
      max_workgroup_size(0), wavefront_size(64), supports_fp16(false),
      supports_bf16(false), supports_tensor_cores(false),
      pci_bus_id(0), pci_device_id(0) {}

// DeviceManager implementation
DeviceManager& DeviceManager::get_instance() {
    static DeviceManager instance;
    return instance;
}

int DeviceManager::device_count() {
    int count = 0;
    hipError_t result = hipGetDeviceCount(&count);
    if (result != hipSuccess) {
        last_error_ = "Failed to get device count: " + std::string(hipGetErrorString(result));
        return 0;
    }
    return count;
}

DeviceProperties DeviceManager::get_device_properties(int device_id) {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }
    
    hipDeviceProp_t prop;
    hipError_t result = hipGetDeviceProperties(&prop, device_id);
    if (result != hipSuccess) {
        throw std::runtime_error("Failed to get device properties: " + 
                                std::string(hipGetErrorString(result)));
    }
    
    DeviceProperties props;
    props.device_id = device_id;
    props.name = prop.name;
    props.arch = prop.gcnArchName;
    props.total_memory = prop.totalGlobalMem;
    props.compute_units = prop.multiProcessorCount;
    props.max_workgroup_size = prop.maxThreadsPerBlock;
    props.wavefront_size = prop.warpSize;
    props.pci_bus_id = prop.pciBusID;
    props.pci_device_id = prop.pciDeviceID;
    
    // Check feature support
    props.supports_fp16 = prop.arch >= 803; // RDNA2+ supports FP16
    props.supports_bf16 = prop.arch >= 900; // RDNA3+ supports BF16
    props.supports_tensor_cores = false; // RDNA doesn't have tensor cores like NVIDIA
    
    // Get free memory
    size_t free, total;
    result = hipMemGetInfo(&free, &total);
    if (result == hipSuccess) {
        props.free_memory = free;
    }
    
    return props;
}

std::vector<DeviceProperties> DeviceManager::get_all_device_properties() {
    int count = device_count();
    std::vector<DeviceProperties> devices;
    devices.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        try {
            devices.push_back(get_device_properties(i));
        } catch (const std::exception& e) {
            std::cerr << "Error getting properties for device " << i << ": " << e.what() << std::endl;
        }
    }
    
    return devices;
}

std::shared_ptr<DeviceContext> DeviceManager::create_context(int device_id) {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }
    
    auto context = std::make_shared<DeviceContext>(device_id);
    if (!context->initialize()) {
        throw std::runtime_error("Failed to initialize device context");
    }
    
    return context;
}

std::shared_ptr<DeviceContext> DeviceManager::get_current_context() {
    return current_context_;
}

void DeviceManager::set_current_context(std::shared_ptr<DeviceContext> context) {
    current_context_ = context;
}

bool DeviceManager::check_device_compatibility(int device_id) {
    if (device_id < 0 || device_id >= device_count()) {
        return false;
    }
    
    try {
        auto props = get_device_properties(device_id);
        // Check if device supports required features
        return props.supports_fp16 && props.compute_units >= 4;
    } catch (...) {
        return false;
    }
}

std::string DeviceManager::get_last_error() {
    return last_error_;
}

// DeviceContext implementation
DeviceContext::DeviceContext(int device_id)
    : device_id_(device_id), hip_context_(nullptr), initialized_(false) {}

DeviceContext::~DeviceContext() {
    if (initialized_) {
        // HIP contexts are managed by the runtime, no explicit destruction needed
    }
}

bool DeviceContext::initialize() {
    hipError_t result = hipSetDevice(device_id_);
    if (result != hipSuccess) {
        return false;
    }
    
    // Create default stream
    default_stream_ = std::make_shared<Stream>(shared_from_this());
    if (!default_stream_->initialize()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

void DeviceContext::synchronize() {
    if (initialized_) {
        hipError_t result = hipDeviceSynchronize();
        if (result != hipSuccess) {
            throw std::runtime_error("Failed to synchronize device: " + 
                                    std::string(hipGetErrorString(result)));
        }
    }
}

bool DeviceContext::is_valid() const {
    return initialized_;
}

int DeviceContext::get_device_id() const {
    return device_id_;
}

DeviceProperties DeviceContext::get_properties() const {
    DeviceManager& manager = DeviceManager::get_instance();
    return manager.get_device_properties(device_id_);
}

std::shared_ptr<Stream> DeviceContext::create_stream() {
    auto stream = std::make_shared<Stream>(shared_from_this());
    if (!stream->initialize()) {
        throw std::runtime_error("Failed to create stream");
    }
    return stream;
}

std::shared_ptr<Stream> DeviceContext::get_default_stream() {
    return default_stream_;
}

// Stream implementation
Stream::Stream(std::shared_ptr<DeviceContext> context)
    : context_(context), hip_stream_(nullptr), initialized_(false) {}

Stream::~Stream() {
    if (initialized_ && hip_stream_) {
        hipError_t result = hipStreamDestroy(static_cast<hipStream_t>(hip_stream_));
        if (result != hipSuccess) {
            std::cerr << "Warning: Failed to destroy stream: " << hipGetErrorString(result) << std::endl;
        }
    }
}

bool Stream::initialize() {
    hipError_t result = hipStreamCreate(&hip_stream_);
    if (result != hipSuccess) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

void Stream::synchronize() {
    if (initialized_ && hip_stream_) {
        hipError_t result = hipStreamSynchronize(static_cast<hipStream_t>(hip_stream_));
        if (result != hipSuccess) {
            throw std::runtime_error("Failed to synchronize stream: " + 
                                    std::string(hipGetErrorString(result)));
        }
    }
}

bool Stream::is_valid() const {
    return initialized_ && hip_stream_ != nullptr;
}

void* Stream::get_native_handle() const {
    return hip_stream_;
}

bool Stream::memcpy(void* dst, const void* src, size_t size) {
    hipError_t result = hipMemcpy(dst, src, size, hipMemcpyDefault);
    return result == hipSuccess;
}

bool Stream::memcpy_async(void* dst, const void* src, size_t size) {
    if (!initialized_ || !hip_stream_) {
        return false;
    }
    
    hipError_t result = hipMemcpyAsync(dst, src, size, hipMemcpyDefault, 
                                      static_cast<hipStream_t>(hip_stream_));
    return result == hipSuccess;
}

// Utility functions
bool is_rdna_supported() {
    int count = 0;
    hipError_t result = hipGetDeviceCount(&count);
    if (result != hipSuccess || count == 0) {
        return false;
    }
    
    // Check if any device is RDNA architecture
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop;
        result = hipGetDeviceProperties(&prop, i);
        if (result == hipSuccess) {
            // RDNA architectures start with "gfx10" (RDNA1/2/3)
            std::string arch(prop.gcnArchName);
            if (arch.find("gfx10") != std::string::npos) {
                return true;
            }
        }
    }
    
    return false;
}

std::string get_roc_version() {
    int version = 0;
    hipError_t result = hipRuntimeGetVersion(&version);
    if (result != hipSuccess) {
        return "Unknown";
    }
    
    return std::to_string(version / 1000000) + "." +
           std::to_string((version % 1000000) / 1000) + "." +
           std::to_string(version % 1000);
}

std::string get_hip_version() {
    int version = 0;
    hipError_t result = hipDriverGetVersion(&version);
    if (result != hipSuccess) {
        return "Unknown";
    }
    
    return std::to_string(version / 1000) + "." +
           std::to_string(version % 1000);
}

} // namespace rdna