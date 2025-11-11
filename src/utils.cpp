#include "rdna/device.h"
#include "rdna/memory.h"
#include "rdna/kernels.h"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace rdna {

// Error handling utilities
std::string get_error_string(hipError_t error) {
#ifdef __HIP_PLATFORM_AMD__
    return hipGetErrorString(error);
#else
    return "HIP runtime not available (development mode)";
#endif
}

std::string get_last_hip_error() {
#ifdef __HIP_PLATFORM_AMD__
    hipError_t error = hipGetLastError();
    return get_error_string(error);
#else
    return "HIP runtime not available (development mode)";
#endif
}

void check_hip_error(hipError_t error, const std::string& context) {
#ifdef __HIP_PLATFORM_AMD__
    if (error != hipSuccess) {
        std::stringstream ss;
        ss << "HIP error in " << context << ": " << get_error_string(error);
        throw std::runtime_error(ss.str());
    }
#else
    // In development mode without ROCm, we'll just log the error
    if (error != 0) { // Using 0 as placeholder for hipSuccess
        std::cout << "[DEV MODE] HIP error simulation in " << context << std::endl;
    }
#endif
}

// Logging utilities
void log_info(const std::string& message) {
    std::cout << "[INFO] " << message << std::endl;
}

void log_warning(const std::string& message) {
    std::cout << "[WARNING] " << message << std::endl;
}

void log_error(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}

void log_debug(const std::string& message) {
#ifdef RDNA_DEBUG
    std::cout << "[DEBUG] " << message << std::endl;
#endif
}

// Performance timing utilities
#ifdef RDNA_PERF_TIMING
#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
    
public:
    Timer(const std::string& timer_name) : name(timer_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "[TIMER] " << name << ": " << duration.count() << " μs" << std::endl;
    }
};

std::unique_ptr<Timer> create_timer(const std::string& name) {
    return std::make_unique<Timer>(name);
}

#else
// Empty implementation when timing is disabled
std::unique_ptr<void> create_timer(const std::string& name) {
    return nullptr;
}
#endif

// Memory utilities
size_t calculate_aligned_size(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

void* align_pointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned_addr);
}

// Device capability checking
bool check_device_capability(int device_id, const std::string& capability) {
    try {
        DeviceManager& manager = DeviceManager::get_instance();
        DeviceProperties props = manager.get_device_properties(device_id);
        
        if (capability == "fp16") {
            return props.supports_fp16;
        } else if (capability == "bf16") {
            return props.supports_bf16;
        } else if (capability == "tensor_cores") {
            return props.supports_tensor_cores;
        } else if (capability == "unified_memory") {
            // Check if device supports unified memory
            // This is a simplified check - actual support may vary
            return props.total_memory > 0; // Most modern GPUs support it
        }
    } catch (const std::exception& e) {
        log_error("Failed to check device capability: " + std::string(e.what()));
        return false;
    }
    
    return false;
}

// Version information
std::string get_library_version() {
    return "0.1.0";
}

std::string get_build_info() {
    std::stringstream ss;
    ss << "RDNA Stack v" << get_library_version() << " built on " << __DATE__ << " " << __TIME__;
#ifdef RDNA_DEBUG
    ss << " (Debug)";
#else
    ss << " (Release)";
#endif
    return ss.str();
}

// Configuration utilities
struct LibraryConfig {
    bool enable_debug_logging = false;
    bool enable_profiling = false;
    size_t memory_cache_limit = 1024 * 1024 * 1024; // 1GB
    bool use_unified_memory = false;
};

class ConfigManager {
private:
    LibraryConfig config_;
    static ConfigManager* instance_;
    
    ConfigManager() = default;
    
public:
    static ConfigManager& get_instance() {
        if (!instance_) {
            instance_ = new ConfigManager();
        }
        return *instance_;
    }
    
    LibraryConfig get_config() const {
        return config_;
    }
    
    void set_config(const LibraryConfig& config) {
        config_ = config;
        
        // Apply configuration changes
        if (config_.memory_cache_limit > 0) {
            MemoryManager::get_instance().get_current_allocator()->set_cache_size_limit(config_.memory_cache_limit);
        }
    }
    
    void set_debug_logging(bool enabled) {
        config_.enable_debug_logging = enabled;
    }
    
    void set_profiling(bool enabled) {
        config_.enable_profiling = enabled;
    }
    
    void set_memory_cache_limit(size_t limit) {
        config_.memory_cache_limit = limit;
        MemoryManager::get_instance().get_current_allocator()->set_cache_size_limit(limit);
    }
};

ConfigManager* ConfigManager::instance_ = nullptr;

LibraryConfig get_library_config() {
    return ConfigManager::get_instance().get_config();
}

void set_library_config(const LibraryConfig& config) {
    ConfigManager::get_instance().set_config(config);
}

void set_debug_logging(bool enabled) {
    ConfigManager::get_instance().set_debug_logging(enabled);
}

void set_profiling(bool enabled) {
    ConfigManager::get_instance().set_profiling(enabled);
}

void set_memory_cache_limit(size_t limit) {
    ConfigManager::get_instance().set_memory_cache_limit(limit);
}

// Diagnostic utilities
std::string get_system_info() {
    std::stringstream ss;
    
    ss << "RDNA Stack System Information:" << std::endl;
    ss << "  Library Version: " << get_library_version() << std::endl;
    ss << "  ROCm Version: " << get_roc_version() << std::endl;
    ss << "  HIP Version: " << get_hip_version() << std::endl;
    ss << "  RDNA Supported: " << (is_rdna_supported() ? "Yes" : "No") << std::endl;
    
    DeviceManager& manager = DeviceManager::get_instance();
    int device_count = manager.device_count();
    ss << "  Device Count: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        try {
            DeviceProperties props = manager.get_device_properties(i);
            ss << "  Device " << i << ": " << props.name << std::endl;
            ss << "    Architecture: " << props.arch << std::endl;
            ss << "    Memory: " << (props.total_memory / (1024 * 1024)) << " MB" << std::endl;
            ss << "    Compute Units: " << props.compute_units << std::endl;
            ss << "    FP16 Support: " << (props.supports_fp16 ? "Yes" : "No") << std::endl;
            ss << "    BF16 Support: " << (props.supports_bf16 ? "Yes" : "No") << std::endl;
        } catch (const std::exception& e) {
            ss << "  Device " << i << ": Error - " << e.what() << std::endl;
        }
    }
    
    return ss.str();
}

void print_system_info() {
    std::cout << get_system_info() << std::endl;
}

// Memory diagnostic
std::string get_memory_info(int device_id) {
    std::stringstream ss;
    
    MemoryManager& manager = MemoryManager::get_instance();
    MemoryStats stats = manager.get_stats(device_id);
    
    ss << "Memory Information for Device " << device_id << ":" << std::endl;
    ss << "  Allocated: " << (stats.allocated_bytes / (1024 * 1024)) << " MB" << std::endl;
    ss << "  Allocated Blocks: " << stats.allocated_blocks << std::endl;
    ss << "  Cached: " << (stats.cached_bytes / (1024 * 1024)) << " MB" << std::endl;
    ss << "  Cached Blocks: " << stats.cached_blocks << std::endl;
    ss << "  Max Allocated: " << (stats.max_allocated_bytes / (1024 * 1024)) << " MB" << std::endl;
    ss << "  Total Allocations: " << stats.total_allocations << std::endl;
    ss << "  Total Frees: " << stats.total_frees << std::endl;
    
    uint64_t total_mem = manager.get_total_memory(device_id);
    uint64_t free_mem = manager.get_free_memory(device_id);
    uint64_t used_mem = manager.get_used_memory(device_id);
    
    ss << "  Total Device Memory: " << (total_mem / (1024 * 1024)) << " MB" << std::endl;
    ss << "  Free Device Memory: " << (free_mem / (1024 * 1024)) << " MB" << std::endl;
    ss << "  Used Device Memory: " << (used_mem / (1024 * 1024)) << " MB" << std::endl;
    
    return ss.str();
}

void print_memory_info(int device_id) {
    std::cout << get_memory_info(device_id) << std::endl;
}

// Kernel diagnostic
std::string get_kernel_info(int device_id) {
    std::stringstream ss;
    
    KernelManager& manager = KernelManager::get_instance();
    bool initialized = manager.are_kernels_initialized(device_id);
    
    ss << "Kernel Information for Device " << device_id << ":" << std::endl;
    ss << "  Kernels Initialized: " << (initialized ? "Yes" : "No") << std::endl;
    
    if (initialized) {
        auto matmul = manager.get_matmul_kernel(device_id);
        auto conv = manager.get_conv_kernel(device_id);
        auto custom = manager.get_custom_kernels(device_id);
        
        ss << "  Matmul Kernel: " << (matmul->is_initialized() ? "Ready" : "Not Ready") << std::endl;
        ss << "  Conv Kernel: " << (conv->is_initialized() ? "Ready" : "Not Ready") << std::endl;
        ss << "  Custom Kernels: " << (custom->is_initialized() ? "Ready" : "Not Ready") << std::endl;
    }
    
    return ss.str();
}

void print_kernel_info(int device_id) {
    std::cout << get_kernel_info(device_id) << std::endl;
}

// Comprehensive diagnostic
void run_diagnostics() {
    std::cout << "=== RDNA Stack Diagnostics ===" << std::endl;
    print_system_info();
    std::cout << std::endl;
    
    DeviceManager& device_manager = DeviceManager::get_instance();
    int device_count = device_manager.device_count();
    
    for (int i = 0; i < device_count; ++i) {
        print_memory_info(i);
        std::cout << std::endl;
        print_kernel_info(i);
        std::cout << std::endl;
    }
    
    std::cout << "=== Diagnostics Complete ===" << std::endl;
}

} // namespace rdna