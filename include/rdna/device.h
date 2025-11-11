#ifndef RDNA_DEVICE_H
#define RDNA_DEVICE_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace rdna {

// Forward declarations
class DeviceContext;
class Stream;

/**
 * @brief Device properties structure
 * 
 * Contains information about RDNA device capabilities
 */
struct DeviceProperties {
    int device_id;
    std::string name;
    std::string arch;
    uint64_t total_memory;
    uint64_t free_memory;
    int compute_units;
    int max_workgroup_size;
    int wavefront_size;
    bool supports_fp16;
    bool supports_bf16;
    bool supports_tensor_cores;
    int pci_bus_id;
    int pci_device_id;
    
    DeviceProperties();
};

/**
 * @brief Device discovery and management
 * 
 * Handles device enumeration, context creation, and device properties
 */
class DeviceManager {
public:
    static DeviceManager& get_instance();
    
    // Device discovery
    int device_count();
    DeviceProperties get_device_properties(int device_id);
    std::vector<DeviceProperties> get_all_device_properties();
    
    // Context management
    std::shared_ptr<DeviceContext> create_context(int device_id);
    std::shared_ptr<DeviceContext> get_current_context();
    void set_current_context(std::shared_ptr<DeviceContext> context);
    
    // Error handling
    bool check_device_compatibility(int device_id);
    std::string get_last_error();
    
private:
    DeviceManager() = default;
    ~DeviceManager() = default;
    
    std::vector<DeviceProperties> devices_;
    std::shared_ptr<DeviceContext> current_context_;
    std::string last_error_;
};

/**
 * @brief Device context wrapper
 * 
 * Manages HIP device context and provides device-specific operations
 */
class DeviceContext {
public:
    DeviceContext(int device_id);
    ~DeviceContext();
    
    bool initialize();
    void synchronize();
    bool is_valid() const;
    int get_device_id() const;
    DeviceProperties get_properties() const;
    
    // Stream management
    std::shared_ptr<Stream> create_stream();
    std::shared_ptr<Stream> get_default_stream();
    
private:
    int device_id_;
    void* hip_context_;
    std::shared_ptr<Stream> default_stream_;
    bool initialized_;
};

/**
 * @brief Stream/queue abstraction
 * 
 * Provides asynchronous execution and synchronization
 */
class Stream {
public:
    Stream(std::shared_ptr<DeviceContext> context);
    ~Stream();
    
    bool initialize();
    void synchronize();
    bool is_valid() const;
    void* get_native_handle() const;
    
    // Memory operations
    bool memcpy(void* dst, const void* src, size_t size);
    bool memcpy_async(void* dst, const void* src, size_t size);
    
private:
    std::shared_ptr<DeviceContext> context_;
    void* hip_stream_;
    bool initialized_;
};

// Utility functions
bool is_rdna_supported();
std::string get_roc_version();
std::string get_hip_version();

} // namespace rdna

#endif // RDNA_DEVICE_H