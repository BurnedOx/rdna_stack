#ifndef RDNA_UTILS_H
#define RDNA_UTILS_H

#include <string>
#include <memory>

namespace rdna {

// Use HIP stub for development without ROCm installation
#include "rdna/hip_stub.h"

// Error handling utilities
std::string get_error_string(hipError_t error);
std::string get_last_hip_error();
void check_hip_error(hipError_t error, const std::string& context);

// Logging utilities
void log_info(const std::string& message);
void log_warning(const std::string& message);
void log_error(const std::string& message);
void log_debug(const std::string& message);

// Performance timing utilities
std::unique_ptr<void> create_timer(const std::string& name);

// Memory utilities
size_t calculate_aligned_size(size_t size, size_t alignment);
bool is_aligned(const void* ptr, size_t alignment);
void* align_pointer(void* ptr, size_t alignment);

// Device capability checking
bool check_device_capability(int device_id, const std::string& capability);

// Version information
std::string get_library_version();
std::string get_build_info();

// Configuration structure
struct LibraryConfig {
    bool enable_debug_logging;
    bool enable_profiling;
    size_t memory_cache_limit;
    bool use_unified_memory;
};

// Configuration management
LibraryConfig get_library_config();
void set_library_config(const LibraryConfig& config);
void set_debug_logging(bool enabled);
void set_profiling(bool enabled);
void set_memory_cache_limit(size_t limit);

// Diagnostic utilities
std::string get_system_info();
void print_system_info();
std::string get_memory_info(int device_id = -1);
void print_memory_info(int device_id = -1);
std::string get_kernel_info(int device_id = -1);
void print_kernel_info(int device_id = -1);
void run_diagnostics();

} // namespace rdna

#endif // RDNA_UTILS_H