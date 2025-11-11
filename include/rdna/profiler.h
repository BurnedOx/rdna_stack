#ifndef RDNA_PROFILER_H
#define RDNA_PROFILER_H

#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>

namespace rdna {

// Performance event types
enum class EventType {
    KERNEL_LAUNCH,
    MEMORY_ALLOCATION,
    MEMORY_COPY,
    MEMORY_SET,
    STREAM_SYNCHRONIZE,
    DEVICE_SYNCHRONIZE
};

// Performance event record
struct PerformanceEvent {
    EventType type;
    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    size_t bytes_processed;
    int device_id;
    void* stream;
    std::string additional_info;
    
    double duration_ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0;
    }
};

// Performance statistics
struct PerformanceStats {
    double total_time_ms;
    double average_time_ms;
    double min_time_ms;
    double max_time_ms;
    size_t call_count;
    size_t total_bytes_processed;
    double throughput_gbps;
};

// Profiler configuration
struct ProfilerConfig {
    bool enable_timing;
    bool enable_memory_tracking;
    bool enable_kernel_tracking;
    size_t max_events;
    std::string output_file;
    
    ProfilerConfig() 
        : enable_timing(true),
          enable_memory_tracking(true),
          enable_kernel_tracking(true),
          max_events(10000) {}
};

// Performance profiler
class PerformanceProfiler {
public:
    static PerformanceProfiler& get_instance();
    
    // Configuration
    void set_config(const ProfilerConfig& config);
    ProfilerConfig get_config() const;
    
    // Event recording
    void start_event(EventType type, const std::string& name, 
                     size_t bytes = 0, const std::string& info = "");
    void end_event(EventType type, const std::string& name);
    
    // Memory tracking
    void record_memory_allocation(size_t size, void* ptr, int device_id);
    void record_memory_deallocation(void* ptr);
    void record_memory_copy(size_t size, void* src, void* dst, int device_id);
    
    // Kernel tracking
    void record_kernel_launch(const std::string& kernel_name, 
                              size_t grid_size[3], size_t block_size[3],
                              size_t shared_memory, int device_id);
    
    // Statistics
    PerformanceStats get_stats(EventType type, const std::string& name = "") const;
    std::unordered_map<std::string, PerformanceStats> get_all_stats() const;
    
    // Reporting
    void generate_report(const std::string& filename = "");
    void print_summary();
    void clear_events();
    
    // Utility functions
    bool is_enabled() const { return config_.enable_timing; }
    size_t get_event_count() const { return events_.size(); }
    
private:
    PerformanceProfiler() = default;
    ~PerformanceProfiler() = default;
    
    ProfilerConfig config_;
    std::vector<PerformanceEvent> events_;
    std::unordered_map<void*, size_t> memory_allocations_;
    mutable std::mutex mutex_;
    
    // Current event tracking
    std::unordered_map<std::string, PerformanceEvent> active_events_;
};

// Automatic event timing (RAII)
class ScopedEvent {
public:
    ScopedEvent(EventType type, const std::string& name, 
                size_t bytes = 0, const std::string& info = "")
        : type_(type), name_(name) {
        PerformanceProfiler::get_instance().start_event(type, name, bytes, info);
    }
    
    ~ScopedEvent() {
        PerformanceProfiler::get_instance().end_event(type_, name_);
    }
    
private:
    EventType type_;
    std::string name_;
};

// Convenience macros for profiling
#ifdef RDNA_PROFILING_ENABLED
    #define RDNA_PROFILE_SCOPE(name) \
        rdna::ScopedEvent scoped_event_##__LINE__(rdna::EventType::KERNEL_LAUNCH, name)
    
    #define RDNA_PROFILE_FUNCTION() \
        RDNA_PROFILE_SCOPE(__FUNCTION__)
    
    #define RDNA_PROFILE_START(event_type, name) \
        rdna::PerformanceProfiler::get_instance().start_event(event_type, name)
    
    #define RDNA_PROFILE_END(event_type, name) \
        rdna::PerformanceProfiler::get_instance().end_event(event_type, name)
    
    #define RDNA_PROFILE_MEMORY_ALLOC(size, ptr, device) \
        rdna::PerformanceProfiler::get_instance().record_memory_allocation(size, ptr, device)
    
    #define RDNA_PROFILE_MEMORY_FREE(ptr) \
        rdna::PerformanceProfiler::get_instance().record_memory_deallocation(ptr)
#else
    #define RDNA_PROFILE_SCOPE(name)
    #define RDNA_PROFILE_FUNCTION()
    #define RDNA_PROFILE_START(event_type, name)
    #define RDNA_PROFILE_END(event_type, name)
    #define RDNA_PROFILE_MEMORY_ALLOC(size, ptr, device)
    #define RDNA_PROFILE_MEMORY_FREE(ptr)
#endif

// Performance optimization utilities
class PerformanceOptimizer {
public:
    static PerformanceOptimizer& get_instance();
    
    // Kernel optimization
    void optimize_kernel_config(const std::string& kernel_name,
                               size_t* grid_size, size_t* block_size,
                               size_t* shared_memory, int device_id);
    
    // Memory optimization
    void suggest_memory_layout(const std::vector<size_t>& shape,
                              std::vector<size_t>* optimal_strides);
    
    // Algorithm selection
    std::string select_best_algorithm(const std::string& operation_type,
                                     const std::vector<std::string>& available_algorithms,
                                     int device_id);
    
    // Cache optimization
    void optimize_cache_behavior(size_t working_set_size, int device_id);
    
    // Performance tuning
    void tune_parameters(const std::string& operation_type, int device_id);
    
private:
    PerformanceOptimizer() = default;
    ~PerformanceOptimizer() = default;
    
    std::unordered_map<std::string, std::unordered_map<int, std::string>> algorithm_cache_;
    std::unordered_map<std::string, std::unordered_map<int, size_t>> optimal_configs_;
};

// Benchmarking utilities
class BenchmarkRunner {
public:
    static BenchmarkRunner& get_instance();
    
    // Run benchmark suite
    void run_benchmarks(int device_id);
    
    // Individual benchmarks
    double benchmark_memory_bandwidth(int device_id, size_t size);
    double benchmark_kernel_latency(const std::string& kernel_name, int device_id);
    double benchmark_matrix_multiply(int m, int n, int k, int device_id);
    double benchmark_convolution(int batch, int height, int width, int channels,
                                int filters, int kernel_size, int device_id);
    
    // Compare against baseline
    void compare_with_baseline(const std::string& operation, double rdna_time,
                              double baseline_time, const std::string& baseline_name);
    
    // Generate performance report
    void generate_benchmark_report(const std::string& filename);
    
private:
    BenchmarkRunner() = default;
    ~BenchmarkRunner() = default;
};

} // namespace rdna

#endif // RDNA_PROFILER_H