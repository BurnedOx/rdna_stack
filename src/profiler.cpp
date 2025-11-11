#include "rdna/profiler.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace rdna {

// PerformanceProfiler implementation
PerformanceProfiler& PerformanceProfiler::get_instance() {
    static PerformanceProfiler instance;
    return instance;
}

void PerformanceProfiler::set_config(const ProfilerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

ProfilerConfig PerformanceProfiler::get_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void PerformanceProfiler::start_event(EventType type, const std::string& name, 
                                     size_t bytes, const std::string& info) {
    if (!config_.enable_timing) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    PerformanceEvent event;
    event.type = type;
    event.name = name;
    event.start_time = std::chrono::high_resolution_clock::now();
    event.bytes_processed = bytes;
    event.additional_info = info;
    event.device_id = -1; // Will be set by context
    
    active_events_[name] = event;
}

void PerformanceProfiler::end_event(EventType type, const std::string& name) {
    if (!config_.enable_timing) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = active_events_.find(name);
    if (it != active_events_.end()) {
        PerformanceEvent& event = it->second;
        event.end_time = std::chrono::high_resolution_clock::now();
        events_.push_back(event);
        active_events_.erase(it);
        
        // Limit event storage
        if (events_.size() > config_.max_events) {
            events_.erase(events_.begin());
        }
    }
}

void PerformanceProfiler::record_memory_allocation(size_t size, void* ptr, int device_id) {
    if (!config_.enable_memory_tracking) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    memory_allocations_[ptr] = size;
}

void PerformanceProfiler::record_memory_deallocation(void* ptr) {
    if (!config_.enable_memory_tracking) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    memory_allocations_.erase(ptr);
}

void PerformanceProfiler::record_memory_copy(size_t size, void* src, void* dst, int device_id) {
    if (!config_.enable_memory_tracking) return;
    
    std::string name = "memcpy";
    start_event(EventType::MEMORY_COPY, name, size);
    // Note: We assume the copy is synchronous for profiling
    end_event(EventType::MEMORY_COPY, name);
}

void PerformanceProfiler::record_kernel_launch(const std::string& kernel_name, 
                                              size_t grid_size[3], size_t block_size[3],
                                              size_t shared_memory, int device_id) {
    if (!config_.enable_kernel_tracking) return;
    
    std::stringstream ss;
    ss << kernel_name << " [" << grid_size[0] << "," << grid_size[1] << "," << grid_size[2] << "]";
    start_event(EventType::KERNEL_LAUNCH, ss.str(), 0, 
                "Grid: " + std::to_string(grid_size[0]) + "x" + std::to_string(grid_size[1]) + "x" + std::to_string(grid_size[2]) +
                " Block: " + std::to_string(block_size[0]) + "x" + std::to_string(block_size[1]) + "x" + std::to_string(block_size[2]));
    // Note: We assume kernel completion is synchronous for profiling
    end_event(EventType::KERNEL_LAUNCH, ss.str());
}

PerformanceStats PerformanceProfiler::get_stats(EventType type, const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    PerformanceStats stats = {0, 0, std::numeric_limits<double>::max(), 0, 0, 0, 0};
    std::vector<double> durations;
    
    for (const auto& event : events_) {
        if (event.type == type && (name.empty() || event.name == name)) {
            double duration = event.duration_ms();
            durations.push_back(duration);
            stats.total_time_ms += duration;
            stats.total_bytes_processed += event.bytes_processed;
            stats.min_time_ms = std::min(stats.min_time_ms, duration);
            stats.max_time_ms = std::max(stats.max_time_ms, duration);
        }
    }
    
    stats.call_count = durations.size();
    if (stats.call_count > 0) {
        stats.average_time_ms = stats.total_time_ms / stats.call_count;
        if (stats.total_time_ms > 0) {
            stats.throughput_gbps = (stats.total_bytes_processed * 8.0) / (stats.total_time_ms * 1e6); // Gbps
        }
    }
    
    return stats;
}

std::unordered_map<std::string, PerformanceStats> PerformanceProfiler::get_all_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, PerformanceStats> result;
    
    // Group events by name
    std::unordered_map<std::string, std::vector<const PerformanceEvent*>> events_by_name;
    for (const auto& event : events_) {
        events_by_name[event.name].push_back(&event);
    }
    
    // Calculate stats for each event name
    for (const auto& pair : events_by_name) {
        PerformanceStats stats = {0, 0, std::numeric_limits<double>::max(), 0, 0, 0, 0};
        std::vector<double> durations;
        
        for (const auto& event : pair.second) {
            double duration = event->duration_ms();
            durations.push_back(duration);
            stats.total_time_ms += duration;
            stats.total_bytes_processed += event->bytes_processed;
            stats.min_time_ms = std::min(stats.min_time_ms, duration);
            stats.max_time_ms = std::max(stats.max_time_ms, duration);
        }
        
        stats.call_count = durations.size();
        if (stats.call_count > 0) {
            stats.average_time_ms = stats.total_time_ms / stats.call_count;
            if (stats.total_time_ms > 0) {
                stats.throughput_gbps = (stats.total_bytes_processed * 8.0) / (stats.total_time_ms * 1e6);
            }
        }
        
        result[pair.first] = stats;
    }
    
    return result;
}

void PerformanceProfiler::generate_report(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostream* output = &std::cout;
    std::ofstream file;
    
    if (!filename.empty()) {
        file.open(filename);
        if (file.is_open()) {
            output = &file;
        }
    }
    
    *output << "RDNA Performance Report\n";
    *output << "=======================\n\n";
    *output << "Total events recorded: " << events_.size() << "\n\n";
    
    // Group by event type and name
    auto all_stats = get_all_stats();
    
    for (const auto& pair : all_stats) {
        const auto& stats = pair.second;
        *output << "Event: " << pair.first << "\n";
        *output << "  Calls: " << stats.call_count << "\n";
        *output << "  Total time: " << std::fixed << std::setprecision(3) << stats.total_time_ms << " ms\n";
        *output << "  Average time: " << stats.average_time_ms << " ms\n";
        *output << "  Min time: " << stats.min_time_ms << " ms\n";
        *output << "  Max time: " << stats.max_time_ms << " ms\n";
        if (stats.total_bytes_processed > 0) {
            *output << "  Throughput: " << stats.throughput_gbps << " Gbps\n";
        }
        *output << "\n";
    }
    
    // Memory allocation summary
    if (config_.enable_memory_tracking) {
        *output << "Memory Allocations:\n";
        size_t total_allocated = 0;
        for (const auto& pair : memory_allocations_) {
            total_allocated += pair.second;
        }
        *output << "  Active allocations: " << memory_allocations_.size() << "\n";
        *output << "  Total allocated: " << total_allocated / (1024.0 * 1024.0) << " MB\n\n";
    }
    
    if (file.is_open()) {
        file.close();
    }
}

void PerformanceProfiler::print_summary() {
    generate_report(""); // Print to console
}

void PerformanceProfiler::clear_events() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
    active_events_.clear();
    memory_allocations_.clear();
}

// PerformanceOptimizer implementation
PerformanceOptimizer& PerformanceOptimizer::get_instance() {
    static PerformanceOptimizer instance;
    return instance;
}

void PerformanceOptimizer::optimize_kernel_config(const std::string& kernel_name,
                                                size_t* grid_size, size_t* block_size,
                                                size_t* shared_memory, int device_id) {
    // Simple optimization rules for RDNA architecture
    if (kernel_name.find("matmul") != std::string::npos) {
        // Optimize for matrix multiplication
        block_size[0] = 16;
        block_size[1] = 16;
        block_size[2] = 1;
        
        // Adjust grid size based on problem size
        grid_size[0] = (grid_size[0] + block_size[0] - 1) / block_size[0];
        grid_size[1] = (grid_size[1] + block_size[1] - 1) / block_size[1];
        grid_size[2] = 1;
        
    } else if (kernel_name.find("conv") != std::string::npos) {
        // Optimize for convolution
        block_size[0] = 8;
        block_size[1] = 8;
        block_size[2] = 4;
        
        grid_size[0] = (grid_size[0] + block_size[0] - 1) / block_size[0];
        grid_size[1] = (grid_size[1] + block_size[1] - 1) / block_size[1];
        grid_size[2] = (grid_size[2] + block_size[2] - 1) / block_size[2];
    }
    
    // Cache the optimal configuration
    std::string key = kernel_name + "_" + std::to_string(device_id);
    optimal_configs_[key][device_id] = block_size[0] * block_size[1] * block_size[2];
}

void PerformanceOptimizer::suggest_memory_layout(const std::vector<size_t>& shape,
                                                std::vector<size_t>* optimal_strides) {
    // Suggest optimal memory layout for RDNA
    optimal_strides->resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        (*optimal_strides)[i] = stride;
        stride *= shape[i];
    }
}

std::string PerformanceOptimizer::select_best_algorithm(const std::string& operation_type,
                                                      const std::vector<std::string>& available_algorithms,
                                                      int device_id) {
    std::string key = operation_type + "_" + std::to_string(device_id);
    
    // Return cached algorithm if available
    auto it = algorithm_cache_.find(key);
    if (it != algorithm_cache_.end()) {
        auto device_it = it->second.find(device_id);
        if (device_it != it->second.end()) {
            return device_it->second;
        }
    }
    
    // Simple algorithm selection
    std::string best_algorithm = "DEFAULT";
    if (!available_algorithms.empty()) {
        best_algorithm = available_algorithms[0];
        
        // Prefer algorithms with "fast" or "optimized" in the name
        for (const auto& alg : available_algorithms) {
            if (alg.find("fast") != std::string::npos || 
                alg.find("optimized") != std::string::npos) {
                best_algorithm = alg;
                break;
            }
        }
    }
    
    // Cache the selection
    algorithm_cache_[key][device_id] = best_algorithm;
    return best_algorithm;
}

void PerformanceOptimizer::optimize_cache_behavior(size_t working_set_size, int device_id) {
    // Simple cache optimization suggestions
    if (working_set_size > 1024 * 1024 * 1024) { // > 1GB
        // Suggest using unified memory for large working sets
        std::cout << "Consider using unified memory for working set > 1GB" << std::endl;
    }
}

void PerformanceOptimizer::tune_parameters(const std::string& operation_type, int device_id) {
    // Parameter tuning based on operation type and device
    std::cout << "Tuning parameters for " << operation_type << " on device " << device_id << std::endl;
    
    if (operation_type == "matmul") {
        std::cout << "Suggested tuning: Use tile size 16x16 for better cache utilization" << std::endl;
    } else if (operation_type == "convolution") {
        std::cout << "Suggested tuning: Use winograd algorithm for 3x3 convolutions" << std::endl;
    }
}

// BenchmarkRunner implementation
BenchmarkRunner& BenchmarkRunner::get_instance() {
    static BenchmarkRunner instance;
    return instance;
}

void BenchmarkRunner::run_benchmarks(int device_id) {
    std::cout << "Running RDNA benchmarks on device " << device_id << std::endl;
    
    // Run individual benchmarks
    double mem_bw = benchmark_memory_bandwidth(device_id, 1024 * 1024 * 1024); // 1GB
    double matmul_time = benchmark_matrix_multiply(1024, 1024, 1024, device_id);
    double conv_time = benchmark_convolution(32, 224, 224, 64, 64, 3, device_id);
    
    std::cout << "Benchmark results:" << std::endl;
    std::cout << "Memory bandwidth: " << mem_bw << " GB/s" << std::endl;
    std::cout << "1024x1024 matmul: " << matmul_time << " ms" << std::endl;
    std::cout << "32x224x224 conv: " << conv_time << " ms" << std::endl;
}

double BenchmarkRunner::benchmark_memory_bandwidth(int device_id, size_t size) {
    // Simple memory bandwidth benchmark
    PerformanceProfiler& profiler = PerformanceProfiler::get_instance();
    profiler.start_event(EventType::MEMORY_COPY, "memory_bandwidth", size);
    
    // Simulate memory operations
    std::vector<char> buffer(size);
    std::fill(buffer.begin(), buffer.end(), 1);
    
    profiler.end_event(EventType::MEMORY_COPY, "memory_bandwidth");
    
    auto stats = profiler.get_stats(EventType::MEMORY_COPY, "memory_bandwidth");
    return (size / (1024.0 * 1024.0 * 1024.0)) / (stats.total_time_ms / 1000.0); // GB/s
}

double BenchmarkRunner::benchmark_kernel_latency(const std::string& kernel_name, int device_id) {
    PerformanceProfiler& profiler = PerformanceProfiler::get_instance();
    profiler.start_event(EventType::KERNEL_LAUNCH, kernel_name);
    
    // Simulate kernel execution
    // In real implementation, this would launch an actual kernel
    
    profiler.end_event(EventType::KERNEL_LAUNCH, kernel_name);
    
    auto stats = profiler.get_stats(EventType::KERNEL_LAUNCH, kernel_name);
    return stats.average_time_ms;
}

double BenchmarkRunner::benchmark_matrix_multiply(int m, int n, int k, int device_id) {
    PerformanceProfiler& profiler = PerformanceProfiler::get_instance();
    std::string name = "matmul_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    profiler.start_event(EventType::KERNEL_LAUNCH, name);
    
    // Simulate matrix multiplication
    // In real implementation, this would use actual matmul kernel
    
    profiler.end_event(EventType::KERNEL_LAUNCH, name);
    
    auto stats = profiler.get_stats(EventType::KERNEL_LAUNCH, name);
    return stats.average_time_ms;
}

double BenchmarkRunner::benchmark_convolution(int batch, int height, int width, int channels,
                                            int filters, int kernel_size, int device_id) {
    PerformanceProfiler& profiler = PerformanceProfiler::get_instance();
    std::string name = "conv2d_b" + std::to_string(batch) + "_c" + std::to_string(channels) + 
                       "_f" + std::to_string(filters);
    profiler.start_event(EventType::KERNEL_LAUNCH, name);
    
    // Simulate convolution
    // In real implementation, this would use actual conv kernel
    
    profiler.end_event(EventType::KERNEL_LAUNCH, name);
    
    auto stats = profiler.get_stats(EventType::KERNEL_LAUNCH, name);
    return stats.average_time_ms;
}

void BenchmarkRunner::compare_with_baseline(const std::string& operation, double rdna_time,
                                           double baseline_time, const std::string& baseline_name) {
    double speedup = baseline_time / rdna_time;
    std::cout << operation << " performance:" << std::endl;
    std::cout << "  RDNA: " << rdna_time << " ms" << std::endl;
    std::cout << "  " << baseline_name << ": " << baseline_time << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    if (speedup > 1.0) {
        std::cout << "  RDNA is " << (speedup - 1.0) * 100 << "% faster" << std::endl;
    } else {
        std::cout << "  RDNA is " << (1.0 - speedup) * 100 << "% slower" << std::endl;
    }
}

void BenchmarkRunner::generate_benchmark_report(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "RDNA Benchmark Report\n";
        file << "=====================\n\n";
        file << "Generated on: " << __DATE__ << " " << __TIME__ << "\n\n";
        
        // This would include actual benchmark results
        file << "Benchmark suite completed.\n";
        file.close();
    }
}

} // namespace rdna