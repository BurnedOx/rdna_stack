#ifndef RDNA_MEMORY_H
#define RDNA_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace rdna {

// Forward declarations
class DeviceContext;
class Stream;

/**
 * @brief Memory allocation information
 */
struct AllocationInfo {
    void* ptr;
    size_t size;
    size_t allocated_size;
    bool is_device_memory;
    int device_id;
    void* stream;
    uint64_t allocation_id;
};

/**
 * @brief Memory statistics
 */
struct MemoryStats {
    uint64_t allocated_bytes;
    uint64_t allocated_blocks;
    uint64_t cached_bytes;
    uint64_t cached_blocks;
    uint64_t max_allocated_bytes;
    uint64_t total_allocations;
    uint64_t total_frees;
};

/**
 * @brief Memory allocation options
 */
struct AllocationOptions {
    bool pinned_host_memory;
    bool unified_memory;
    bool managed_memory;
    size_t alignment;
    void* stream;
};

/**
 * @brief Caching memory allocator
 * 
 * Implements a caching allocator similar to PyTorch's CUDA allocator
 * with block splitting, caching, and memory reuse.
 */
class MemoryAllocator {
public:
    MemoryAllocator(std::shared_ptr<DeviceContext> context);
    ~MemoryAllocator();
    
    // Memory allocation/deallocation
    void* allocate(size_t size, const AllocationOptions& options = {});
    void deallocate(void* ptr);
    
    // Memory operations
    bool memcpy(void* dst, const void* src, size_t size, void* stream = nullptr);
    bool memset(void* ptr, int value, size_t size, void* stream = nullptr);
    
    // Memory management
    void empty_cache();
    MemoryStats get_stats() const;
    AllocationInfo get_allocation_info(void* ptr) const;
    
    // Cache management
    void set_cache_size_limit(size_t limit);
    size_t get_cache_size_limit() const;
    
    // Device memory info
    uint64_t get_total_memory() const;
    uint64_t get_free_memory() const;
    uint64_t get_used_memory() const;
    
private:
    struct Block {
        void* ptr;
        size_t size;
        size_t allocated_size;
        bool in_use;
        uint64_t allocation_id;
        void* stream;
    };
    
    struct CacheBlock {
        void* ptr;
        size_t size;
        void* stream;
        uint64_t last_used;
    };
    
    // Block management
    Block* find_free_block(size_t size);
    Block* allocate_new_block(size_t size, const AllocationOptions& options);
    void split_block(Block* block, size_t needed_size);
    void merge_adjacent_blocks();
    
    // Cache management
    void add_to_cache(Block* block);
    CacheBlock* find_cached_block(size_t size);
    void evict_from_cache(size_t needed_size);
    void cleanup_cache();
    
    std::shared_ptr<DeviceContext> context_;
    std::vector<std::unique_ptr<Block>> blocks_;
    std::vector<std::unique_ptr<CacheBlock>> cache_;
    mutable std::mutex mutex_;
    
    MemoryStats stats_;
    size_t cache_size_limit_;
    uint64_t allocation_counter_;
};

/**
 * @brief Memory manager singleton
 * 
 * Provides global memory management across devices
 */
class MemoryManager {
public:
    static MemoryManager& get_instance();
    
    // Device-specific allocators
    std::shared_ptr<MemoryAllocator> get_allocator(int device_id);
    std::shared_ptr<MemoryAllocator> get_current_allocator();
    
    // Global memory operations
    void* allocate(size_t size, int device_id = -1, const AllocationOptions& options = {});
    void deallocate(void* ptr);
    
    // Memory operations
    bool memcpy(void* dst, const void* src, size_t size, void* stream = nullptr);
    bool memset(void* ptr, int value, size_t size, void* stream = nullptr);
    
    // Management
    void empty_cache(int device_id = -1);
    MemoryStats get_stats(int device_id = -1) const;
    
    // Device memory info
    uint64_t get_total_memory(int device_id = -1) const;
    uint64_t get_free_memory(int device_id = -1) const;
    uint64_t get_used_memory(int device_id = -1) const;
    
private:
    MemoryManager() = default;
    ~MemoryManager() = default;
    
    std::unordered_map<int, std::shared_ptr<MemoryAllocator>> allocators_;
    mutable std::mutex mutex_;
};

// Default allocation options
constexpr AllocationOptions DEFAULT_ALLOCATION_OPTIONS = {
    .pinned_host_memory = false,
    .unified_memory = false,
    .managed_memory = false,
    .alignment = 256
};

// Utility functions
bool is_device_pointer(const void* ptr);
int get_device_for_pointer(const void* ptr);
size_t get_memory_alignment();

} // namespace rdna

#endif // RDNA_MEMORY_H