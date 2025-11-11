#include "rdna/memory.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace rdna {

// MemoryAllocator implementation
MemoryAllocator::MemoryAllocator(std::shared_ptr<DeviceContext> context)
    : context_(context), cache_size_limit_(1024 * 1024 * 1024), // 1GB default limit
      allocation_counter_(0) {
    stats_ = {0, 0, 0, 0, 0, 0, 0};
}

MemoryAllocator::~MemoryAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all blocks
    for (auto& block : blocks_) {
        if (block->ptr) {
            hipError_t result = hipFree(block->ptr);
            if (result != hipSuccess) {
                std::cerr << "Warning: Failed to free memory block: " << hipGetErrorString(result) << std::endl;
            }
        }
    }
    blocks_.clear();
    cache_.clear();
}

void* MemoryAllocator::allocate(size_t size, const AllocationOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (size == 0) {
        return nullptr;
    }
    
    // Apply alignment
    size_t aligned_size = ((size + options.alignment - 1) / options.alignment) * options.alignment;
    
    // Try to find a free block
    Block* block = find_free_block(aligned_size);
    if (!block) {
        // Allocate new block
        block = allocate_new_block(aligned_size, options);
        if (!block) {
            return nullptr;
        }
    }
    
    // Split block if it's larger than needed
    if (block->size > aligned_size + sizeof(Block)) {
        split_block(block, aligned_size);
    }
    
    block->in_use = true;
    block->stream = options.stream;
    block->allocation_id = ++allocation_counter_;
    
    stats_.allocated_bytes += block->allocated_size;
    stats_.allocated_blocks++;
    stats_.total_allocations++;
    
    if (stats_.allocated_bytes > stats_.max_allocated_bytes) {
        stats_.max_allocated_bytes = stats_.allocated_bytes;
    }
    
    return block->ptr;
}

void MemoryAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the block
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
        [ptr](const std::unique_ptr<Block>& block) {
            return block->ptr == ptr;
        });
    
    if (it == blocks_.end()) {
        std::cerr << "Warning: Attempt to free unknown pointer" << std::endl;
        return;
    }
    
    Block* block = it->get();
    if (!block->in_use) {
        std::cerr << "Warning: Double free detected" << std::endl;
        return;
    }
    
    block->in_use = false;
    stats_.allocated_bytes -= block->allocated_size;
    stats_.allocated_blocks--;
    stats_.total_frees++;
    
    // Add to cache if the block is large enough
    if (block->size >= 1024) { // Cache blocks larger than 1KB
        add_to_cache(block);
    } else {
        // Small blocks are immediately freed
        hipError_t result = hipFree(block->ptr);
        if (result != hipSuccess) {
            std::cerr << "Warning: Failed to free memory: " << hipGetErrorString(result) << std::endl;
        }
        blocks_.erase(it);
    }
    
    // Try to merge adjacent free blocks
    merge_adjacent_blocks();
}

bool MemoryAllocator::memcpy(void* dst, const void* src, size_t size, void* stream) {
    hipError_t result;
    
    if (stream) {
        result = hipMemcpyAsync(dst, src, size, hipMemcpyDefault, static_cast<hipStream_t>(stream));
        if (result != hipSuccess) return false;
        result = hipStreamSynchronize(static_cast<hipStream_t>(stream));
    } else {
        result = hipMemcpy(dst, src, size, hipMemcpyDefault);
    }
    
    return result == hipSuccess;
}

bool MemoryAllocator::memset(void* ptr, int value, size_t size, void* stream) {
    hipError_t result;
    
    if (stream) {
        result = hipMemsetAsync(ptr, value, size, static_cast<hipStream_t>(stream));
        if (result != hipSuccess) return false;
        result = hipStreamSynchronize(static_cast<hipStream_t>(stream));
    } else {
        result = hipMemset(ptr, value, size);
    }
    
    return result == hipSuccess;
}

void MemoryAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanup_cache();
}

MemoryStats MemoryAllocator::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

AllocationInfo MemoryAllocator::get_allocation_info(void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
        [ptr](const std::unique_ptr<Block>& block) {
            return block->ptr == ptr;
        });
    
    if (it == blocks_.end()) {
        return AllocationInfo{};
    }
    
    const Block* block = it->get();
    return AllocationInfo{
        block->ptr,
        block->size,
        block->allocated_size,
        true, // is_device_memory
        context_->get_device_id(),
        block->stream,
        block->allocation_id
    };
}

void MemoryAllocator::set_cache_size_limit(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_size_limit_ = limit;
    if (stats_.cached_bytes > limit) {
        evict_from_cache(stats_.cached_bytes - limit);
    }
}

size_t MemoryAllocator::get_cache_size_limit() const {
    return cache_size_limit_;
}

uint64_t MemoryAllocator::get_total_memory() const {
    size_t free, total;
    hipError_t result = hipMemGetInfo(&free, &total);
    return (result == hipSuccess) ? total : 0;
}

uint64_t MemoryAllocator::get_free_memory() const {
    size_t free, total;
    hipError_t result = hipMemGetInfo(&free, &total);
    return (result == hipSuccess) ? free : 0;
}

uint64_t MemoryAllocator::get_used_memory() const {
    size_t free, total;
    hipError_t result = hipMemGetInfo(&free, &total);
    return (result == hipSuccess) ? (total - free) : 0;
}

// Private implementation methods
MemoryAllocator::Block* MemoryAllocator::find_free_block(size_t size) {
    for (auto& block : blocks_) {
        if (!block->in_use && block->size >= size) {
            return block.get();
        }
    }
    return nullptr;
}

MemoryAllocator::Block* MemoryAllocator::allocate_new_block(size_t size, const AllocationOptions& options) {
    void* ptr = nullptr;
    hipError_t result;
    
    if (options.pinned_host_memory) {
        result = hipHostMalloc(&ptr, size, hipHostMallocDefault);
    } else if (options.unified_memory) {
        result = hipMallocManaged(&ptr, size);
    } else {
        result = hipMalloc(&ptr, size);
    }
    
    if (result != hipSuccess) {
        return nullptr;
    }
    
    auto block = std::make_unique<Block>();
    block->ptr = ptr;
    block->size = size;
    block->allocated_size = size;
    block->in_use = false;
    block->stream = options.stream;
    
    Block* block_ptr = block.get();
    blocks_.push_back(std::move(block));
    return block_ptr;
}

void MemoryAllocator::split_block(Block* block, size_t needed_size) {
    if (block->size <= needed_size + sizeof(Block)) {
        return;
    }
    
    size_t remaining_size = block->size - needed_size;
    void* remaining_ptr = static_cast<char*>(block->ptr) + needed_size;
    
    auto new_block = std::make_unique<Block>();
    new_block->ptr = remaining_ptr;
    new_block->size = remaining_size;
    new_block->allocated_size = remaining_size;
    new_block->in_use = false;
    new_block->stream = block->stream;
    
    block->size = needed_size;
    
    // Insert after current block
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
        [block](const std::unique_ptr<Block>& b) { return b.get() == block; });
    
    if (it != blocks_.end()) {
        blocks_.insert(std::next(it), std::move(new_block));
    }
}

void MemoryAllocator::merge_adjacent_blocks() {
    // Simple implementation - could be optimized
    bool merged;
    do {
        merged = false;
        for (size_t i = 0; i < blocks_.size() - 1; ++i) {
            Block* current = blocks_[i].get();
            Block* next = blocks_[i + 1].get();
            
            if (!current->in_use && !next->in_use &&
                static_cast<char*>(current->ptr) + current->size == next->ptr) {
                
                current->size += next->size;
                blocks_.erase(blocks_.begin() + i + 1);
                merged = true;
                break;
            }
        }
    } while (merged);
}

void MemoryAllocator::add_to_cache(Block* block) {
    auto cached_block = std::make_unique<CacheBlock>();
    cached_block->ptr = block->ptr;
    cached_block->size = block->size;
    cached_block->stream = block->stream;
    cached_block->last_used = allocation_counter_;
    
    cache_.push_back(std::move(cached_block));
    stats_.cached_bytes += block->size;
    stats_.cached_blocks++;
}

MemoryAllocator::CacheBlock* MemoryAllocator::find_cached_block(size_t size) {
    for (auto& cached_block : cache_) {
        if (cached_block->size >= size) {
            return cached_block.get();
        }
    }
    return nullptr;
}

void MemoryAllocator::evict_from_cache(size_t needed_size) {
    // Sort by last used (LRU)
    std::sort(cache_.begin(), cache_.end(),
        [](const std::unique_ptr<CacheBlock>& a, const std::unique_ptr<CacheBlock>& b) {
            return a->last_used < b->last_used;
        });
    
    size_t freed_size = 0;
    auto it = cache_.begin();
    while (it != cache_.end() && freed_size < needed_size) {
        hipError_t result = hipFree((*it)->ptr);
        if (result == hipSuccess) {
            freed_size += (*it)->size;
            stats_.cached_bytes -= (*it)->size;
            stats_.cached_blocks--;
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void MemoryAllocator::cleanup_cache() {
    for (auto& cached_block : cache_) {
        hipError_t result = hipFree(cached_block->ptr);
        if (result != hipSuccess) {
            std::cerr << "Warning: Failed to free cached memory: " << hipGetErrorString(result) << std::endl;
        }
    }
    cache_.clear();
    stats_.cached_bytes = 0;
    stats_.cached_blocks = 0;
}

// MemoryManager implementation
MemoryManager& MemoryManager::get_instance() {
    static MemoryManager instance;
    return instance;
}

std::shared_ptr<MemoryAllocator> MemoryManager::get_allocator(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (device_id == -1) {
        int current_device;
        hipError_t result = hipGetDevice(&current_device);
        if (result != hipSuccess) {
            throw std::runtime_error("Failed to get current device");
        }
        device_id = current_device;
    }
    
    auto it = allocators_.find(device_id);
    if (it == allocators_.end()) {
        auto context = std::make_shared<DeviceContext>(device_id);
        if (!context->initialize()) {
            throw std::runtime_error("Failed to initialize device context for allocator");
        }
        
        auto allocator = std::make_shared<MemoryAllocator>(context);
        allocators_[device_id] = allocator;
        return allocator;
    }
    
    return it->second;
}

std::shared_ptr<MemoryAllocator> MemoryManager::get_current_allocator() {
    int current_device;
    hipError_t result = hipGetDevice(&current_device);
    if (result != hipSuccess) {
        throw std::runtime_error("Failed to get current device");
    }
    return get_allocator(current_device);
}

void* MemoryManager::allocate(size_t size, int device_id, const AllocationOptions& options) {
    auto allocator = get_allocator(device_id);
    return allocator->allocate(size, options);
}

void MemoryManager::deallocate(void* ptr) {
    if (!ptr) return;
    
    int device_id = get_device_for_pointer(ptr);
    if (device_id >= 0) {
        auto allocator = get_allocator(device_id);
        allocator->deallocate(ptr);
    }
}

bool MemoryManager::memcpy(void* dst, const void* src, size_t size, void* stream) {
    // This is a simplified implementation
    hipError_t result;
    
    if (stream) {
        result = hipMemcpyAsync(dst, src, size, hipMemcpyDefault, static_cast<hipStream_t>(stream));
        if (result != hipSuccess) return false;
        result = hipStreamSynchronize(static_cast<hipStream_t>(stream));
    } else {
        result = hipMemcpy(dst, src, size, hipMemcpyDefault);
    }
    
    return result == hipSuccess;
}

bool MemoryManager::memset(void* ptr, int value, size_t size, void* stream) {
    hipError_t result;
    
    if (stream) {
        result = hipMemsetAsync(ptr, value, size, static_cast<hipStream_t>(stream));
        if (result != hipSuccess) return false;
        result = hipStreamSynchronize(static_cast<hipStream_t>(stream));
    } else {
        result = hipMemset(ptr, value, size);
    }
    
    return result == hipSuccess;
}

void MemoryManager::empty_cache(int device_id) {
    auto allocator = get_allocator(device_id);
    allocator->empty_cache();
}

MemoryStats MemoryManager::get_stats(int device_id) const {
    auto allocator = const_cast<MemoryManager*>(this)->get_allocator(device_id);
    return allocator->get_stats();
}

uint64_t MemoryManager::get_total_memory(int device_id) const {
    auto allocator = const_cast<MemoryManager*>(this)->get_allocator(device_id);
    return allocator->get_total_memory();
}

uint64_t MemoryManager::get_free_memory(int device_id) const {
    auto allocator = const_cast<MemoryManager*>(this)->get_allocator(device_id);
    return allocator->get_free_memory();
}

uint64_t MemoryManager::get_used_memory(int device_id) const {
    auto allocator = const_cast<MemoryManager*>(this)->get_allocator(device_id);
    return allocator->get_used_memory();
}

// Utility functions
bool is_device_pointer(const void* ptr) {
    hipPointerAttribute_t attributes;
    hipError_t result = hipPointerGetAttributes(&attributes, ptr);
    return result == hipSuccess && attributes.memoryType == hipMemoryTypeDevice;
}

int get_device_for_pointer(const void* ptr) {
    hipPointerAttribute_t attributes;
    hipError_t result = hipPointerGetAttributes(&attributes, ptr);
    if (result != hipSuccess) {
        return -1;
    }
    return attributes.device;
}

size_t get_memory_alignment() {
    return 256; // Typical alignment for GPU memory
}

} // namespace rdna