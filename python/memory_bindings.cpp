#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rdna/memory.h"

namespace py = pybind11;
using namespace rdna;

void bind_memory(py::module& m) {
    // AllocationInfo binding
    py::class_<AllocationInfo>(m, "AllocationInfo")
        .def(py::init<>())
        .def_readonly("ptr", &AllocationInfo::ptr)
        .def_readonly("size", &AllocationInfo::size)
        .def_readonly("allocated_size", &AllocationInfo::allocated_size)
        .def_readonly("is_device_memory", &AllocationInfo::is_device_memory)
        .def_readonly("device_id", &AllocationInfo::device_id)
        .def_readonly("stream", &AllocationInfo::stream)
        .def_readonly("allocation_id", &AllocationInfo::allocation_id);

    // MemoryStats binding
    py::class_<MemoryStats>(m, "MemoryStats")
        .def(py::init<>())
        .def_readonly("allocated_bytes", &MemoryStats::allocated_bytes)
        .def_readonly("allocated_blocks", &MemoryStats::allocated_blocks)
        .def_readonly("cached_bytes", &MemoryStats::cached_bytes)
        .def_readonly("cached_blocks", &MemoryStats::cached_blocks)
        .def_readonly("max_allocated_bytes", &MemoryStats::max_allocated_bytes)
        .def_readonly("total_allocations", &MemoryStats::total_allocations)
        .def_readonly("total_frees", &MemoryStats::total_frees);

    // AllocationOptions binding
    py::class_<AllocationOptions>(m, "AllocationOptions")
        .def(py::init<>())
        .def_readwrite("pinned_host_memory", &AllocationOptions::pinned_host_memory)
        .def_readwrite("unified_memory", &AllocationOptions::unified_memory)
        .def_readwrite("managed_memory", &AllocationOptions::managed_memory)
        .def_readwrite("alignment", &AllocationOptions::alignment)
        .def_readwrite("stream", &AllocationOptions::stream);

    // MemoryAllocator binding
    py::class_<MemoryAllocator, std::shared_ptr<MemoryAllocator>>(m, "MemoryAllocator")
        .def(py::init<std::shared_ptr<DeviceContext>>())
        .def("allocate", &MemoryAllocator::allocate, 
             py::arg("size"), py::arg("options") = DEFAULT_ALLOCATION_OPTIONS)
        .def("deallocate", &MemoryAllocator::deallocate)
        .def("memcpy", &MemoryAllocator::memcpy)
        .def("memset", &MemoryAllocator::memset)
        .def("empty_cache", &MemoryAllocator::empty_cache)
        .def("get_stats", &MemoryAllocator::get_stats)
        .def("get_allocation_info", &MemoryAllocator::get_allocation_info)
        .def("set_cache_size_limit", &MemoryAllocator::set_cache_size_limit)
        .def("get_cache_size_limit", &MemoryAllocator::get_cache_size_limit)
        .def("get_total_memory", &MemoryAllocator::get_total_memory)
        .def("get_free_memory", &MemoryAllocator::get_free_memory)
        .def("get_used_memory", &MemoryAllocator::get_used_memory);

    // MemoryManager binding
    py::class_<MemoryManager>(m, "MemoryManager")
        .def_static("get_instance", &MemoryManager::get_instance, 
                   py::return_value_policy::reference)
        .def("get_allocator", &MemoryManager::get_allocator)
        .def("get_current_allocator", &MemoryManager::get_current_allocator)
        .def("allocate", &MemoryManager::allocate, 
             py::arg("size"), py::arg("device_id") = -1, 
             py::arg("options") = DEFAULT_ALLOCATION_OPTIONS)
        .def("deallocate", &MemoryManager::deallocate)
        .def("memcpy", &MemoryManager::memcpy)
        .def("memset", &MemoryManager::memset)
        .def("empty_cache", &MemoryManager::empty_cache)
        .def("get_stats", &MemoryManager::get_stats)
        .def("get_total_memory", &MemoryManager::get_total_memory)
        .def("get_free_memory", &MemoryManager::get_free_memory)
        .def("get_used_memory", &MemoryManager::get_used_memory);

    // Memory management functions (PyTorch-like API)
    m.def("empty_cache", [](int device_id) {
        MemoryManager& manager = MemoryManager::get_instance();
        manager.empty_cache(device_id);
    }, "Empty memory cache", py::arg("device_id") = -1);

    m.def("memory_allocated", [](int device_id) {
        MemoryManager& manager = MemoryManager::get_instance();
        MemoryStats stats = manager.get_stats(device_id);
        return stats.allocated_bytes;
    }, "Get allocated memory in bytes", py::arg("device_id") = -1);

    m.def("max_memory_allocated", [](int device_id) {
        MemoryManager& manager = MemoryManager::get_instance();
        MemoryStats stats = manager.get_stats(device_id);
        return stats.max_allocated_bytes;
    }, "Get maximum allocated memory in bytes", py::arg("device_id") = -1);

    m.def("memory_reserved", [](int device_id) {
        MemoryManager& manager = MemoryManager::get_instance();
        MemoryStats stats = manager.get_stats(device_id);
        return stats.allocated_bytes + stats.cached_bytes;
    }, "Get reserved memory (allocated + cached) in bytes", py::arg("device_id") = -1);

    m.def("memory_cached", [](int device_id) {
        MemoryManager& manager = MemoryManager::get_instance();
        MemoryStats stats = manager.get_stats(device_id);
        return stats.cached_bytes;
    }, "Get cached memory in bytes", py::arg("device_id") = -1);

    m.def("memory_summary", [](int device_id) {
        MemoryManager& manager = MemoryManager::get_instance();
        MemoryStats stats = manager.get_stats(device_id);
        uint64_t total = manager.get_total_memory(device_id);
        uint64_t free = manager.get_free_memory(device_id);
        uint64_t used = manager.get_used_memory(device_id);
        
        std::stringstream ss;
        ss << "Memory Summary (Device " << device_id << "):\n";
        ss << "  Allocated: " << (stats.allocated_bytes / (1024.0 * 1024.0)) << " MB\n";
        ss << "  Cached: " << (stats.cached_bytes / (1024.0 * 1024.0)) << " MB\n";
        ss << "  Total Device: " << (total / (1024.0 * 1024.0)) << " MB\n";
        ss << "  Free Device: " << (free / (1024.0 * 1024.0)) << " MB\n";
        ss << "  Used Device: " << (used / (1024.0 * 1024.0)) << " MB\n";
        ss << "  Max Allocated: " << (stats.max_allocated_bytes / (1024.0 * 1024.0)) << " MB\n";
        
        return ss.str();
    }, "Get memory summary", py::arg("device_id") = -1);

    // Utility functions
    m.def("is_device_pointer", &is_device_pointer, "Check if pointer is device memory");
    m.def("get_device_for_pointer", &get_device_for_pointer, "Get device ID for pointer");
    m.def("get_memory_alignment", &get_memory_alignment, "Get memory alignment");
}