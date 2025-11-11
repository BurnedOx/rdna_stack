#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rdna/utils.h"

namespace py = pybind11;
using namespace rdna;

void bind_utils(py::module& m) {
    // LibraryConfig binding
    py::class_<LibraryConfig>(m, "LibraryConfig")
        .def(py::init<>())
        .def_readwrite("enable_debug_logging", &LibraryConfig::enable_debug_logging)
        .def_readwrite("enable_profiling", &LibraryConfig::enable_profiling)
        .def_readwrite("memory_cache_limit", &LibraryConfig::memory_cache_limit)
        .def_readwrite("use_unified_memory", &LibraryConfig::use_unified_memory);

    // Configuration functions
    m.def("get_library_config", &get_library_config, "Get current library configuration");
    m.def("set_library_config", &set_library_config, "Set library configuration");
    
    // Diagnostic functions
    m.def("get_system_info", &get_system_info, "Get system information as string");
    m.def("print_system_info", &print_system_info, "Print system information");
    m.def("get_memory_info", &get_memory_info, "Get memory information", py::arg("device_id") = -1);
    m.def("print_memory_info", &print_memory_info, "Print memory information", py::arg("device_id") = -1);
    m.def("get_kernel_info", &get_kernel_info, "Get kernel information", py::arg("device_id") = -1);
    m.def("print_kernel_info", &print_kernel_info, "Print kernel information", py::arg("device_id") = -1);
    m.def("run_diagnostics", &run_diagnostics, "Run comprehensive diagnostics");
    
    // Version information
    m.def("get_library_version", &get_library_version, "Get library version");
    m.def("get_build_info", &get_build_info, "Get build information");
    
    // Device capability checking
    m.def("check_device_capability", &check_device_capability, 
          "Check device capability", py::arg("device_id"), py::arg("capability"));
    
    // Logging functions (exposed for advanced users)
    m.def("log_info", &log_info, "Log info message", py::arg("message"));
    m.def("log_warning", &log_warning, "Log warning message", py::arg("message"));
    m.def("log_error", &log_error, "Log error message", py::arg("message"));
    m.def("log_debug", &log_debug, "Log debug message", py::arg("message"));
    
    // Memory utilities
    m.def("calculate_aligned_size", &calculate_aligned_size, 
          "Calculate aligned size", py::arg("size"), py::arg("alignment"));
    m.def("is_aligned", &is_aligned, "Check if pointer is aligned", 
          py::arg("ptr"), py::arg("alignment"));
    m.def("align_pointer", &align_pointer, "Align pointer", 
          py::arg("ptr"), py::arg("alignment"));
    
    // Performance timing (conditionally compiled)
    m.def("create_timer", &create_timer, "Create performance timer", py::arg("name"));
    
    // Error handling utilities
    m.def("get_last_hip_error", &get_last_hip_error, "Get last HIP error");
    m.def("check_hip_error", [](int error_code, const std::string& context) {
        check_hip_error(static_cast<hipError_t>(error_code), context);
    }, "Check HIP error", py::arg("error_code"), py::arg("context"));
}

// Additional utility functions for Python
void bind_python_utils(py::module& m) {
    // Python-specific utilities
    m.def("get_python_version", []() {
        return PY_VERSION;
    }, "Get Python version");
    
    // Memory view utilities for Python buffers
    m.def("get_buffer_info", [](py::buffer buf) {
        py::buffer_info info = buf.request();
        return py::dict(
            "ptr"_a = reinterpret_cast<uintptr_t>(info.ptr),
            "size"_a = info.size,
            "itemsize"_a = info.itemsize,
            "format"_a = info.format,
            "ndim"_a = info.ndim,
            "shape"_a = info.shape,
            "strides"_a = info.strides
        );
    }, "Get buffer information");
    
    // DLPack support (placeholder for future implementation)
    m.def("to_dlpack", [](py::object obj) {
        throw std::runtime_error("DLPack support not yet implemented");
    }, "Convert to DLPack tensor");
    
    m.def("from_dlpack", [](py::object dlpack) {
        throw std::runtime_error("DLPack support not yet implemented");
    }, "Convert from DLPack tensor");
}