#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "rdna/device.h"
#include "rdna/memory.h"
#include "rdna/kernels.h"
#include "rdna/utils.h"

namespace py = pybind11;

// Forward declarations
void bind_device(py::module& m);
void bind_memory(py::module& m);
void bind_kernels(py::module& m);
void bind_utils(py::module& m);

PYBIND11_MODULE(rdna_py, m) {
    m.doc() = "RDNA Stack Python Bindings - AMD GPU acceleration for PyTorch and TensorFlow";
    
    // Module version
    m.attr("__version__") = "0.1.0";
    
    // Bind all components
    bind_device(m);
    bind_memory(m);
    bind_kernels(m);
    bind_utils(m);
    
    // Module-level functions
    m.def("is_available", &rdna::is_rdna_supported, "Check if RDNA devices are available");
    m.def("device_count", []() { return rdna::DeviceManager::get_instance().device_count(); },
          "Get number of available RDNA devices");
    m.def("get_roc_version", &rdna::get_roc_version, "Get ROCm version");
    m.def("get_hip_version", &rdna::get_hip_version, "Get HIP version");
    m.def("get_library_version", &rdna::get_library_version, "Get RDNA stack version");
    
    // Initialize the library
    m.def("initialize", []() {
        // Initialize device manager and kernels
        rdna::DeviceManager& device_manager = rdna::DeviceManager::get_instance();
        rdna::KernelManager& kernel_manager = rdna::KernelManager::get_instance();
        
        int device_count = device_manager.device_count();
        for (int i = 0; i < device_count; ++i) {
            kernel_manager.initialize_kernels(i);
        }
        
        return true;
    }, "Initialize the RDNA stack");
    
    // Diagnostic function
    m.def("diagnostics", &rdna::run_diagnostics, "Run comprehensive diagnostics");
    
    // Configuration functions
    m.def("set_debug_logging", &rdna::set_debug_logging, "Enable or disable debug logging");
    m.def("set_profiling", &rdna::set_profiling, "Enable or disable profiling");
    m.def("set_memory_cache_limit", &rdna::set_memory_cache_limit, "Set memory cache limit in bytes");
}