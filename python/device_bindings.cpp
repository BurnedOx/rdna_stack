#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rdna/device.h"

namespace py = pybind11;
using namespace rdna;

void bind_device(py::module& m) {
    // DeviceProperties binding
    py::class_<DeviceProperties>(m, "DeviceProperties")
        .def(py::init<>())
        .def_readonly("device_id", &DeviceProperties::device_id)
        .def_readonly("name", &DeviceProperties::name)
        .def_readonly("arch", &DeviceProperties::arch)
        .def_readonly("total_memory", &DeviceProperties::total_memory)
        .def_readonly("free_memory", &DeviceProperties::free_memory)
        .def_readonly("compute_units", &DeviceProperties::compute_units)
        .def_readonly("max_workgroup_size", &DeviceProperties::max_workgroup_size)
        .def_readonly("wavefront_size", &DeviceProperties::wavefront_size)
        .def_readonly("supports_fp16", &DeviceProperties::supports_fp16)
        .def_readonly("supports_bf16", &DeviceProperties::supports_bf16)
        .def_readonly("supports_tensor_cores", &DeviceProperties::supports_tensor_cores)
        .def_readonly("pci_bus_id", &DeviceProperties::pci_bus_id)
        .def_readonly("pci_device_id", &DeviceProperties::pci_device_id)
        .def("__repr__", [](const DeviceProperties& props) {
            return "<DeviceProperties device_id=" + std::to_string(props.device_id) + 
                   " name='" + props.name + "'>";
        });

    // DeviceManager binding
    py::class_<DeviceManager>(m, "DeviceManager")
        .def_static("get_instance", &DeviceManager::get_instance, 
                   py::return_value_policy::reference)
        .def("device_count", &DeviceManager::device_count)
        .def("get_device_properties", &DeviceManager::get_device_properties)
        .def("get_all_device_properties", &DeviceManager::get_all_device_properties)
        .def("create_context", &DeviceManager::create_context)
        .def("get_current_context", &DeviceManager::get_current_context)
        .def("set_current_context", &DeviceManager::set_current_context)
        .def("check_device_compatibility", &DeviceManager::check_device_compatibility)
        .def("get_last_error", &DeviceManager::get_last_error);

    // Stream binding
    py::class_<Stream, std::shared_ptr<Stream>>(m, "Stream")
        .def(py::init<std::shared_ptr<DeviceContext>>())
        .def("initialize", &Stream::initialize)
        .def("synchronize", &Stream::synchronize)
        .def("is_valid", &Stream::is_valid)
        .def("get_native_handle", &Stream::get_native_handle)
        .def("memcpy", &Stream::memcpy)
        .def("memcpy_async", &Stream::memcpy_async);

    // DeviceContext binding
    py::class_<DeviceContext, std::shared_ptr<DeviceContext>>(m, "DeviceContext")
        .def(py::init<int>())
        .def("initialize", &DeviceContext::initialize)
        .def("synchronize", &DeviceContext::synchronize)
        .def("is_valid", &DeviceContext::is_valid)
        .def("get_device_id", &DeviceContext::get_device_id)
        .def("get_properties", &DeviceContext::get_properties)
        .def("create_stream", &DeviceContext::create_stream)
        .def("get_default_stream", &DeviceContext::get_default_stream);

    // Device management functions
    m.def("current_device", []() {
        int device_id;
        // hipError_t result = hipGetDevice(&device_id);
        // if (result != hipSuccess) return -1;
        device_id = 0; // Placeholder
        return device_id;
    }, "Get current device ID");

    m.def("set_device", [](int device_id) {
        // hipError_t result = hipSetDevice(device_id);
        // return result == hipSuccess;
        return true; // Placeholder
    }, "Set current device");

    m.def("synchronize", []() {
        // hipError_t result = hipDeviceSynchronize();
        // return result == hipSuccess;
        return true; // Placeholder
    }, "Synchronize all devices");

    m.def("get_device_properties", [](int device_id) {
        DeviceManager& manager = DeviceManager::get_instance();
        return manager.get_device_properties(device_id);
    }, "Get device properties", py::arg("device_id") = 0);

    m.def("get_device_capability", [](int device_id, const std::string& capability) {
        DeviceManager& manager = DeviceManager::get_instance();
        auto props = manager.get_device_properties(device_id);
        
        if (capability == "fp16") return props.supports_fp16;
        if (capability == "bf16") return props.supports_bf16;
        if (capability == "tensor_cores") return props.supports_tensor_cores;
        
        return false;
    }, "Check device capability", py::arg("device_id") = 0, py::arg("capability"));
}