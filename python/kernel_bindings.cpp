#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rdna/kernels.h"

namespace py = pybind11;
using namespace rdna;

void bind_kernels(py::module& m) {
    // KernelConfig binding
    py::class_<KernelConfig>(m, "KernelConfig")
        .def(py::init<>())
        .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t>())
        .def_readwrite("grid_size", &KernelConfig::grid_size)
        .def_readwrite("block_size", &KernelConfig::block_size)
        .def_readwrite("shared_memory_size", &KernelConfig::shared_memory_size)
        .def_readwrite("stream", &KernelConfig::stream);

    // TensorDesc binding
    py::class_<TensorDesc>(m, "TensorDesc")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&, int>())
        .def_readwrite("shape", &TensorDesc::shape)
        .def_readwrite("strides", &TensorDesc::strides)
        .def_readwrite("data_type", &TensorDesc::data_type)
        .def_readwrite("contiguous", &TensorDesc::contiguous)
        .def("num_elements", &TensorDesc::num_elements)
        .def("get_size", &TensorDesc::get_size);

    // MatmulConfig binding
    py::class_<MatmulConfig>(m, "MatmulConfig")
        .def(py::init<>())
        .def_readwrite("transpose_a", &MatmulConfig::transpose_a)
        .def_readwrite("transpose_b", &MatmulConfig::transpose_b)
        .def_readwrite("alpha", &MatmulConfig::alpha)
        .def_readwrite("beta", &MatmulConfig::beta);

    // ConvConfig binding
    py::class_<ConvConfig>(m, "ConvConfig")
        .def(py::init<>())
        .def_readwrite("padding", &ConvConfig::padding)
        .def_readwrite("stride", &ConvConfig::stride)
        .def_readwrite("dilation", &ConvConfig::dilation)
        .def_readwrite("groups", &ConvConfig::groups)
        .def_readwrite("benchmark", &ConvConfig::benchmark);

    // OperatorKernel base class binding
    py::class_<OperatorKernel>(m, "OperatorKernel")
        .def("initialize", &OperatorKernel::initialize)
        .def("is_initialized", &OperatorKernel::is_initialized)
        .def("get_name", &OperatorKernel::get_name);

    // MatmulKernel binding
    py::class_<MatmulKernel, OperatorKernel, std::shared_ptr<MatmulKernel>>(m, "MatmulKernel")
        .def(py::init<std::shared_ptr<DeviceContext>>())
        .def("matmul", &MatmulKernel::matmul)
        .def("batched_matmul", &MatmulKernel::batched_matmul);

    // ConvKernel binding
    py::class_<ConvKernel, OperatorKernel, std::shared_ptr<ConvKernel>>(m, "ConvKernel")
        .def(py::init<std::shared_ptr<DeviceContext>>())
        .def("conv2d_forward", &ConvKernel::conv2d_forward)
        .def("conv2d_backward_data", &ConvKernel::conv2d_backward_data)
        .def("conv2d_backward_filter", &ConvKernel::conv2d_backward_filter)
        .def("find_best_algorithm", &ConvKernel::find_best_algorithm);

    // CustomKernels binding
    py::class_<CustomKernels, OperatorKernel, std::shared_ptr<CustomKernels>>(m, "CustomKernels")
        .def(py::init<std::shared_ptr<DeviceContext>>())
        .def("add", &CustomKernels::add)
        .def("multiply", &CustomKernels::multiply)
        .def("relu", &CustomKernels::relu)
        .def("gelu", &CustomKernels::gelu)
        .def("softmax", &CustomKernels::softmax)
        .def("sum", &CustomKernels::sum)
        .def("mean", &CustomKernels::mean);

    // KernelManager binding
    py::class_<KernelManager>(m, "KernelManager")
        .def_static("get_instance", &KernelManager::get_instance, 
                   py::return_value_policy::reference)
        .def("get_matmul_kernel", &KernelManager::get_matmul_kernel)
        .def("get_conv_kernel", &KernelManager::get_conv_kernel)
        .def("get_custom_kernels", &KernelManager::get_custom_kernels)
        .def("initialize_kernels", &KernelManager::initialize_kernels)
        .def("are_kernels_initialized", &KernelManager::are_kernels_initialized)
        .def("dispatch_matmul", &KernelManager::dispatch_matmul)
        .def("dispatch_conv2d", &KernelManager::dispatch_conv2d);

    // Kernel operation functions
    m.def("matmul", [](const TensorDesc& a, py::buffer a_buf,
                       const TensorDesc& b, py::buffer b_buf,
                       const TensorDesc& c, py::buffer c_buf,
                       const MatmulConfig& config, int device_id, void* stream) {
        KernelManager& manager = KernelManager::get_instance();
        
        py::buffer_info a_info = a_buf.request();
        py::buffer_info b_info = b_buf.request();
        py::buffer_info c_info = c_buf.request();
        
        return manager.dispatch_matmul(a, a_info.ptr, b, b_info.ptr, c, c_info.ptr, 
                                      config, device_id, stream);
    }, "Matrix multiplication", 
       py::arg("a"), py::arg("a_data"),
       py::arg("b"), py::arg("b_data"),
       py::arg("c"), py::arg("c_data"),
       py::arg("config") = MatmulConfig(),
       py::arg("device_id") = -1, py::arg("stream") = nullptr);

    m.def("conv2d", [](const TensorDesc& input, py::buffer input_buf,
                       const TensorDesc& filter, py::buffer filter_buf,
                       const TensorDesc& output, py::buffer output_buf,
                       const ConvConfig& config, int device_id, void* stream) {
        KernelManager& manager = KernelManager::get_instance();
        
        py::buffer_info input_info = input_buf.request();
        py::buffer_info filter_info = filter_buf.request();
        py::buffer_info output_info = output_buf.request();
        
        return manager.dispatch_conv2d(input, input_info.ptr, filter, filter_info.ptr,
                                      output, output_info.ptr, config, device_id, stream);
    }, "2D convolution", 
       py::arg("input"), py::arg("input_data"),
       py::arg("filter"), py::arg("filter_data"),
       py::arg("output"), py::arg("output_data"),
       py::arg("config") = ConvConfig(),
       py::arg("device_id") = -1, py::arg("stream") = nullptr);

    // Utility functions
    m.def("calculate_matmul_kernel_config", &calculate_matmul_kernel_config, 
          "Calculate kernel configuration for matmul");
    m.def("calculate_conv_kernel_config", &calculate_conv_kernel_config,
          "Calculate kernel configuration for convolution");
    m.def("get_data_type_size", &get_data_type_size,
          "Get size of data type in bytes");
}