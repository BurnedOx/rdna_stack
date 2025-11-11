#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/platform/stream_executor.h"

#include "rdna/device.h"
#include "rdna/memory.h"
#include "rdna/kernels.h"

namespace tensorflow {
namespace rdna {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// RDNA device registration
class RDNADevice : public DeviceBase {
public:
    RDNADevice(Env* env, const DeviceAttributes& device_attributes)
        : DeviceBase(env), device_attributes_(device_attributes) {
        // Initialize RDNA device
        device_id_ = device_attributes.device_id();
        rdna::DeviceManager::get_instance().create_context(device_id_);
        rdna::KernelManager::get_instance().initialize_kernels(device_id_);
    }
    
    ~RDNADevice() override {
        // Cleanup RDNA resources
    }
    
    Allocator* GetAllocator(AllocatorAttributes attr) override {
        // Return RDNA memory allocator
        static rdna::MemoryAllocator* allocator = nullptr;
        if (!allocator) {
            auto context = rdna::DeviceManager::get_instance().create_context(device_id_);
            allocator = new rdna::MemoryAllocator(context);
        }
        return reinterpret_cast<Allocator*>(allocator);
    }
    
    const DeviceAttributes& attributes() const override {
        return device_attributes_;
    }
    
private:
    DeviceAttributes device_attributes_;
    int device_id_;
};

// RDNA device factory
class RDNADeviceFactory : public DeviceFactory {
public:
    Status ListPhysicalDevices(std::vector<string>* devices) override {
        int count = rdna::DeviceManager::get_instance().device_count();
        for (int i = 0; i < count; ++i) {
            devices->push_back(strings::StrCat("/physical_device:RDNA:", i));
        }
        return Status::OK();
    }
    
    Status CreateDevices(const SessionOptions& options,
                        const string& name_prefix,
                        std::vector<Device*>* devices) override {
        int count = rdna::DeviceManager::get_instance().device_count();
        for (int i = 0; i < count; ++i) {
            DeviceAttributes device_attributes;
            device_attributes.set_name(strings::StrCat(name_prefix, "/device:RDNA:", i));
            device_attributes.set_device_type("RDNA");
            device_attributes.set_device_id(i);
            
            // Set device properties
            auto props = rdna::DeviceManager::get_instance().get_device_properties(i);
            device_attributes.set_physical_device_desc(
                strings::StrCat("RDNA Device: ", props.name));
            
            devices->push_back(new RDNADevice(options.env, device_attributes));
        }
        return Status::OK();
    }
};

REGISTER_LOCAL_DEVICE_FACTORY("RDNA", RDNADeviceFactory);

// RDNA kernel base class
class RDNAOpKernel : public OpKernel {
public:
    explicit RDNAOpKernel(OpKernelConstruction* context) : OpKernel(context) {
        // Get device context
        device_id_ = context->device()->attributes().device_id();
        context_ = rdna::DeviceManager::get_instance().create_context(device_id_);
        kernel_manager_ = &rdna::KernelManager::get_instance();
    }
    
protected:
    int device_id_;
    std::shared_ptr<rdna::DeviceContext> context_;
    rdna::KernelManager* kernel_manager_;
};

// RDNA MatMul operation
class RDNAMatMulOp : public RDNAOpKernel {
public:
    explicit RDNAMatMulOp(OpKernelConstruction* context) : RDNAOpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
        OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
    }
    
    void Compute(OpKernelContext* context) override {
        const Tensor& a = context->input(0);
        const Tensor& b = context->input(1);
        
        // Validate inputs
        OP_REQUIRES(context, a.dims() == 2 && b.dims() == 2,
                    errors::InvalidArgument("Inputs must be 2D"));
        
        // Create output tensor
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {a.dim_size(0), b.dim_size(1)}, &output));
        
        // Prepare RDNA tensor descriptors
        rdna::TensorDesc a_desc({a.dim_size(0), a.dim_size(1)}, 0);
        rdna::TensorDesc b_desc({b.dim_size(0), b.dim_size(1)}, 0);
        rdna::TensorDesc c_desc({output->dim_size(0), output->dim_size(1)}, 0);
        
        rdna::MatmulConfig config;
        config.transpose_a = transpose_a_;
        config.transpose_b = transpose_b_;
        
        // Dispatch matmul operation
        bool success = kernel_manager_->dispatch_matmul(
            a_desc, a.tensor_data().data(),
            b_desc, b.tensor_data().data(),
            c_desc, output->tensor_data().data(),
            config, device_id_);
        
        OP_REQUIRES(context, success, errors::Internal("RDNA matmul operation failed"));
    }
    
private:
    bool transpose_a_;
    bool transpose_b_;
};

REGISTER_KERNEL_BUILDER(Name("RDNAMatMul").Device("RDNA"), RDNAMatMulOp);

// RDNA Conv2D operation
class RDNAConv2DOp : public RDNAOpKernel {
public:
    explicit RDNAConv2DOp(OpKernelConstruction* context) : RDNAOpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
        OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    }
    
    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const Tensor& filter = context->input(1);
        
        // Validate inputs
        OP_REQUIRES(context, input.dims() == 4 && filter.dims() == 4,
                    errors::InvalidArgument("Inputs must be 4D"));
        
        // Calculate output dimensions
        int32 batch = input.dim_size(0);
        int32 in_height = input.dim_size(1);
        int32 in_width = input.dim_size(2);
        int32 in_channels = input.dim_size(3);
        
        int32 filter_height = filter.dim_size(0);
        int32 filter_width = filter.dim_size(1);
        int32 out_channels = filter.dim_size(3);
        
        int32 out_height, out_width;
        OP_REQUIRES_OK(context, GetWindowedOutputSize(in_height, filter_height,
                                                     strides_[1], padding_,
                                                     &out_height));
        OP_REQUIRES_OK(context, GetWindowedOutputSize(in_width, filter_width,
                                                     strides_[2], padding_,
                                                     &out_width));
        
        // Create output tensor
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
                        {batch, out_height, out_width, out_channels}, &output));
        
        // Prepare RDNA tensor descriptors
        rdna::TensorDesc input_desc({batch, in_height, in_width, in_channels}, 0);
        rdna::TensorDesc filter_desc({filter_height, filter_width, in_channels, out_channels}, 0);
        rdna::TensorDesc output_desc({batch, out_height, out_width, out_channels}, 0);
        
        rdna::ConvConfig config;
        config.padding = {0, 0}; // Simplified
        config.stride = {strides_[1], strides_[2]};
        config.dilation = {dilations_[1], dilations_[2]};
        config.groups = 1;
        
        // Dispatch conv2d operation
        bool success = kernel_manager_->dispatch_conv2d(
            input_desc, input.tensor_data().data(),
            filter_desc, filter.tensor_data().data(),
            output_desc, output->tensor_data().data(),
            config, device_id_);
        
        OP_REQUIRES(context, success, errors::Internal("RDNA conv2d operation failed"));
    }
    
private:
    std::vector<int32> strides_;
    Padding padding_;
    std::vector<int32> dilations_;
};

REGISTER_KERNEL_BUILDER(Name("RDNAConv2D").Device("RDNA"), RDNAConv2DOp);

// RDNA Add operation
class RDNAAddOp : public RDNAOpKernel {
public:
    explicit RDNAAddOp(OpKernelConstruction* context) : RDNAOpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        const Tensor& a = context->input(0);
        const Tensor& b = context->input(1);
        
        // Validate inputs
        OP_REQUIRES(context, a.shape() == b.shape(),
                    errors::InvalidArgument("Input shapes must match"));
        
        // Create output tensor
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, a.shape(), &output));
        
        // Prepare RDNA tensor descriptors
        std::vector<size_t> shape;
        for (int i = 0; i < a.dims(); ++i) {
            shape.push_back(a.dim_size(i));
        }
        
        rdna::TensorDesc a_desc(shape, 0);
        rdna::TensorDesc b_desc(shape, 0);
        rdna::TensorDesc c_desc(shape, 0);
        
        // Use custom kernels for element-wise addition
        auto custom_kernels = kernel_manager_->get_custom_kernels(device_id_);
        bool success = custom_kernels->add(
            a_desc, a.tensor_data().data(),
            b_desc, b.tensor_data().data(),
            c_desc, output->tensor_data().data());
        
        OP_REQUIRES(context, success, errors::Internal("RDNA add operation failed"));
    }
};

REGISTER_KERNEL_BUILDER(Name("RDNAAdd").Device("RDNA"), RDNAAddOp);

} // namespace rdna
} // namespace tensorflow