#ifndef RDNA_KERNELS_H
#define RDNA_KERNELS_H

#include <cstddef>
#include <vector>
#include <memory>
#include "device.h"

namespace rdna {

// Forward declarations
class DeviceContext;
class Stream;

/**
 * @brief Kernel launch configuration
 */
struct KernelConfig {
    size_t grid_size[3];
    size_t block_size[3];
    size_t shared_memory_size;
    void* stream;
    
    KernelConfig();
    KernelConfig(size_t grid_x, size_t grid_y, size_t grid_z,
                 size_t block_x, size_t block_y, size_t block_z);
};

/**
 * @brief Tensor descriptor for kernel operations
 */
struct TensorDesc {
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    int data_type;  // 0: float32, 1: float16, 2: bfloat16
    bool contiguous;
    
    TensorDesc();
    TensorDesc(const std::vector<size_t>& shape, int data_type = 0);
    size_t num_elements() const;
    size_t get_size() const;
};

/**
 * @brief Matmul operation configuration
 */
struct MatmulConfig {
    bool transpose_a;
    bool transpose_b;
    float alpha;
    float beta;
    
    MatmulConfig();
};

/**
 * @brief Convolution operation configuration
 */
struct ConvConfig {
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    int groups;
    bool benchmark;
    
    ConvConfig();
};

/**
 * @brief Base class for operator kernels
 */
class OperatorKernel {
public:
    virtual ~OperatorKernel() = default;
    
    virtual bool initialize() = 0;
    virtual bool is_initialized() const = 0;
    virtual std::string get_name() const = 0;
    
protected:
    bool initialized_;
};

/**
 * @brief Matmul kernel using rocBLAS and MIOpen
 */
class MatmulKernel : public OperatorKernel {
public:
    MatmulKernel(std::shared_ptr<DeviceContext> context);
    ~MatmulKernel();
    
    bool initialize() override;
    bool is_initialized() const override;
    std::string get_name() const override;
    
    // Matrix multiplication: C = alpha * op(A) * op(B) + beta * C
    bool matmul(const TensorDesc& a, const void* a_data,
                const TensorDesc& b, const void* b_data,
                const TensorDesc& c, void* c_data,
                const MatmulConfig& config = MatmulConfig(),
                void* stream = nullptr);
    
    // Batched matmul
    bool batched_matmul(const std::vector<TensorDesc>& a_batch, const void* a_data,
                        const std::vector<TensorDesc>& b_batch, const void* b_data,
                        const std::vector<TensorDesc>& c_batch, void* c_data,
                        const MatmulConfig& config = MatmulConfig(),
                        void* stream = nullptr);
    
private:
    std::shared_ptr<DeviceContext> context_;
    void* rocblas_handle_;
};

/**
 * @brief Convolution kernel using MIOpen
 */
class ConvKernel : public OperatorKernel {
public:
    ConvKernel(std::shared_ptr<DeviceContext> context);
    ~ConvKernel();
    
    bool initialize() override;
    bool is_initialized() const override;
    std::string get_name() const override;
    
    // 2D convolution forward
    bool conv2d_forward(const TensorDesc& input, const void* input_data,
                        const TensorDesc& filter, const void* filter_data,
                        const TensorDesc& output, void* output_data,
                        const ConvConfig& config = ConvConfig(),
                        void* stream = nullptr);
    
    // 2D convolution backward (input gradient)
    bool conv2d_backward_data(const TensorDesc& filter, const void* filter_data,
                               const TensorDesc& output_grad, const void* output_grad_data,
                               const TensorDesc& input_grad, void* input_grad_data,
                               const ConvConfig& config = ConvConfig(),
                               void* stream = nullptr);
    
    // 2D convolution backward (filter gradient)
    bool conv2d_backward_filter(const TensorDesc& input, const void* input_data,
                                 const TensorDesc& output_grad, const void* output_grad_data,
                                 const TensorDesc& filter_grad, void* filter_grad_data,
                                 const ConvConfig& config = ConvConfig(),
                                 void* stream = nullptr);
    
    // Find best algorithm for given configuration
    std::string find_best_algorithm(const TensorDesc& input,
                                     const TensorDesc& filter,
                                     const TensorDesc& output,
                                     const ConvConfig& config);
    
private:
    std::shared_ptr<DeviceContext> context_;
    void* miopen_handle_;
};

/**
 * @brief Custom HIP kernels for operations not covered by MIOpen
 */
class CustomKernels : public OperatorKernel {
public:
    CustomKernels(std::shared_ptr<DeviceContext> context);
    ~CustomKernels();
    
    bool initialize() override;
    bool is_initialized() const override;
    std::string get_name() const override;
    
    // Element-wise operations
    bool add(const TensorDesc& a, const void* a_data,
             const TensorDesc& b, const void* b_data,
             const TensorDesc& c, void* c_data,
             void* stream = nullptr);
    
    bool multiply(const TensorDesc& a, const void* a_data,
                  const TensorDesc& b, const void* b_data,
                  const TensorDesc& c, void* c_data,
                  void* stream = nullptr);
    
    // Activation functions
    bool relu(const TensorDesc& input, const void* input_data,
              const TensorDesc& output, void* output_data,
              void* stream = nullptr);
    
    bool gelu(const TensorDesc& input, const void* input_data,
              const TensorDesc& output, void* output_data,
              void* stream = nullptr);
    
    bool softmax(const TensorDesc& input, const void* input_data,
                 const TensorDesc& output, void* output_data,
                 int dim, void* stream = nullptr);
    
    // Reduction operations
    bool sum(const TensorDesc& input, const void* input_data,
             const TensorDesc& output, void* output_data,
             const std::vector<int>& dims, void* stream = nullptr);
    
    bool mean(const TensorDesc& input, const void* input_data,
              const TensorDesc& output, void* output_data,
              const std::vector<int>& dims, void* stream = nullptr);
    
private:
    std::shared_ptr<DeviceContext> context_;
    std::vector<void*> compiled_kernels_;
};

/**
 * @brief Kernel manager for operator dispatch
 */
class KernelManager {
public:
    static KernelManager& get_instance();
    
    // Kernel access
    std::shared_ptr<MatmulKernel> get_matmul_kernel(int device_id);
    std::shared_ptr<ConvKernel> get_conv_kernel(int device_id);
    std::shared_ptr<CustomKernels> get_custom_kernels(int device_id);
    
    // Kernel initialization
    bool initialize_kernels(int device_id);
    bool are_kernels_initialized(int device_id) const;
    
    // Operation dispatch
    bool dispatch_matmul(const TensorDesc& a, const void* a_data,
                         const TensorDesc& b, const void* b_data,
                         const TensorDesc& c, void* c_data,
                         const MatmulConfig& config = MatmulConfig(),
                         int device_id = -1, void* stream = nullptr);
    
    bool dispatch_conv2d(const TensorDesc& input, const void* input_data,
                         const TensorDesc& filter, const void* filter_data,
                         const TensorDesc& output, void* output_data,
                         const ConvConfig& config = ConvConfig(),
                         int device_id = -1, void* stream = nullptr);
    
private:
    KernelManager() = default;
    ~KernelManager() = default;
    
    struct DeviceKernels {
        std::shared_ptr<MatmulKernel> matmul;
        std::shared_ptr<ConvKernel> conv;
        std::shared_ptr<CustomKernels> custom;
        bool initialized;
    };
    
    std::unordered_map<int, DeviceKernels> kernels_;
    mutable std::mutex mutex_;
};

// Utility functions for kernel operations
KernelConfig calculate_matmul_kernel_config(const TensorDesc& a, const TensorDesc& b);
KernelConfig calculate_conv_kernel_config(const TensorDesc& input, const TensorDesc& filter);
size_t get_data_type_size(int data_type);

} // namespace rdna

#endif // RDNA_KERNELS_H