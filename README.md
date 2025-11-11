# RDNA Stack - AMD GPU Acceleration for PyTorch and TensorFlow

A production-quality open-source software stack enabling PyTorch and TensorFlow users to run models on AMD consumer and datacenter GPUs using `device='rdna'` (analogous UX to `device='cuda'`).

## 🚀 Features

- **PyTorch Integration**: Complete `torch.device('rdna')` support with CUDA-like API
- **TensorFlow Integration**: Full `/device:RDNA:0` device placement support
- **High-Performance Kernels**: Optimized HIP kernels using MIOpen and rocBLAS
- **Memory Management**: Caching allocator with PyTorch-like memory API
- **Multi-Device Support**: Multi-GPU and multi-stream execution
- **Mixed Precision**: FP16/BF16 support for RDNA2/3 architectures
- **Performance Profiling**: Comprehensive benchmarking and optimization tools
- **DLPack Interop**: Zero-copy tensor sharing between frameworks

## 📋 Prerequisites

- **AMD RDNA GPU**: RX 6000/7000 series or Instinct series
- **ROCm 6.x**: AMD's open software platform for GPU computing
- **Python 3.10+**: Modern Python version with improved features

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BurnedOx/rdna_stack
cd rdna-stack

# Build from source
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python package
pip install -e .

# Verify installation
python -c "import rdna; print('RDNA Stack loaded successfully')"
```

### Usage Examples

#### PyTorch Integration
```python
import torch
import rdna

# Check availability and set device
if rdna.is_available():
    device = torch.device('rdna')
    print(f"Using RDNA device: {rdna.get_device_properties(0).name}")
else:
    device = torch.device('cpu')
    print("RDNA not available, using CPU")

# Model training with RDNA
model = torchvision.models.resnet50().to(device)
x = torch.randn(32, 3, 224, 224, device=device)
output = model(x)

# Memory management
rdna.empty_cache()
print(f"Memory allocated: {rdna.memory_allocated() / 1024**3:.2f} GB")
```

#### TensorFlow Integration
```python
import tensorflow as tf
import rdna

# Configure RDNA devices
rdna.config.set_visible_devices([0])

# Model training with RDNA
with tf.device('/device:RDNA:0'):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])
    
    # Training loop
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # ... training code
```

#### Direct Python API
```python
import rdna
import numpy as np

# Device management
print(f"RDNA available: {rdna.is_available()}")
print(f"Device count: {rdna.device_count()}")

# Performance profiling
rdna.set_profiling(True)
rdna.run_diagnostics()

# Memory operations
data = np.random.randn(1000, 1000).astype(np.float32)
# RDNA memory operations would go here
```

## 🏗️ Project Structure

```
rdna-stack/
├── include/rdna/          # C++ headers
│   ├── device.h          # Device discovery and management
│   ├── memory.h          # Memory allocator
│   ├── kernels.h         # Operator kernels
│   ├── profiler.h        # Performance profiling
│   └── utils.h           # Utilities and diagnostics
├── src/                  # C++/HIP implementation
│   ├── device.cpp        # Device management
│   ├── memory.cpp        # Memory allocator
│   ├── kernels.cpp       # Kernel implementations
│   ├── profiler.cpp      # Performance profiling
│   └── utils.cpp         # Utility functions
├── python/               # Python bindings
│   ├── rdna_module.cpp   # Main pybind11 module
│   ├── __init__.py       # Python package
│   └── CMakeLists.txt    # Python build configuration
├── pytorch/              # PyTorch backend plugin
│   └── rdna_backend.cpp  # PyTorch C++ extension
├── tensorflow/           # TensorFlow device plugin
│   └── rdna_device.cc    # TensorFlow custom ops
├── tests/                # Comprehensive test suite
├── examples/             # Usage examples and benchmarks
├── benchmarks/           # Performance benchmarks
└── docs/                 # Documentation
```

## 🔧 Supported Operations

### Core Operations
- **Matrix Multiplication (GEMM)**: rocBLAS-powered with autotuning
- **2D Convolution**: MIOpen integration with algorithm selection
- **Element-wise Operations**: Custom HIP kernels for add, multiply, etc.
- **Activation Functions**: ReLU, GELU, Softmax with fused kernels
- **Reduction Operations**: Sum, mean, max/min reductions

### Advanced Features
- **Autotuning**: Runtime optimization for kernel parameters
- **Mixed Precision**: FP16/BF16 training pipelines
- **Multi-Stream**: Concurrent kernel execution
- **Unified Memory**: CPU-GPU shared memory support
- **Performance Profiling**: Detailed timing and bandwidth metrics

## 📊 Performance Targets

- **Within 5-15% of CUDA performance** for key kernels on comparable hardware
- **Optimized for RDNA wavefront architecture** with proper workgroup sizing
- **MIOpen and rocBLAS integration** for vendor-optimized operations
- **Automatic kernel selection** based on problem size and device capabilities

## 💻 Supported Hardware

### Consumer GPUs
- AMD Radeon RX 6800/6900 XT (RDNA2)
- AMD Radeon RX 7900 XT/XTX (RDNA3)

### Datacenter GPUs
- AMD Instinct MI100/MI200 series
- Future RDNA-based Instinct accelerators

## 📦 Dependencies

- **ROCm 6.x**: HIP runtime, MIOpen, rocBLAS
- **PyTorch 2.0+** or **TensorFlow 2.12+**
- **CMake 3.18+**: Modern build system
- **pybind11**: Seamless C++/Python interop
- **Python 3.10+**: Required for new language features

## 🛠️ Building from Source

### Ubuntu 22.04+ with ROCm

```bash
# Install ROCm 6.x
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

# Build RDNA Stack
git clone https://github.com/rdna-stack/rdna-stack
cd rdna-stack
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run comprehensive tests
ctest -V
python -m pytest tests/ -v
```

### Development Build (Simulation Mode)

```bash
# Build without ROCm for development
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_SIMULATION=ON
make

# Test basic functionality
python examples/basic_usage.py
```

## 🧪 Testing & Validation

```bash
# Run unit tests
python -m pytest tests/test_basic.py -v

# Run performance benchmarks
python benchmarks/matmul_benchmark.py
python benchmarks/convolution_benchmark.py

# Validate framework integration
python examples/pytorch_integration.py
python examples/tensorflow_integration.py
```

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following Google C++ Style Guide
4. **Add tests** for new functionality
5. **Submit a pull request** with comprehensive description

### Code Standards

- **C++17**: Modern C++ with RAII and smart pointers
- **Python 3.10+**: Type hints and modern syntax
- **clang-format**: Consistent code formatting
- **Comprehensive testing**: Unit tests for all new features
- **Documentation**: Clear docstrings and API documentation

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## 🗺️ Roadmap

### v1.0.0 (Current)
- ✅ Complete PyTorch backend with device registration
- ✅ Full TensorFlow device plugin with custom ops
- ✅ Performance profiling and optimization framework
- ✅ Comprehensive test suite and benchmarks
- ✅ Production-ready packaging and documentation

### v1.1.0 (Next)
- Additional operator coverage (RNN, attention)
- Enhanced autotuning and performance optimization
- Expanded hardware support (new RDNA architectures)
- Improved debugging and diagnostics

### v2.0.0 (Future)
- JAX and other framework support
- Advanced compiler integration
- Heterogeneous computing support
- Production deployment tools

## 🆘 Support

- **[Documentation](docs/)**: Comprehensive guides and API reference
- **[Issue Tracker](https://github.com/rdna-stack/rdna-stack/issues)**: Bug reports and feature requests
- **[Discussion Forum](https://github.com/rdna-stack/rdna-stack/discussions)**: Community support and discussions
- **[Examples](examples/)**: Ready-to-run code samples

## 📚 Citing RDNA Stack

If you use RDNA Stack in your research, please cite:

```bibtex
@software{rdna_stack,
  title = {RDNA Stack: AMD GPU Acceleration for PyTorch and TensorFlow},
  author = {RDNA Stack Contributors},
  url = {https://github.com/rdna-stack/rdna-stack},
  year = {2025},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments

This project builds upon the excellent work of:
- **AMD ROCm team** for HIP, MIOpen, and rocBLAS
- **PyTorch and TensorFlow communities** for amazing ML frameworks
- **pybind11 contributors** for seamless C++/Python interop
- **Open source community** for continuous improvement

---

**Disclaimer**: This project is not officially affiliated with AMD. It uses public ROCm APIs and is community-maintained. Always check AMD's official documentation for the latest supported hardware and software versions.