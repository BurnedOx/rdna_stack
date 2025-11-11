#!/usr/bin/env python3
"""
Basic usage example for RDNA Stack
Demonstrates device discovery, memory management, and basic operations
"""

import sys
import os

# Add the project root to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import rdna
    print("✅ RDNA Stack imported successfully")
except ImportError as e:
    print(f"❌ Failed to import RDNA Stack: {e}")
    print("Make sure the project is built and Python bindings are available")
    sys.exit(1)

def demonstrate_device_discovery():
    """Demonstrate device discovery and properties"""
    print("\n=== Device Discovery ===")
    
    # Check availability
    available = rdna.is_available()
    print(f"RDNA available: {available}")
    
    if not available:
        print("No RDNA devices found. Running in simulation mode.")
        return
    
    # Get device count
    count = rdna.device_count()
    print(f"Number of RDNA devices: {count}")
    
    # Get device properties
    for i in range(count):
        props = rdna.get_device_properties(i)
        print(f"Device {i}: {props.name}")
        print(f"  Architecture: {props.arch}")
        print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
        print(f"  Compute Units: {props.compute_units}")
        print(f"  FP16 Support: {props.supports_fp16}")
        print(f"  BF16 Support: {props.supports_bf16}")

def demonstrate_memory_management():
    """Demonstrate memory management API"""
    print("\n=== Memory Management ===")
    
    # Memory statistics
    allocated = rdna.memory_allocated()
    cached = rdna.memory_cached()
    print(f"Memory allocated: {allocated / (1024**2):.2f} MB")
    print(f"Memory cached: {cached / (1024**2):.2f} MB")
    
    # Memory summary
    summary = rdna.memory_summary()
    print("Memory summary:")
    print(summary)
    
    # Empty cache
    rdna.empty_cache()
    print("Cache emptied")

def demonstrate_kernel_operations():
    """Demonstrate basic kernel operations"""
    print("\n=== Kernel Operations ===")
    
    # Note: Actual kernel operations require ROCm installation
    # This demonstrates the API structure
    
    print("Kernel operations would be available with ROCm installation")
    print("Operations include: matmul, convolution, element-wise ops")

def demonstrate_pytorch_style():
    """Demonstrate PyTorch-style API usage"""
    print("\n=== PyTorch-style API ===")
    
    # Device context manager
    try:
        with rdna.device('rdna:0'):
            print("Operations inside rdna device context")
    except Exception as e:
        print(f"Device context simulation: {e}")
    
    # Memory management (PyTorch compatible)
    rdna.empty_cache()
    print("Cache emptied (PyTorch style)")

def demonstrate_tensorflow_style():
    """Demonstrate TensorFlow-style API usage"""
    print("\n=== TensorFlow-style API ===")
    
    # Device context manager
    try:
        with rdna.tf_device('/device:RDNA:0'):
            print("Operations inside TensorFlow RDNA device context")
    except Exception as e:
        print(f"Device context simulation: {e}")
    
    # Configuration
    try:
        rdna.config.set_visible_devices([0])
        print("Visible devices configured")
    except Exception as e:
        print(f"Configuration simulation: {e}")

def run_diagnostics():
    """Run comprehensive diagnostics"""
    print("\n=== Diagnostics ===")
    rdna.diagnostics()

def main():
    """Main demonstration function"""
    print("RDNA Stack Basic Usage Demo")
    print("=" * 40)
    
    # Run demonstrations
    demonstrate_device_discovery()
    demonstrate_memory_management()
    demonstrate_kernel_operations()
    demonstrate_pytorch_style()
    demonstrate_tensorflow_style()
    run_diagnostics()
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")
    print("\nNext steps:")
    print("1. Install ROCm 6.x for actual GPU support")
    print("2. Build the project with: mkdir build && cd build && cmake .. && make")
    print("3. Install Python package: pip install -e .")
    print("4. Run with actual AMD RDNA hardware")

if __name__ == "__main__":
    main()