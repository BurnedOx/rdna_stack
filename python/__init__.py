"""
RDNA Stack - AMD GPU acceleration for PyTorch and TensorFlow

This package provides RDNA device support for PyTorch and TensorFlow,
enabling AMD GPU acceleration with a CUDA-like API.
"""

from .rdna_py import *

__version__ = "0.1.0"
__author__ = "RDNA Stack Contributors"
__license__ = "Apache-2.0"

# Check availability on import
def _check_availability():
    """Check if RDNA devices are available and initialize the library."""
    try:
        if is_available():
            initialize()
            return True
        else:
            print("Warning: No RDNA devices found. RDNA stack will use CPU fallback.")
            return False
    except Exception as e:
        print(f"Warning: Failed to initialize RDNA stack: {e}")
        return False

# Perform availability check on import
_available = _check_availability()

def is_available():
    """Check if RDNA devices are available.
    
    Returns:
        bool: True if RDNA devices are available and initialized
    """
    return _available

def device_count():
    """Get the number of available RDNA devices.
    
    Returns:
        int: Number of RDNA devices
    """
    return rdna_py.device_count()

def current_device():
    """Get the current RDNA device.
    
    Returns:
        int: Current device ID
    """
    # This will be implemented in the C++ bindings
    return 0

def set_device(device_id):
    """Set the current RDNA device.
    
    Args:
        device_id (int): Device ID to set as current
    """
    # This will be implemented in the C++ bindings
    pass

# PyTorch-like API
class device:
    """Context manager for device placement (PyTorch-style)."""
    
    def __init__(self, device_str):
        if device_str.startswith('rdna'):
            if ':' in device_str:
                self.device_id = int(device_str.split(':')[1])
            else:
                self.device_id = 0
        else:
            raise ValueError(f"Unsupported device: {device_str}")
    
    def __enter__(self):
        self.prev_device = current_device()
        set_device(self.device_id)
        return self
    
    def __exit__(self, *args):
        set_device(self.prev_device)

# TensorFlow-like API
class tf_device:
    """Context manager for device placement (TensorFlow-style)."""
    
    def __init__(self, device_str):
        if device_str.startswith('/device:RDNA'):
            parts = device_str.split(':')
            if len(parts) > 2:
                self.device_id = int(parts[2])
            else:
                self.device_id = 0
        else:
            raise ValueError(f"Unsupported device: {device_str}")
    
    def __enter__(self):
        self.prev_device = current_device()
        set_device(self.device_id)
        return self
    
    def __exit__(self, *args):
        set_device(self.prev_device)

# Convenience functions for PyTorch users
def is_rdna_available():
    """Check if RDNA is available (PyTorch-compatible)."""
    return is_available()

def rdna_device_count():
    """Get RDNA device count (PyTorch-compatible)."""
    return device_count()

def empty_cache():
    """Empty the memory cache (PyTorch-compatible)."""
    # This will be implemented in the C++ bindings
    pass

def memory_allocated(device_id=None):
    """Get memory allocated on device (PyTorch-compatible)."""
    # This will be implemented in the C++ bindings
    return 0

def max_memory_allocated(device_id=None):
    """Get maximum memory allocated on device (PyTorch-compatible)."""
    # This will be implemented in the C++ bindings
    return 0

def get_device_properties(device_id=None):
    """Get device properties (PyTorch-compatible)."""
    # This will be implemented in the C++ bindings
    return {}

# Convenience functions for TensorFlow users
class config:
    """Configuration class for TensorFlow-style API."""
    
    @staticmethod
    def set_visible_devices(devices):
        """Set visible RDNA devices.
        
        Args:
            devices (list): List of device IDs to make visible
        """
        # This will be implemented in the C++ bindings
        pass
    
    @staticmethod
    def get_memory_info(device_id=None):
        """Get memory information for device.
        
        Args:
            device_id (int, optional): Device ID. Defaults to current device.
            
        Returns:
            dict: Memory information dictionary
        """
        # This will be implemented in the C++ bindings
        return {}

# Export public API
__all__ = [
    'is_available',
    'device_count',
    'current_device',
    'set_device',
    'device',
    'tf_device',
    'is_rdna_available',
    'rdna_device_count',
    'empty_cache',
    'memory_allocated',
    'max_memory_allocated',
    'get_device_properties',
    'config',
    'diagnostics',
    'set_debug_logging',
    'set_profiling',
    'set_memory_cache_limit',
]