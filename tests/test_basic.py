#!/usr/bin/env python3
"""
Basic tests for RDNA Stack core functionality
These tests can run without ROCm installation (simulation mode)
"""

import unittest
import sys
import os

# Add the project root to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import rdna
except ImportError:
    # If the module isn't built, we'll skip tests
    pass


@unittest.skipIf('rdna' not in sys.modules, "RDNA module not available")
class TestRDNABasic(unittest.TestCase):
    
    def test_import(self):
        """Test that the module can be imported"""
        self.assertTrue('rdna' in sys.modules)
    
    def test_is_available(self):
        """Test device availability check"""
        # This should work even in simulation mode
        available = rdna.is_available()
        self.assertIsInstance(available, bool)
    
    def test_device_count(self):
        """Test device count function"""
        count = rdna.device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)
    
    def test_memory_functions(self):
        """Test memory management functions"""
        # These should work in simulation mode
        allocated = rdna.memory_allocated()
        self.assertIsInstance(allocated, int)
        
        cached = rdna.memory_cached()
        self.assertIsInstance(cached, int)
        
        # Test empty cache (should not raise)
        rdna.empty_cache()
    
    def test_version_info(self):
        """Test version information functions"""
        version = rdna.get_library_version()
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)
        
        roc_version = rdna.get_roc_version()
        self.assertIsInstance(roc_version, str)
        
        hip_version = rdna.get_hip_version()
        self.assertIsInstance(hip_version, str)
    
    def test_diagnostics(self):
        """Test diagnostic functions (should not raise)"""
        # These functions should run without errors in simulation mode
        rdna.print_system_info()
        rdna.print_memory_info()
        rdna.run_diagnostics()


class TestRDNAAPISimulation(unittest.TestCase):
    """Tests that demonstrate the API structure without requiring ROCm"""
    
    def test_api_structure(self):
        """Verify that expected API functions exist"""
        api_functions = [
            'is_available', 'device_count', 'current_device', 'set_device',
            'empty_cache', 'memory_allocated', 'max_memory_allocated',
            'get_device_properties', 'get_roc_version', 'get_hip_version',
            'get_library_version', 'diagnostics', 'set_debug_logging',
            'set_profiling', 'set_memory_cache_limit'
        ]
        
        for func_name in api_functions:
            self.assertTrue(hasattr(rdna, func_name), 
                          f"Missing API function: {func_name}")
    
    def test_device_context_managers(self):
        """Test device context manager API structure"""
        # PyTorch-style device context
        self.assertTrue(hasattr(rdna, 'device'))
        
        # TensorFlow-style device context  
        self.assertTrue(hasattr(rdna, 'tf_device'))
        
        # Configuration API
        self.assertTrue(hasattr(rdna, 'config'))


if __name__ == '__main__':
    # Check if we can import rdna, otherwise skip tests
    try:
        import rdna
        print("Running RDNA Stack tests...")
        unittest.main(verbosity=2)
    except ImportError:
        print("RDNA module not available. Tests skipped.")
        print("To run tests, build the project first:")
        print("mkdir build && cd build && cmake .. && make")