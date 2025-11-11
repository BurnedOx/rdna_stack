#!/usr/bin/env python3

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import subprocess
import os
from pathlib import Path

# Read version from CMakeLists.txt
def get_version():
    cmake_file = Path("CMakeLists.txt")
    if cmake_file.exists():
        with open(cmake_file) as f:
            for line in f:
                if line.strip().startswith("set(RDNA_STACK_VERSION"):
                    version = line.split()[-1].strip(")")
                    return version
    return "0.1.0"

# Custom build extension for CMake
class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]

        build_args = [
            '--config', 'Release',
            '--', '-j2'
        ]

        # Create build directory
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        
        # Configure and build
        subprocess.check_call(['cmake', '-S', '.', '-B', build_temp] + cmake_args)
        subprocess.check_call(['cmake', '--build', build_temp] + build_args)

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rdna-stack',
    version=get_version(),
    description='AMD GPU acceleration for PyTorch and TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='RDNA Stack Contributors',
    author_email='mainakdebnath@example.com',
    url='https://github.com/rdna-stack/rdna-stack',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='amd gpu rdna pytorch tensorflow machine-learning deep-learning',
    packages=['rdna'],
    package_dir={'rdna': 'python'},
    ext_modules=[CMakeExtension('rdna_py')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
    ],
    extras_require={
        'pytorch': ['torch>=2.0.0'],
        'tensorflow': ['tensorflow>=2.12.0'],
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
            'mypy',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme',
            'sphinx-autodoc-typehints',
        ],
    },
    entry_points={
        'console_scripts': [
            'rdna-diagnostics=rdna:run_diagnostics',
        ],
    },
    zip_safe=False,
)