# Python bindings CMake configuration

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(rdna_py
    rdna_module.cpp
    device_bindings.cpp
    memory_bindings.cpp
    kernel_bindings.cpp
    utils_bindings.cpp
)

# Link with core library
target_link_libraries(rdna_py PRIVATE
    rdna-core
    pybind11::module
)

# Set target properties
target_include_directories(rdna_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_definitions(rdna_py PRIVATE
    VERSION_INFO=${RDNA_STACK_VERSION}
)

# Install the Python module
install(TARGETS rdna_py
    LIBRARY DESTINATION lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
)

# Install Python package files
install(FILES
    __init__.py
    DESTINATION lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/rdna
)

# Set output directory
set_target_properties(rdna_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)