# Source directory CMake configuration

# Add HIP/C++ source files
add_library(rdna-core STATIC
    device.cpp
    memory.cpp
    kernels.cpp
    utils.cpp
)

# Set target properties
target_include_directories(rdna-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(rdna-core PUBLIC
    hip::host
    roc::rocblas
    MIOpen
)

target_compile_definitions(rdna-core PUBLIC
    RDNA_CORE_EXPORT
)

target_compile_options(rdna-core PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

# Set output directory
set_target_properties(rdna-core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
)

# Install the library
install(TARGETS rdna-core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/rdna
    DESTINATION include
)