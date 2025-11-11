cmake_minimum_required(VERSION 3.18)
project(rdna-stack LANGUAGES CXX HIP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build configuration
set(CMAKE_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Project version
set(RDNA_STACK_VERSION_MAJOR 0)
set(RDNA_STACK_VERSION_MINOR 1)
set(RDNA_STACK_VERSION_PATCH 0)
set(RDNA_STACK_VERSION ${RDNA_STACK_VERSION_MAJOR}.${RDNA_STACK_VERSION_MINOR}.${RDNA_STACK_VERSION_PATCH})

# ROCm and HIP configuration
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(MIOpen REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(tests)
add_subdirectory(examples)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rdna-stack-config-version.cmake"
    VERSION ${RDNA_STACK_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install configuration
install(DIRECTORY include/ DESTINATION include)

# Export targets
export(EXPORT rdna-stack-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/rdna-stack-targets.cmake"
    NAMESPACE rdna::
)

# Create package
set(ConfigPackageLocation lib/cmake/rdna-stack)
install(EXPORT rdna-stack-targets
    FILE rdna-stack-targets.cmake
    NAMESPACE rdna::
    DESTINATION ${ConfigPackageLocation}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/rdna-stack-config-version.cmake"
    DESTINATION ${ConfigPackageLocation}
)

# Summary
message(STATUS "RDNA Stack Configuration:")
message(STATUS "  Version: ${RDNA_STACK_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  ROCm found: ${hip_FOUND}")
message(STATUS "  rocBLAS found: ${rocblas_FOUND}")
message(STATUS "  MIOpen found: ${MIOpen_FOUND}")