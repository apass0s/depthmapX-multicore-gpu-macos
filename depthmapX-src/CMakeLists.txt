set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum macOS deployment version")
project(depthmapX)
cmake_minimum_required(VERSION 3.13.0)
set(CMAKE_CXX_STANDARD 17)

# Optional CUDA module (gpu/); without a CUDA compiler everything else still builds
# and the accelerated analyses run on the CPU.
option(DEPTHMAPX_GPU "Build the depthmapX_gpu CUDA module when a CUDA compiler is found" ON)
if (DEPTHMAPX_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(STATUS "No CUDA compiler found: building without depthmapX_gpu")
    endif()
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /EHsc")
    execute_process(COMMAND make_version_header.bat WORKING_DIRECTORY depthmapX)
endif()

# policy for target sources - we don't expect any old CMakes
cmake_policy(SET CMP0076 NEW)

include_directories(".")

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE APP_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE APP_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# generate version_defs.h
include_directories(${CMAKE_BINARY_DIR})
configure_file("${CMAKE_SOURCE_DIR}/version_defs.h.in" "${CMAKE_BINARY_DIR}/version_defs.h" @ONLY)

string(TIMESTAMP APP_DATE %Y-%m-%d)

set(modules_core "" CACHE INTERNAL "modules_core" FORCE)
set(MODULES_GUI FALSE)
set(MODULES_CLI FALSE)
set(MODULES_CLI_TEST FALSE)
set(MODULES_CORE TRUE)
set(MODULES_CORE_TEST FALSE)

add_subdirectory(genlib)
add_subdirectory(genlibTest)
add_subdirectory(salalib)
add_subdirectory(gpu)
add_subdirectory(modules) # only the core modules are loaded here
add_subdirectory(mgraph440)
add_subdirectory(mgraph440Test)
add_subdirectory(salaTest)
add_subdirectory(depthmapXcli)
add_subdirectory(cliTest)
add_subdirectory(depthmapXTest)
add_subdirectory(depthmapX)
add_subdirectory(GuiUnitTest)
add_subdirectory(moduleTest)

# Building either executable also builds the (run-time loaded) CUDA module.
if (TARGET depthmapX_gpu)
    add_dependencies(depthmapXcli depthmapX_gpu)
    add_dependencies(depthmapX depthmapX_gpu)
endif()

