# macOS 11 is the first release on Apple Silicon, and the oldest the bundled OpenMP
# runtime (libomp) supports.
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment version")
project(depthmapX)
cmake_minimum_required(VERSION 3.13.0)
set(CMAKE_CXX_STANDARD 17)

# Without Qt (e.g. when cross-compiling) only the command line program, the libraries
# and their tests are built.
option(DEPTHMAPX_GUI "Build the depthmapX graphical interface and its tests (needs Qt 5)" ON)

# Optional GPU module (gpu/), loaded at run time: CUDA on Windows and Linux, Metal on
# macOS. Without it everything else still builds and the accelerated analyses run on
# the CPU.
option(DEPTHMAPX_GPU "Build the depthmapX_gpu module (CUDA, or Metal on macOS) when possible" ON)
if (DEPTHMAPX_GPU AND APPLE)
    # NVIDIA CUDA does not exist on macOS; gpu/metal implements the same C interface.
    enable_language(OBJCXX)
elseif (DEPTHMAPX_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(STATUS "No CUDA compiler found: building without depthmapX_gpu")
    endif()
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") # Clang and AppleClang
    set(warnings "-Wall -Wextra")
    # No fused multiply-add contraction (Clang contracts by default, and arm64 always has
    # FMA): a*b+c must round like the upstream code compiled anywhere else.
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
                        $<$<COMPILE_LANGUAGE:OBJCXX>:-ffp-contract=off>)
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
if (DEPTHMAPX_GUI)
    add_subdirectory(depthmapXTest)
    add_subdirectory(depthmapX)
    add_subdirectory(GuiUnitTest)
endif()
add_subdirectory(moduleTest)

if (APPLE)
    # A packaged depthmapXcli finds libomp.dylib next to itself (tools/mac/package_*).
    set_property(TARGET depthmapXcli APPEND PROPERTY BUILD_RPATH "@executable_path")
    set_property(TARGET depthmapXcli APPEND PROPERTY INSTALL_RPATH "@executable_path")
endif()

# Building either executable also builds the (run-time loaded) GPU module.
if (TARGET depthmapX_gpu)
    add_dependencies(depthmapXcli depthmapX_gpu)
    if (TARGET depthmapX)
        add_dependencies(depthmapX depthmapX_gpu)
        if (APPLE)
            # the loader also looks in the app bundle's Frameworks folder
            add_custom_command(TARGET depthmapX POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_BUNDLE_CONTENT_DIR:depthmapX>/Frameworks
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:depthmapX_gpu>
                        $<TARGET_BUNDLE_CONTENT_DIR:depthmapX>/Frameworks/)
        endif()
    endif()
endif()

