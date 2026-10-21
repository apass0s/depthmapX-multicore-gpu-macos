# Builds the command line program (and its test suites) of an UNMODIFIED upstream
# depthmapX source tree, without Qt, as the reference for tools/gpu/validate.py
# (--upstream). Upstream's own CMakeLists.txt always builds the GUI; this wrapper adds only
# the subdirectories that do not need Qt, and changes no upstream file.
#
#   cmake -S depthmapX-src/tools/mac/upstream-cli -B build-upstream -G Ninja
#         -DCMAKE_BUILD_TYPE=Release -DUPSTREAM_DIR=<upstream depthmapX 0.9.1 source>
#         [the same -DCMAKE_TOOLCHAIN_FILE=... as the build being validated]
#
# Use the same compiler and flags as that build: results are compared bit for bit.

cmake_minimum_required(VERSION 3.13.0)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment version")
project(depthmapX_upstream_cli C CXX)
set(CMAKE_CXX_STANDARD 17)

set(UPSTREAM_DIR "" CACHE PATH "Unmodified upstream depthmapX source tree")
if (NOT EXISTS "${UPSTREAM_DIR}/salalib/mgraph.cpp")
    message(FATAL_ERROR "UPSTREAM_DIR must point to an upstream depthmapX source tree")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # as in depthmapX-src: no fused multiply-add contraction
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()
cmake_policy(SET CMP0076 NEW)

include_directories(${UPSTREAM_DIR})
if (APPLE AND CMAKE_CROSSCOMPILING)
    # Zig's macOS headers lack the (unused) <sys/timeb.h> that mgraph440/comm.h includes
    include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/stub)
endif()

set(APP_BRANCH "upstream")
set(APP_COMMIT "v0.9.1")
include_directories(${CMAKE_BINARY_DIR})
configure_file("${UPSTREAM_DIR}/version_defs.h.in" "${CMAKE_BINARY_DIR}/version_defs.h" @ONLY)
string(TIMESTAMP APP_DATE %Y-%m-%d)

set(modules_core "" CACHE INTERNAL "modules_core" FORCE)
set(MODULES_GUI FALSE)
set(MODULES_CLI FALSE)
set(MODULES_CLI_TEST FALSE)
set(MODULES_CORE TRUE)
set(MODULES_CORE_TEST FALSE)

foreach(dir genlib genlibTest salalib modules mgraph440 mgraph440Test salaTest depthmapXcli cliTest moduleTest)
    add_subdirectory(${UPSTREAM_DIR}/${dir} ${dir})
endforeach()
