# CMake toolchain: cross-compile for macOS from Windows (or Linux) with Zig 0.14.1.
#
# Zig bundles clang, libc++ and the macOS system library stubs, so no Mac and no Xcode
# SDK is needed for the command line program and the libraries. It has no Qt, Cocoa or
# Metal framework headers: configure with -DDEPTHMAPX_GUI=OFF -DDEPTHMAPX_GPU=OFF.
#
#   cmake -S depthmapX-src -B build-mac-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release
#         -DCMAKE_TOOLCHAIN_FILE=depthmapX-src/tools/mac/zig-macos.cmake
#         -DZIG=<path to zig(.exe)> -DZIG_MAC_ARCH=arm64     (or x86_64)
#         -DDEPTHMAPX_LIBOMP_ROOT=<folder with include/omp.h and lib/libomp.dylib>
#         -DDEPTHMAPX_GUI=OFF -DDEPTHMAPX_GPU=OFF
#
# Zig 0.16 fails to link for macOS ("failed to parse TBD file"); 0.14.1 works.

set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_VERSION 20.0) # Darwin 20 = macOS 11

if (NOT ZIG)
    set(ZIG "$ENV{ZIG}")
endif()
if (NOT ZIG_MAC_ARCH)
    set(ZIG_MAC_ARCH "$ENV{ZIG_MAC_ARCH}")
endif()
if (NOT ZIG OR NOT ZIG_MAC_ARCH)
    message(FATAL_ERROR "zig-macos.cmake needs -DZIG=<zig executable> and -DZIG_MAC_ARCH=arm64|x86_64")
endif()
# try_compile projects get the same settings
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES ZIG ZIG_MAC_ARCH)

if (ZIG_MAC_ARCH STREQUAL "arm64")
    set(zig_target aarch64-macos.11.0-none)
    set(CMAKE_SYSTEM_PROCESSOR arm64)
elseif (ZIG_MAC_ARCH STREQUAL "x86_64")
    set(zig_target x86_64-macos.11.0-none)
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
else()
    message(FATAL_ERROR "ZIG_MAC_ARCH must be arm64 or x86_64, not ${ZIG_MAC_ARCH}")
endif()

set(CMAKE_C_COMPILER "${ZIG}" cc -target ${zig_target})
set(CMAKE_CXX_COMPILER "${ZIG}" c++ -target ${zig_target})
# Objective-C++ (the Metal module) compiles too, but Zig has no Foundation or Metal
# headers: building depthmapX_gpu fails at #import, so cross-builds use -DDEPTHMAPX_GPU=OFF.
set(CMAKE_OBJCXX_COMPILER "${ZIG}" c++ -target ${zig_target})

# Ninja needs single executables for the archiver: small wrapper scripts around zig.
set(wrapper_dir "${CMAKE_BINARY_DIR}/zig-wrappers")
if (CMAKE_HOST_WIN32)
    file(WRITE "${wrapper_dir}/zig-ar.cmd" "@\"${ZIG}\" ar %*\r\n")
    file(WRITE "${wrapper_dir}/zig-ranlib.cmd" "@\"${ZIG}\" ranlib %*\r\n")
    set(CMAKE_AR "${wrapper_dir}/zig-ar.cmd" CACHE FILEPATH "" FORCE)
    set(CMAKE_RANLIB "${wrapper_dir}/zig-ranlib.cmd" CACHE FILEPATH "" FORCE)
    # CMake requires install_name_tool for Objective-C++; only `cmake --install` would
    # run it, which cross-builds never do.
    file(WRITE "${wrapper_dir}/no-install-name-tool.cmd"
         "@echo install_name_tool is not available in a cross-build 1>&2\r\n@exit /b 1\r\n")
    set(CMAKE_INSTALL_NAME_TOOL "${wrapper_dir}/no-install-name-tool.cmd" CACHE FILEPATH "" FORCE)
else()
    file(WRITE "${wrapper_dir}/zig-ar" "#!/bin/sh\nexec \"${ZIG}\" ar \"$@\"\n")
    file(WRITE "${wrapper_dir}/zig-ranlib" "#!/bin/sh\nexec \"${ZIG}\" ranlib \"$@\"\n")
    file(CHMOD "${wrapper_dir}/zig-ar" "${wrapper_dir}/zig-ranlib" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    set(CMAKE_AR "${wrapper_dir}/zig-ar" CACHE FILEPATH "" FORCE)
    set(CMAKE_RANLIB "${wrapper_dir}/zig-ranlib" CACHE FILEPATH "" FORCE)
endif()

# No SDK: do not let CMake ask xcrun / xcode-select for one.
set(CMAKE_OSX_SYSROOT "" CACHE PATH "" FORCE)
set(CMAKE_OSX_ARCHITECTURES "" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
