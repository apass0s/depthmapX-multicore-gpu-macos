# CMake toolchain (tests only): the Zig 0.14.1 clang + libc++ toolchain used for the
# macOS cross-builds, targeting Windows so the result can run here. It checks the code
# with Apple's compiler family and standard library (libc++), where MSVC and its STL
# might hide differences. Zig has no OpenMP runtime for Windows, so this build also
# exercises the code path without OpenMP (one thread).
#
#   cmake -S depthmapX-src -B build-zigwin -G Ninja -DCMAKE_BUILD_TYPE=Release
#         -DCMAKE_TOOLCHAIN_FILE=depthmapX-src/tools/mac/zig-windows.cmake -DZIG=<zig.exe>
#         -DDEPTHMAPX_GUI=OFF -DDEPTHMAPX_GPU=OFF [-DDEPTHMAPX_METAL_EMULATOR=ON]

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

if (NOT ZIG)
    set(ZIG "$ENV{ZIG}")
endif()
if (NOT ZIG)
    message(FATAL_ERROR "zig-windows.cmake needs -DZIG=<zig executable>")
endif()
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES ZIG)

set(CMAKE_C_COMPILER "${ZIG}" cc -target x86_64-windows-gnu)
set(CMAKE_CXX_COMPILER "${ZIG}" c++ -target x86_64-windows-gnu)

set(wrapper_dir "${CMAKE_BINARY_DIR}/zig-wrappers")
file(WRITE "${wrapper_dir}/zig-ar.cmd" "@\"${ZIG}\" ar %*\r\n")
file(WRITE "${wrapper_dir}/zig-ranlib.cmd" "@\"${ZIG}\" ranlib %*\r\n")
set(CMAKE_AR "${wrapper_dir}/zig-ar.cmd" CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB "${wrapper_dir}/zig-ranlib.cmd" CACHE FILEPATH "" FORCE)

# no OpenMP runtime for this target: do not pick up the host's
set(OpenMP_CXX_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_DISABLE_FIND_PACKAGE_OpenMP TRUE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
