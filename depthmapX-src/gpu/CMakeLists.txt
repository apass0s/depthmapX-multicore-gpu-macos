# depthmapX GPU module
#
# host/  is compiled into salalib: the multicore (OpenMP) versions of the accelerated
#        analyses and the run-time loader for the GPU module. It never links CUDA or Metal.
# cuda/  builds depthmapX_gpu (DLL / shared library) with nvcc (Windows, Linux).
# metal/ builds libdepthmapX_gpu.dylib on macOS: the same C interface (include/) on Metal,
#        which it loads at run time (dmx_metal_device_rt.cpp; or the SDK: dmx_metal_device.mm).
#        metal/emu/ runs the Metal kernels on CPU threads, to test them without a Mac.
# salalib loads the module at run time, so the programs run where it, the driver or a
# suitable GPU is missing.

target_sources(salalib PRIVATE
    host/accelcommon.cpp
    host/gpuloader.cpp
    host/visibilitybits.cpp
    host/vgavisuallocalaccel.cpp
    host/vgavisualglobalaccel.cpp
    host/vgapathaccel.cpp
    host/segmangularaccel.cpp
    host/segmtulipaccel.cpp
    host/segmtopometaccel.cpp
    host/axialintegrationaccel.cpp
    host/axialbrandes.cpp)

# Apple's clang understands OpenMP but ships no runtime: point FindOpenMP at an LLVM libomp
# (Homebrew's "libomp", or any folder with include/omp.h and lib/libomp.dylib).
if (APPLE)
    set(DEPTHMAPX_LIBOMP_ROOT "" CACHE PATH "Folder with include/omp.h and lib/libomp.dylib (macOS)")
    # Cross-builds link a text stub (tools/mac/make_tbd.py) naming @rpath/libomp.dylib instead.
    set(DEPTHMAPX_LIBOMP_LIBRARY "" CACHE FILEPATH "libomp to link (default: <root>/lib/libomp.dylib)")
    set(libomp_root "${DEPTHMAPX_LIBOMP_ROOT}")
    if (NOT libomp_root)
        foreach(candidate "$ENV{HOMEBREW_PREFIX}/opt/libomp" /opt/homebrew/opt/libomp /usr/local/opt/libomp)
            if (EXISTS "${candidate}/include/omp.h")
                set(libomp_root "${candidate}")
                break()
            endif()
        endforeach()
    endif()
    if (libomp_root AND NOT OpenMP_CXX_FLAGS)
        message(STATUS "libomp: ${libomp_root}")
        foreach(lang C CXX)
            set(OpenMP_${lang}_FLAGS "-Xpreprocessor -fopenmp -I${libomp_root}/include")
            set(OpenMP_${lang}_LIB_NAMES omp)
        endforeach()
        if (DEPTHMAPX_LIBOMP_LIBRARY)
            set(OpenMP_omp_LIBRARY "${DEPTHMAPX_LIBOMP_LIBRARY}")
        else()
            set(OpenMP_omp_LIBRARY "${libomp_root}/lib/libomp.dylib")
        endif()
    endif()
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(salalib PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: the accelerated CPU paths will run on one thread")
endif()
if (UNIX)
    target_link_libraries(salalib PUBLIC ${CMAKE_DL_LIBS})
endif()

if (CMAKE_CUDA_COMPILER)
    add_library(depthmapX_gpu SHARED
        cuda/dmx_gpu_device.cu
        cuda/dmx_vis_local.cu
        cuda/dmx_vis_global.cu)
    target_include_directories(depthmapX_gpu PRIVATE include)
    target_compile_definitions(depthmapX_gpu PRIVATE DMX_GPU_BUILDING_DLL)
    set_target_properties(depthmapX_gpu PROPERTIES
        # Turing, Ampere and Ada machine code, plus Ada PTX that newer GPUs JIT-compile
        CUDA_ARCHITECTURES "75-real;86-real;89-real;89-virtual"
        # the CUDA runtime is linked in: users only need the NVIDIA driver
        CUDA_RUNTIME_LIBRARY Static
        CUDA_STANDARD 17
        CUDA_VISIBILITY_PRESET hidden)
    # No fused multiply-add contraction: keeps any floating-point kernel bit-compatible with the CPU.
    target_compile_options(depthmapX_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

    # The loader looks next to the executable, so put a copy beside each one.
    foreach(exe_dir depthmapXcli depthmapX)
        add_custom_command(TARGET depthmapX_gpu POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/${exe_dir}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:depthmapX_gpu> ${CMAKE_BINARY_DIR}/${exe_dir}/)
    endforeach()
endif()

# Metal kernel source (dmx_metal_shared.h + dmx_metal_kernels.metal) as a C string, for
# compiling at run time; regenerated when either file changes.
function(dmx_embed_metal_source out_file)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/dmx_metal_shared.h shared_src)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/dmx_metal_kernels.metal kernels_src)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/dmx_metal_shared.h ${CMAKE_CURRENT_SOURCE_DIR}/metal/dmx_metal_kernels.metal)
    file(WRITE ${out_file} "static const char kKernelSource[] = R\"DMXMSL(${shared_src}\n${kernels_src})DMXMSL\";\n")
endfunction()

if (APPLE AND DEPTHMAPX_GPU)
    dmx_embed_metal_source(${CMAKE_CURRENT_BINARY_DIR}/dmx_metal_source.inc)
    if (DEPTHMAPX_METAL_SDK)
        add_library(depthmapX_gpu SHARED
            metal/dmx_metal_api.cpp
            metal/dmx_metal_device.mm)
        target_compile_options(depthmapX_gpu PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
        target_link_libraries(depthmapX_gpu PRIVATE "-framework Metal" "-framework Foundation")
    else()
        # Default: Metal, Foundation and the Objective-C runtime are opened at run time.
        add_library(depthmapX_gpu SHARED
            metal/dmx_metal_api.cpp
            metal/dmx_metal_device_rt.cpp)
    endif()
    target_include_directories(depthmapX_gpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR} metal)
    set_target_properties(depthmapX_gpu PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        OBJCXX_VISIBILITY_PRESET hidden
        OBJCXX_STANDARD 17
        # dlopen'ed by full path; @rpath keeps the bundle relocatable
        INSTALL_NAME_DIR "@rpath")
    if (CMAKE_HOST_WIN32)
        # cross-builds: CMake would write the Windows path separator into "@rpath\..."
        set_target_properties(depthmapX_gpu PROPERTIES NO_SONAME ON)
        target_link_options(depthmapX_gpu PRIVATE "LINKER:-install_name,@rpath/libdepthmapX_gpu.dylib")
    endif()

    # The loader looks next to the executable (and in the app's Frameworks folder, see
    # the top-level CMakeLists.txt), so put a copy beside the command line program.
    add_custom_command(TARGET depthmapX_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/depthmapXcli
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:depthmapX_gpu> ${CMAKE_BINARY_DIR}/depthmapXcli/)
endif()

# The Metal module on its CPU emulator, as a drop-in GPU module for tests: copy it, named
# depthmapX_gpu.dll / libdepthmapX_gpu.so, next to a depthmapXcli (see tools/mac/).
option(DEPTHMAPX_METAL_EMULATOR "Build the Metal module on its CPU emulator (tests without a Mac)" OFF)
if (DEPTHMAPX_METAL_EMULATOR)
    find_package(Threads REQUIRED)
    add_library(depthmapX_gpu_metalemu SHARED
        metal/dmx_metal_api.cpp
        metal/emu/dmx_metal_emu.cpp)
    target_include_directories(depthmapX_gpu_metalemu PRIVATE metal)
    target_compile_definitions(depthmapX_gpu_metalemu PRIVATE DMX_GPU_BUILDING_DLL)
    target_link_libraries(depthmapX_gpu_metalemu PRIVATE Threads::Threads)
    set_target_properties(depthmapX_gpu_metalemu PROPERTIES
        CXX_STANDARD 20 # std::barrier
        CXX_VISIBILITY_PRESET hidden
        OUTPUT_NAME depthmapX_gpu
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/metal-emu
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/metal-emu
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/metal-emu)
    if (WIN32)
        set_target_properties(depthmapX_gpu_metalemu PROPERTIES PREFIX "") # depthmapX_gpu.dll, also with MinGW
    endif()
endif()

# The SDK-free Metal device layer (dmx_metal_device_rt.cpp) on a fake Objective-C runtime
# and Metal (metal/emu/objcmock, Windows x64 with GCC or Clang) over the CPU emulator:
# tests its messages, reference counting and results without a Mac. Drop-in GPU module
# like the emulator above (output directory metal-objcmock).
option(DEPTHMAPX_METAL_OBJC_MOCK "Build the SDK-free Metal module on a fake Objective-C runtime (tests, Windows x64)" OFF)
if (DEPTHMAPX_METAL_OBJC_MOCK)
    find_package(Threads REQUIRED)
    dmx_embed_metal_source(${CMAKE_CURRENT_BINARY_DIR}/dmx_metal_source.inc)
    add_library(depthmapX_gpu_objcmock SHARED
        metal/emu/objcmock/objc_mock.cpp
        metal/emu/objcmock/emu_device.cpp
        metal/dmx_metal_api.cpp
        metal/dmx_metal_device_rt.cpp)
    # objcmock/ first: its dlfcn.h stands in for the system one
    target_include_directories(depthmapX_gpu_objcmock PRIVATE metal/emu/objcmock ${CMAKE_CURRENT_BINARY_DIR} metal)
    target_compile_definitions(depthmapX_gpu_objcmock PRIVATE DMX_GPU_BUILDING_DLL)
    target_link_libraries(depthmapX_gpu_objcmock PRIVATE Threads::Threads)
    set_target_properties(depthmapX_gpu_objcmock PROPERTIES
        CXX_STANDARD 20
        CXX_VISIBILITY_PRESET hidden
        OUTPUT_NAME depthmapX_gpu
        PREFIX ""
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/metal-objcmock
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/metal-objcmock
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/metal-objcmock)
endif()
