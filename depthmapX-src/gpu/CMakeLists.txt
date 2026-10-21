# depthmapX GPU module
#
# host/  is compiled into salalib: the multicore (OpenMP) versions of the accelerated
#        analyses and the run-time loader for the CUDA DLL. It never links CUDA.
# cuda/  builds depthmapX_gpu (DLL / shared library) with nvcc. salalib loads it at run
#        time, so binaries run on machines without CUDA.

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
