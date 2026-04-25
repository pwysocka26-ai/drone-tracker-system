# Drone tracker C++ port (in progress).
# Build:
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:/Users/pwyso/drone-tracker-system/third_party/opencv/build"
#   cmake --build build --config Release

cmake_minimum_required(VERSION 3.15)
project(dtracker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
endif()

# OpenCV + Windows API: unikaj kolizji makr (min/max) i ciezkich naglowkow
if (WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ======== Dependencies ========

# ONNX Runtime + DirectML (local, extracted from NuGet)
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-directml"
    CACHE PATH "ONNX Runtime DirectML root")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/build/native/include")
set(ORT_LIB "${ORT_ROOT}/runtimes/win-x64/native/onnxruntime.lib")
set(ORT_DLL "${ORT_ROOT}/runtimes/win-x64/native/onnxruntime.dll")

if (NOT EXISTS "${ORT_LIB}")
    message(FATAL_ERROR "ONNX Runtime DirectML nie znaleziony w ${ORT_ROOT}. Pobierz z nuget.org (Microsoft.ML.OnnxRuntime.DirectML) i rozpakuj do ${ORT_ROOT}.")
endif()

# OpenCV (prebuilt Windows)
# Domyslnie szuka w third_party/opencv/build; mozna nadpisac -DOpenCV_DIR=...
if (NOT OpenCV_DIR)
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../third_party/opencv/build" CACHE PATH "OpenCV build dir")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)

# ======== Core library ========

add_library(dtracker_lib STATIC
    src/inference.cpp
    src/kalman.cpp
    src/multi_target_tracker.cpp
    src/target_manager.cpp
    src/narrow_tracker.cpp
    src/lock_pipeline.cpp
    src/dashboard.cpp
    src/telemetry.cpp
)
target_include_directories(dtracker_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${ORT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(dtracker_lib
    PUBLIC
        ${OpenCV_LIBS}
        ${ORT_LIB}
)

# ======== Apps ========

add_executable(poc_inference app/poc_inference.cpp)
target_link_libraries(poc_inference PRIVATE dtracker_lib)

add_executable(dtracker_main app/main.cpp)
target_link_libraries(dtracker_main PRIVATE dtracker_lib)

# Windows: skopiuj ONNX Runtime DLL + OpenCV DLL do output dir kazdego exe.
# Funkcja, zeby nie duplikowac.
function(dtracker_copy_runtime_dlls _target)
    if (NOT WIN32)
        return()
    endif()
    add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" $<TARGET_FILE_DIR:${_target}>
        COMMENT "Copy onnxruntime.dll -> ${_target}"
    )
    set(_opencv_dll "${OpenCV_DIR}/x64/vc16/bin/opencv_world4100.dll")
    if (EXISTS "${_opencv_dll}")
        add_custom_command(TARGET ${_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_opencv_dll}" $<TARGET_FILE_DIR:${_target}>
            COMMENT "Copy opencv_world4100.dll -> ${_target}"
        )
    endif()
    # OpenCV videoio pluginy (ffmpeg + media foundation) -- ladowane dynamicznie
    # przy cv::VideoCapture / VideoWriter. Bez nich .mp4 nie otworzy.
    foreach(_plugin opencv_videoio_ffmpeg4100_64.dll opencv_videoio_msmf4100_64.dll)
        set(_plugin_path "${OpenCV_DIR}/x64/vc16/bin/${_plugin}")
        if (EXISTS "${_plugin_path}")
            add_custom_command(TARGET ${_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_plugin_path}" $<TARGET_FILE_DIR:${_target}>
                COMMENT "Copy ${_plugin} -> ${_target}"
            )
        endif()
    endforeach()
    # DirectML DLL (jesli istnieje obok ORT)
    set(_dml_dll "${ORT_ROOT}/runtimes/win-x64/native/DirectML.dll")
    if (EXISTS "${_dml_dll}")
        add_custom_command(TARGET ${_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dml_dll}" $<TARGET_FILE_DIR:${_target}>
            COMMENT "Copy DirectML.dll -> ${_target}"
        )
    endif()
endfunction()

dtracker_copy_runtime_dlls(poc_inference)
dtracker_copy_runtime_dlls(dtracker_main)
