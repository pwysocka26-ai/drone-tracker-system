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

# OpenCV: prefer opencv_contrib_install (z contrib modulami w tym tracking dla CSRT)
# jesli zostala lokalnie zbudowana, inaczej fallback na prebuilt third_party/opencv.
# Jesli kompilacja contrib niedostepna, kompiluj local_tracker w stub-mode.
if (NOT OpenCV_DIR)
    set(_contrib_install "${CMAKE_SOURCE_DIR}/../third_party/opencv_contrib_install")
    if (EXISTS "${_contrib_install}")
        set(OpenCV_DIR "${_contrib_install}" CACHE PATH "OpenCV build dir")
        message(STATUS "Using OpenCV with contrib (local build): ${_contrib_install}")
    else()
        set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../third_party/opencv/build" CACHE PATH "OpenCV build dir")
        message(STATUS "Using OpenCV prebuilt (no contrib, CSRT disabled): ${OpenCV_DIR}")
    endif()
endif()

# Probuj znalezc tracking module (z contrib). Z BUILD_opencv_world=ON nie ma
# osobnego targetu opencv_tracking, sprawdzamy header file zamiast.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
set(_tracking_hdr_candidates
    "${OpenCV_INCLUDE_DIRS}/opencv2/tracking.hpp"
)
foreach(_dir ${OpenCV_INCLUDE_DIRS})
    list(APPEND _tracking_hdr_candidates "${_dir}/opencv2/tracking.hpp")
endforeach()
set(_tracking_found FALSE)
foreach(_hdr ${_tracking_hdr_candidates})
    if (EXISTS "${_hdr}")
        set(_tracking_found TRUE)
        break()
    endif()
endforeach()
if (_tracking_found)
    message(STATUS "OpenCV tracking module header found (CSRT enabled)")
else()
    message(STATUS "OpenCV tracking module NOT found -- LocalTracker compiled in stub mode")
    add_compile_definitions(DTRACKER_NO_OPENCV_CONTRIB)
endif()

# ======== Core library ========

add_library(dtracker_lib STATIC
    src/inference.cpp
    src/kalman.cpp
    src/multi_target_tracker.cpp
    src/target_manager.cpp
    src/narrow_tracker.cpp
    src/lock_pipeline.cpp
    src/local_tracker.cpp
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
    # Auto-detect vc16 (VS 2019 prebuilt) vs vc17 (VS 2022 lokalny build).
    set(_opencv_bin "")
    foreach(_vc vc17 vc16)
        if (EXISTS "${OpenCV_DIR}/x64/${_vc}/bin/opencv_world4100.dll")
            set(_opencv_bin "${OpenCV_DIR}/x64/${_vc}/bin")
            break()
        endif()
    endforeach()
    if (_opencv_bin)
        add_custom_command(TARGET ${_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_opencv_bin}/opencv_world4100.dll" $<TARGET_FILE_DIR:${_target}>
            COMMENT "Copy opencv_world4100.dll -> ${_target}"
        )
        # OpenCV videoio pluginy (ffmpeg + msmf) -- ladowane dynamicznie
        foreach(_plugin opencv_videoio_ffmpeg4100_64.dll opencv_videoio_msmf4100_64.dll)
            if (EXISTS "${_opencv_bin}/${_plugin}")
                add_custom_command(TARGET ${_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_opencv_bin}/${_plugin}" $<TARGET_FILE_DIR:${_target}>
                    COMMENT "Copy ${_plugin} -> ${_target}"
                )
            endif()
        endforeach()
    endif()
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

# ======== Tests ========
# Self-contained test runner (test_framework.hpp). Brak zewnetrznych deps.
add_executable(dtracker_tests
    tests/test_main.cpp
    tests/test_kalman.cpp
    tests/test_mtt.cpp
    tests/test_target_manager.cpp
)
target_include_directories(dtracker_tests PRIVATE tests)
target_link_libraries(dtracker_tests PRIVATE dtracker_lib)
dtracker_copy_runtime_dlls(dtracker_tests)
