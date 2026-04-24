# drone-tracker C++ port

**Status**: in progress (start 2026-04-24)
**Target**: 19 ms / klatka @ GMKtec EVO-X2 (AMD Ryzen AI Max+ 395, Radeon 8060S + NPU XDNA2), testy wewnętrzne 30.04.

## Struktura

```
cpp/
├── CMakeLists.txt
├── include/dtracker/       # publiczne headers modułów
│   ├── types.hpp           # BBox, Detection, IDetector interface
│   ├── inference.hpp       # YoloOnnxDetector (ONNX Runtime + DirectML)
│   ├── track.hpp           # (TBD D2)
│   ├── kalman.hpp          # (TBD D2)
│   ├── multi_target_tracker.hpp # (TBD D3)
│   ├── target_manager.hpp  # (TBD D3)
│   ├── narrow_tracker.hpp  # (TBD D4)
│   ├── lock_pipeline.hpp   # (TBD D4) FSM
│   ├── local_tracker.hpp   # (TBD D5) CSRT recovery
│   ├── dashboard.hpp       # (TBD D5)
│   └── telemetry.hpp       # (TBD D5)
├── src/                    # implementacje .cpp
├── app/                    # main.cpp / poc_inference.cpp
└── tests/                  # unit / integration testy
```

## Wymagania środowiska (Windows)

- **Visual Studio 2022** (MSVC v143, C++17)
- **CMake 3.15+**
- **ONNX Runtime 1.20.1 DirectML** — pobrane do `../third_party/onnxruntime-directml/` (z NuGet Microsoft.ML.OnnxRuntime.DirectML)
- **OpenCV 4.10 Windows prebuilt** — rozpakowane do `../third_party/opencv/` (z https://github.com/opencv/opencv/releases)

## Build

Z PowerShell albo cmd (MSVC Developer Command Prompt):

```bat
cd cpp
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Build produkuje `build/Release/poc_inference.exe` z wymaganymi DLL skopiowanymi obok.

## Uruchomienie POC

```bat
cd cpp\build\Release
poc_inference.exe ..\..\..\artifacts\cvat_import\obj_train_data\frame_000050.jpg ..\..\..\data\weights\v3_best.onnx
```

Powinno wyprintować avg inference time + listę bboxów + zapisać `poc_inference_out.jpg` z wizualizacją.

## Oczekiwane wyniki benchmark (POC, v3 yolov8m @ imgsz=640)

| execution provider | ~ms / klatka | notatka |
|---|---|---|
| CPU (16C Zen 5) | ~80-150 ms | fallback |
| DirectML (Radeon 8060S iGPU) | ~15-30 ms | oczekiwany target |
| NPU XDNA 2 (INT8 quant, później) | <10 ms | stretch goal |

Target 19 ms całego pipeline = YOLO ≤15 ms + tracker ≤4 ms.
