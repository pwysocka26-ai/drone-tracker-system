# drone-tracker C++ port

**Status**: end-to-end pipeline complete, demo-ready (testy wewnętrzne 2026-04-30)
**Hardware target**: GMKtec EVO-X2 (AMD Ryzen AI Max+ 395 + Radeon 8060S iGPU + NPU XDNA 2)
**Performance**: ~232 ms / klatka (FP16 + lazy CSRT, DirectML) — 12× nad targetem 19 ms

Pełen plan optymalizacji do 19 ms: `docs/d1_summary_2026_04_25.md` (D4 v4 yolov8s + D5 INT8 NPU)

## Struktura

```
cpp/
├── CMakeLists.txt                 # MSVC 17 2022 x64 + auto-detect OpenCV
├── include/dtracker/              # publiczne headers modułów
│   ├── types.hpp                  # BBox, Detection, IDetector
│   ├── inference.hpp              # YoloOnnxDetector (ORT + DirectML)
│   ├── kalman.hpp                 # SimpleKalman2D constant velocity
│   ├── track.hpp                  # Track struct (move-only z kalman)
│   ├── multi_target_tracker.hpp   # MTT: greedy match + Kalman per track
│   ├── target_manager.hpp         # TM: identity anchor + persistent_owner_id
│   ├── narrow_tracker.hpp         # Narrow: PID + adaptive feedforward
│   ├── lock_pipeline.hpp          # FSM: ACQUIRE/LOCKED/HOLD/REACQUIRE
│   ├── local_tracker.hpp          # CSRT recovery (real, opencv contrib)
│   ├── dashboard.hpp              # cv::imshow render
│   └── telemetry.hpp              # JSONL writer + diagnostic fields
├── src/                           # implementacje .cpp
├── app/
│   ├── poc_inference.cpp          # benchmark single-image
│   └── main.cpp                   # full end-to-end pipeline
├── tests/                         # 47/47 parity tests
│   ├── test_framework.hpp         # self-contained TEST/ASSERT macros
│   ├── test_kalman.cpp            # 8 tests
│   ├── test_mtt.cpp               # 9 tests
│   ├── test_target_manager.cpp    # 6 tests
│   ├── test_lock_pipeline.cpp     # 12 tests
│   └── test_narrow_tracker.cpp    # 12 tests
└── build/                         # MSBuild output (gitignored)
```

## Wymagania

- **Visual Studio 2022** (MSVC v143, C++17)
- **CMake 3.15+** (lokalny w `../third_party/cmake-3.31.2-windows-x86_64/`)
- **ONNX Runtime 1.20.1 DirectML** w `../third_party/onnxruntime-directml/`
  (z NuGet `Microsoft.ML.OnnxRuntime.DirectML`)
- **OpenCV 4.10 z contrib** (preferowane) w `../third_party/opencv_contrib_install/`
  (lokalny build; instrukcja w `../docs/env_setup.md` sekcja 11.4)
  - Fallback: prebuilt OpenCV 4.10 w `../third_party/opencv/` (CSRT disabled,
    LocalTracker w stub mode)

## Build

CMake auto-detects OpenCV path (prefer `opencv_contrib_install/` over `opencv/build/`)
i auto-detect vc16 vs vc17 dla DLL copy.

```bat
cd cpp
"%CMAKE%" -S . -B build -G "Visual Studio 17 2022" -A x64
"%CMAKE%" --build build --config Release
```

Build produkuje 3 executables w `build/Release/`:
- `poc_inference.exe` — single-image benchmark
- `dtracker_main.exe` — pełen end-to-end pipeline
- `dtracker_tests.exe` — parity test suite (47 tests)

Wszystkie z auto-copy DLL: `onnxruntime.dll`, `opencv_world4100.dll`,
`opencv_videoio_ffmpeg4100_64.dll`, `opencv_videoio_msmf4100_64.dll`,
ewentualnie `DirectML.dll`.

## Run — full pipeline

```bat
cd cpp\build\Release
dtracker_main.exe ^
  --video ..\..\..\artifacts\test_videos\video_test_wide_short.mp4 ^
  --model ..\..\..\data\weights\v3_best_fp16_imgsz960.onnx ^
  --out-dir ..\..\..\artifacts\runs
```

CLI flags:
- `--video PATH` — input .mp4
- `--model PATH` — ONNX (default: `v3_best_fp16_imgsz960.onnx` FP16)
- `--out-dir PATH` — gdzie zapisać `<timestamp>/{telemetry.jsonl,run_summary.json,video/}`
- `--imgsz N` — model input size (default 960)
- `--conf F` — confidence threshold (default 0.20)
- `--max-frames N` — limit klatek (default brak)
- `--no-gui` — bez `cv::imshow` (do bg runs)
- `--no-record` — bez VideoWriter
- `--cpu` — wyłącz DirectML (CPU-only, **11× wolniej**)

Keyboard runtime: `Q`/`Esc` exit, `S` screenshot, `0-9` manual lock, `,`/`.` switch.

## Run — POC benchmark

```bat
cd cpp\build\Release
poc_inference.exe ..\..\..\artifacts\cvat_import\obj_train_data\frame_000150.jpg ^
                  ..\..\..\data\weights\v3_best_fp16_imgsz960.onnx ^
                  960
```

3rd positional = imgsz lub `cpu` (disables DirectML).

## Run — parity tests

```bat
cd cpp\build\Release
dtracker_tests.exe
```

Powinno pokazać `=== 47 passed, 0 failed ===`.

## Empiryczne wyniki (2026-04-25)

### Inference latency (POC, frame_000150, GMKtec Radeon 8060S)

| variant | inference | fps |
|---|---|---|
| FP32 imgsz=960 DirectML | 185 ms | 5.4 |
| **FP16 imgsz=960 DirectML** (default) | **104 ms** | 9.5 |
| FP32 imgsz=960 CPU | 2108 ms | 0.5 |
| FP16 imgsz=960 CPU | 2096 ms | 0.5 |

DirectML to **11× speedup** vs CPU. FP16 to **1.77× speedup** vs FP32 z zerową utratą
accuracy (conf 0.077 vs 0.076, bbox identyczny).

### Pipeline 8800 klatek (FP16 + lazy CSRT)

| metryka | wartość |
|---|---|
| LOCKED frames | 41% |
| HOLD frames | 41% |
| REACQUIRE frames | 15% |
| narrow synthetic_hold | <2% (CSRT bridge) |
| unique persistent_owner_id | 13 (= 13 fizycznych dronów) |
| per-frame total | ~232 ms |

CSRT lazy mode aktywuje update tylko gdy YOLO degraded (51% klatek skipped).

### Path do 19 ms target (50 fps kamery)

| etap | inference | factor over budget |
|---|---|---|
| FP32 imgsz=960 (start D0) | 267 ms | 14× |
| FP16 imgsz=960 (D1) | 104 ms | 5.5× |
| + lazy CSRT (D1 night) | 9 ms tracker = **164 ms total** | **8.6×** |
| + v4 yolov8s @ imgsz=640 (D4) | ~30 ms | 2.1× |
| + INT8 NPU XDNA2 (D5) | ~15 ms | **within budget** |

## Architektura per-moduł — do CLAUDE.md

Pełne opisy każdego modułu w `../CLAUDE.md` sekcja "Architektura".
Visual review werdykty + diagnoza w `../memory/project_visual_review_d2_d3_2026_04_25.md`.

## Znane ograniczenia

- **Lock FSM uproszczony** — brak REFINE state pomiędzy ACQUIRE i LOCKED (Python ma 5-state).
  Impact: niski (LOCKED frames 40%+ na demo bez tego).
- **OpenCV contrib build** — DLL `opencv_videoio_msmf4100_64.dll` nie jest w nowym
  contrib install (brak Media Foundation w default config). FFmpeg działa, MP4 OK.
- **CSRT init cost** — re-init przy każdym `owner_id` change kosztuje ~5-10 ms.
  Lazy init nie zaimplementowany (drobna optymalizacja post-demo).

## Migracja na nowy komputer

Pełna instrukcja w `../docs/env_setup.md`. Szybko:
1. `git pull` ściągnie kod (cpp/, docs/, training/, tools/)
2. Fizyczny transfer ~7 GB: `third_party/`, `data/`, `artifacts/`, memory `~/.claude/`
3. Build per instrukcja powyżej
