# Dual-camera HAL — Hardware Abstraction Layer

**Status**: Phase 1 design done (2026-04-26).
**Cel**: tracker jako blackbox z możliwością integracji dowolnej kamery / PTZ przez vendor adapter, bez zmian w core algorytmie.

---

## Problem

Wymóg klienta: tracker ma sterować dwie kamery (wide + narrow z PTZ głowicy) w produkcji. Sprzęt może być różny dziś (np. FLIR PTZ + GigE wide camera) i jutro inny (np. niestandardowa kamera z własnym SDK CameraLink + RS485 PTZ Pelco-D). Tracker **nie powinien być przepisany** dla każdego nowego hardware.

## Rozwiązanie — 3-warstwowa architektura

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Core Tracker (dtracker_lib)                        │
│  ────────────────────────────────────────                    │
│  YOLO + MTT + TargetManager + LockPipeline + NarrowTracker   │
│  + LocalTargetTracker (CSRT)                                  │
│                                                               │
│  ZERO knowledge of hardware. Operuje na cv::Mat klatkach     │
│  i zwraca decyzje (bbox owner, suggested PTZ command).       │
└─────────────────────────────────────────────────────────────┘
              ↑                              ↑
              │ uses                         │ uses
   ┌─────────────────────────┐    ┌─────────────────────────┐
   │ Layer 2: HAL interfaces │    │ Layer 2: HAL interfaces │
   │ IFrameSource             │    │ IPTZController           │
   │ ────────                  │    │ ───────────              │
   │  open(uri)                │    │  connect(uri)            │
   │  read(Frame&)             │    │  set_pan_tilt_velocity   │
   │  info() -> {w,h,fps,...}  │    │  absolute_pan_tilt       │
   │  close()                  │    │  set_zoom                │
   │                           │    │  get_state() -> PTZState │
   │                           │    │  capabilities()          │
   └─────────────────────────┘    └─────────────────────────┘
              ↑                              ↑
   ┌──────────┼──────────┐         ┌──────────┼──────────┐
   ▼          ▼          ▼          ▼          ▼          ▼
FileFrameSrc RtspSrc  GigESrc   MockPTZ   PelcoD    OnvifPTZ
(.mp4)      (IP cam) (Basler)  (testowy) (RS485)   (IP)
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
[Layer 3: vendor-specific drivers / SDK / protocol implementacje]
```

### Layer 1: Core Tracker
- Już istnieje: `cpp/src/{inference,multi_target_tracker,target_manager,...}.cpp`
- Nie wie o hardware
- Wejście: `cv::Mat frame`, **wyjście**: decyzje tracking

### Layer 2: HAL interfaces
- `cpp/include/dtracker/io/frame_source.hpp` — `IFrameSource`
- `cpp/include/dtracker/io/ptz_controller.hpp` — `IPTZController`
- Czyste abstract C++17 interfaces (virtual methods, struct types)
- Stable API — vendor pisze raz, core się rozwija

### Layer 3: Per-hardware adapters
- **My dostarczamy referencyjne**:
  - `FileFrameSource` (cv::VideoCapture wrapper) — testy + dev
  - `MockPTZController` (header-only, in-memory state) — unit testy
- **Vendor / integrator pisze własne** dla konkretnego hardware:
  - `RtspFrameSource` — IP cameras
  - `CameraLinkFrameSource` — industrial CameraLink (FLIR, BIRD)
  - `GigEVisionFrameSource` — Basler/Pleora
  - `PelcoDPTZ` — RS485 PELCO-D protocol
  - `OnvifPTZ` — IP cameras ONVIF
  - `ViscaPTZ` — Sony VISCA over RS232/IP

## Kontrakt interfejsów

### IFrameSource
```cpp
class IFrameSource {
public:
    virtual bool open(const std::string& uri) = 0;
    virtual bool read(Frame& out) = 0;
    virtual const FrameSourceInfo& info() const = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;
};

struct Frame {
    cv::Mat image;           // BGR
    double  timestamp_s;
    long    frame_idx;
    bool    is_keyframe;
};

struct FrameSourceInfo {
    int width, height;
    double fps;
    long total_frames;       // -1 dla streams
    std::string codec;
    std::string uri;
};
```

URI scheme — vendor decyduje:
- `file:///path/to/video.mp4`
- `rtsp://user:pass@host:554/stream`
- `device:0` (kamera index)
- `cameralink://port=1&device=PleoraCam01` (custom)

### IPTZController
```cpp
class IPTZController {
public:
    virtual bool connect(const std::string& uri) = 0;
    virtual bool set_pan_tilt_velocity(double pan_dps, double tilt_dps) = 0;
    virtual bool absolute_pan_tilt(double pan_deg, double tilt_deg) = 0;
    virtual bool set_zoom(double zoom_x) = 0;
    virtual PTZState get_state() = 0;
    virtual const PTZCapabilities& capabilities() const = 0;
    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;
};

struct PTZState {
    double pan_deg, tilt_deg, zoom_x;
    double timestamp_s;
    bool moving;
    bool valid;
};

struct PTZCapabilities {
    bool supports_velocity_control;
    bool supports_absolute_position;
    bool supports_zoom;
    bool supports_telemetry;
    double pan_min_deg, pan_max_deg;
    double tilt_min_deg, tilt_max_deg;
    double max_pan_velocity_dps, max_tilt_velocity_dps;
    double zoom_min_x, zoom_max_x;
};
```

Jednostki: stopnie (pan/tilt), stopnie/s (velocity), krotność (zoom). Pan +90 = right, tilt +90 = up.

## Use case w aplikacji

```cpp
// Konfiguracja runtime — wybór adapterów per hardware
auto wide_cam = std::make_shared<RtspFrameSource>();
wide_cam->open("rtsp://192.168.1.50/wide");

auto narrow_cam = std::make_shared<CameraLinkFrameSource>();
narrow_cam->open("cameralink://port=1");

auto ptz = std::make_shared<PelcoDPTZ>();
ptz->connect("serial:///dev/ttyUSB0?baud=2400&address=1");

// Core tracker — accepts abstract interfaces
DualCameraPipeline pipeline(wide_cam, narrow_cam, ptz);
pipeline.run();
```

Identyczny kod aplikacji — tylko zmiana `make_shared<...>` na inny adapter dla nowego hardware.

## Refactor istniejącego pipeline

**Phase 1 (current)**: interfejsy + reference impls (FileFrameSource + MockPTZController) — DONE.

**Phase 2 (next)**: refaktor `cpp/app/main.cpp`:
- Zamiast `cv::VideoCapture cap(args.video)` — `auto cam = std::make_shared<FileFrameSource>(); cam->open(args.video)`
- Pętla używa `cam->read(frame)` zamiast `cap.read(frame)`
- Identyczne wyniki (FileFrameSource jest cienką otoczką na `cv::VideoCapture`)

**Phase 3 (gdy dual-camera scope)**: rozszerzyć `main.cpp` na 2 kamery:
- `--video-wide URI` + `--video-narrow URI`
- 2 osobne IFrameSource pointers
- Sync na timestamp (frame.timestamp_s)
- Wide → YOLO + tracking → emit PTZ command via IPTZController
- Narrow → drugi YOLO inference (lub re-use), bbox refinement
- Calibration wide↔narrow (intrinsic + extrinsic) jako osobny moduł

**Phase 4 (production)**: vendor pisze adapter dla swojego hardware.

## Korzyści

1. **Hardware-agnostic core** — tracker dziala na każdym sprzęcie który ma adapter
2. **Testable bez hardware** — FileFrameSource + MockPTZController umożliwiają full unit testing
3. **Vendor-friendly** — vendor pisze adapter raz, my się nie zmieniamy. SDK header (4 .hpp pliki) + przykładowy adapter to wszystko czego potrzebują
4. **Replace na żywo** — adapter wybierany w runtime configu, nawet plugin loading (`.so`/`.dll` dynamic load) możliwy
5. **Multi-vendor** — w jednym deployment można mieć różne kamery od różnych producentów

## Realny scope

| etap | praca | status |
|---|---|---|
| **Phase 1**: interfaces + reference impls + tests | 4h | ✅ DONE 2026-04-26 |
| Phase 2: refaktor `main.cpp` na IFrameSource | 1-2h | TODO |
| Phase 3: dual-camera mode (`--video-wide` + `--video-narrow`) | 1-2 dni | TODO |
| Phase 4 per hardware: vendor adapter | 3-10 dni each | per hardware |
| Phase 5: real-world integration test | 5-10 dni z hardware | post-demo |

**Pierwszy pełen dual-camera production setup z konkretnym hardware: 6-8 tygodni.**

## Tests

Phase 1 tests w `cpp/tests/test_io_hal.cpp` — 10 testów:

- `IO_FileFrameSource_OpenNonExistentFails`
- `IO_FileFrameSource_OpenRealVideo`
- `IO_FileFrameSource_ReadIncrementsFrameIdx`
- `IO_FileFrameSource_CloseIdempotent`
- `IO_MockPTZ_NotConnectedRejects`
- `IO_MockPTZ_ConnectThenCommand`
- `IO_MockPTZ_VelocityIntegratesPosition`
- `IO_MockPTZ_ZoomSetters`
- `IO_MockPTZ_DisconnectInvalidates`
- `IO_PolymorphicUsage_AppHoldsAbstractInterface`

Pełen suite po Phase 1: **57/57 pass** (Kalman 8 + MTT 9 + TM 6 + Lock 12 + Narrow 12 + IO 10).

## Next steps

1. ✅ Phase 1 commit + push (interfaces + reference impls + tests + ten dokument)
2. Phase 2: refaktor `main.cpp` (1-2h, low risk — FileFrameSource jest equivalent do `cv::VideoCapture`)
3. Po refaktorze: można pisać `--video-wide` + `--video-narrow` z dowolnymi adapterami
4. Vendor outreach (gdy zna się hardware): dostarczyć im 4 nagłówki + ten dokument
