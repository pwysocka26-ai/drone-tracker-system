# Środowisko pracy — odtwarzanie na nowej maszynie

Cel: przeniesienie pełnego środowiska `drone-tracker-system` na nowy komputer
(zakładany: Windows 11 AMD, docelowo GMKtec EVO-X2). Po tych krokach musi
działać: Python pipeline, C++ port build + inference na ONNX/DirectML, CVAT
Docker, trening v4 lokalnie/Colab.

**Snapshot stanu: 2026-04-26.** Weryfikowane wersje są konkretne — nie używaj
"latest", bo cicho łamią się eksporty ONNX i CSRT.

**Co się zmieniło od 2026-04-24:**
- C++ port DONE (full parity z Python: TM identity anchor, Narrow PID+FF, Lock 5-state FSM, CSRT recovery, ROI search reacquire, R/T toggles, dual-camera HAL Phase 1+2+3) — 57/57 testów pass
- v4 yolov8s @ 640 FP16 wytrenowany na Colab Pro (Twoje 1898 CVAT klatek + v3 merge = 4032 obrazki) — **DEFAULT model** w `cpp/app/main.cpp`
- OpenCV rebuilt z contrib (CSRT), w `third_party/opencv_contrib_install/`
- Memory: ~25 plików (od 20)
- `cpp/` zacommitowany na branch (nie jest już untracked)

---

## 1. Prerekwizyty (zainstaluj zanim sklonujesz repo)

### 1.1 System operacyjny

- **Windows 11 x64** (22H2+). Linux/macOS nietestowane dla C++ części (DirectML
  only Windows).

### 1.2 Narzędzia do zainstalowania ręcznie

| Narzędzie | Wersja potwierdzona | Dlaczego | Jak zainstalować |
|---|---|---|---|
| **Visual Studio 2022** Community | 17.x, komponent MSVC v143 (Desktop development with C++) | Build C++ porta | https://visualstudio.microsoft.com/ — przy install zaznacz "Desktop development with C++" |
| **Git for Windows** | 2.53.0+ | Klonowanie repo + bash | https://git-scm.com/download/win — zostaw domyślne opcje |
| **Python** | **3.11.9** (nie 3.12, nie 3.13) | ultralytics==8.4.30 wymaga | https://www.python.org/downloads/release/python-3119/ — zaznacz "Add python.exe to PATH" |
| **Docker Desktop** | 28+ | CVAT labellowanie | https://www.docker.com/products/docker-desktop/ — po instalacji zaloguj się i uruchom |
| **NVIDIA/AMD driver** | latest | DirectML GPU inference | Windows Update albo sajt producenta |

**NIE instaluj**:
- opencv-python przez pip — zainstalujemy **opencv-contrib-python** (ma CSRT)
- ONNX Runtime standalone — mamy prebuilt w `third_party/`

### 1.3 Weryfikacja

W PowerShell/cmd:

```powershell
python --version        # musi być 3.11.9
git --version           # 2.50+
docker --version        # 28+, daemon musi działać
docker ps               # bez errora = daemon up
cl                      # uruchom w "Developer Command Prompt for VS 2022" — powinno odpowiedzieć "Microsoft (R) C/C++ Optimizing Compiler"
```

---

## 2. Klon repo + przełączenie brancha

```bash
cd C:\Users\pwyso
git clone https://github.com/pwysocka26-ai/drone-tracker-system.git
cd drone-tracker-system
git checkout feature/velocity-tracking
```

**UWAGA — brakujące rzeczy w repo:**

Te katalogi są w `.gitignore` i **nie przenoszą się przez git clone**:
- `cpp/` — cały port C++ (poprzednia sesja, ok. 1800 LOC)
- `third_party/` — OpenCV + ONNX Runtime + CMake (2.8 GB)
- `data/` — weights i datasety (3.1 GB)
- `artifacts/` — runy i test videos (533 MB)
- `tools/*` większość (poza 4 plikami whitelisted w `.gitignore`)

**Plan transferu tych katalogów**: sekcja **10**.

---

## 3. Python packages (globalnie, bez venv)

Obecnie projekt używa globalnego Pythona 3.11.9 bez `.venv/`. Zainstaluj zgodnie
z `requirements.txt` + dodatkowe:

```powershell
cd C:\Users\pwyso\drone-tracker-system

# podstawowe z requirements.txt
pip install -r requirements.txt

# dodatkowe (używane przez tools/, training/, eksport ONNX)
pip install torch==2.11.0 torchvision==0.26.0
pip install onnx==1.21.0 onnxruntime==1.25.0 onnxruntime-directml==1.24.4
pip install onnxslim==0.1.91 pillow==12.1.1 ultralytics-thop==2.0.18
```

Weryfikacja:

```powershell
python -c "import ultralytics, cv2, onnxruntime, torch; print(ultralytics.__version__, cv2.__version__, onnxruntime.__version__, torch.__version__)"
# expected: 8.4.30 4.13.0 1.25.0 2.11.0+...
```

**Gotcha — CSRT**: `cv2.TrackerCSRT_create()` działa tylko z `opencv-contrib-python`.
Jeśli masz zainstalowany `opencv-python`, CSRT cicho fail'uje. Odinstaluj:

```powershell
pip uninstall opencv-python opencv-python-headless
pip install --force-reinstall opencv-contrib-python==4.13.0.92
```

---

## 4. third_party/ — C++ dependencies

**Nie ma tego w repo** (2.8 GB). Są dwie opcje:

### Opcja A — skopiuj `third_party/` z obecnego komputera (zalecane)

To najszybsze. Przenieś katalog `C:\Users\pwyso\drone-tracker-system\third_party\`
całym blokiem (USB / rsync / OneDrive / zip). Zawartość:

```
third_party/
├── cmake-3.31.2-windows-x86_64/         # CMake (nie potrzeba systemowego)
├── cmake.zip                             # (backup, można usunąć)
├── onnxruntime-directml/                 # ORT 1.20.1 DirectML NuGet rozpakowany
├── onnxruntime-directml.nupkg            # (backup, można usunąć)
├── onnxruntime-win-x64-gpu-1.20.1/       # ORT CUDA variant (nieużywany na AMD, ale zostaje)
├── onnxruntime-win-x64-gpu-directml.zip  # (backup)
├── opencv/                                # OpenCV 4.10 Windows prebuilt rozpakowany
└── opencv-4.10-win.exe                    # (installer backup)
```

### Opcja B — pobierz od zera (gdyby Opcja A nie działała)

```powershell
cd C:\Users\pwyso\drone-tracker-system
mkdir third_party
cd third_party
```

**CMake** (prebuilt, żeby nie instalować systemowo):
```powershell
curl -L -o cmake.zip https://github.com/Kitware/CMake/releases/download/v3.31.2/cmake-3.31.2-windows-x86_64.zip
tar -xf cmake.zip
```

**ONNX Runtime 1.20.1 DirectML** (z NuGet):
```powershell
curl -L -o onnxruntime-directml.nupkg https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.20.1
mkdir onnxruntime-directml
tar -xf onnxruntime-directml.nupkg -C onnxruntime-directml
# po rozpakowaniu: third_party/onnxruntime-directml/runtimes/win-x64/native/onnxruntime.{dll,lib}
# third_party/onnxruntime-directml/build/native/include/onnxruntime_cxx_api.h
```

**OpenCV 4.10 Windows prebuilt**:
```powershell
curl -L -o opencv-4.10-win.exe https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe
# uruchom .exe — to jest self-extracting SFX, wskaź cel: third_party/  (utworzy opencv/)
.\opencv-4.10-win.exe
# po rozpakowaniu: third_party/opencv/build/x64/vc16/bin/opencv_world4100.dll
```

**BLOCKER — OpenCV bez contrib**: prebuilt **nie zawiera** `opencv_tracking` (CSRT).
Dla C++ `local_target_tracker.cpp` (recovery) potrzebujemy rebuild z contrib.
Obecnie `cpp/include/dtracker/local_tracker.hpp` to stub. Decyzja per Paulina:
po demo 30.04 albo przed (wymaga ~1-2h rebuild z source, patrz sekcja **11**).

Weryfikacja po Opcji A lub B:

```powershell
ls third_party\onnxruntime-directml\runtimes\win-x64\native\
# must contain: onnxruntime.dll, onnxruntime.lib

ls third_party\opencv\build\x64\vc16\bin\
# must contain: opencv_world4100.dll
```

---

## 5. Data artefacts — weights, videos, datasety

**Nie ma tego w repo** (3.6 GB łącznie).

### 5.1 Krytyczne (minimum do uruchomienia)

| Plik | Rozmiar | Skąd | Zastosowanie |
|---|---|---|---|
| **`data/weights/v4_best_fp16_imgsz640.onnx`** | **22 MB** | Drive `drone_tracker/training/runs/yolov8s_img640_v4/weights/` | **DEFAULT** v4 detector (87% LOCKED, 36 ms) |
| `data/weights/v4_best_fp16_imgsz960.onnx` | 22 MB | Drive `drone_tracker/training/runs/yolov8s_img960_v4/weights/` | v4 accuracy variant (84% LOCKED, 78 ms) |
| `data/weights/v3_best_fp16_imgsz960.onnx` | 53 MB | Drive `drone_tracker/runs/v3_drone_m_imgsz960/weights/` | v3 reference (do regression test) |
| `data/weights/v3_best.pt` | 52 MB | Drive `drone_tracker/runs/v3_drone_m_imgsz960/weights/best.pt` | v3 PyTorch source |
| `artifacts/test_videos/video_test_wide_short.mp4` | ~210 MB | Drive `drone_tracker/videos/` | **Default test video** (8800 klatek, 50 fps) |
| `artifacts/test_videos/video_test_wide.mp4` | ~2 GB | Drive `drone_tracker/videos/CSTN-*.mp4` | Pełny test (94894 klatek) |
| `data/test.mp4` | ~9 MB | Drive `drone_tracker/videos/test.mp4` | Drone-over-sea (KNOWN HARD) |
| `video.mp4` | ~10 MB | Drive `drone_tracker/videos/` | Drone Mavic nad łąką |
| `data/cvat_exports/cvat_v4_export.zip` | 175 MB | (źródło v4 dataset) | Re-build datasetu v4 jeśli trzeba |
| `training/v4_dataset.zip` | 1.26 GB | wygenerowane lokalnie | Upload na Drive dla re-train |

### 5.2 Opcjonalne (dla trenowania v4)

| Katalog | Rozmiar | Skąd |
|---|---|---|
| `data/ext_rgb_drone/` | ~800 MB | Google Drive — 3 klipy DJI (dji0002/0003/0005) |
| `data/ext_sea/` | ~200 MB | Google Drive — testowe videos morze |
| `data/roboflow_drone_v1/` | ~400 MB | Roboflow public dataset |
| `data/antiuav_probe/` | ~300 MB | Anti-UAV dataset eval |

### 5.3 Jak pobrać z Google Drive

Szczegóły w memory `reference_google_drive_assets.md`. Skrótowo:

- **Ręczny przez przeglądarkę** — wejdź na [drive.google.com](https://drive.google.com), folder `drone_tracker/`, download
- **Claude MCP Google Drive** — `/mcp` → `claude.ai Google Drive` → auth → narzędzia do fetch plików
- **rclone/gdown CLI** — dla dużych transferów (setup osobno)

### 5.4 Przeniesienie z obecnego komputera (jeśli masz fizyczny dostęp)

Najszybciej: przenieś całe `data/` i `artifacts/` przez USB albo wewnętrzną sieć.
**Uwaga**: te katalogi są `.gitignore`, nie wrzucisz ich na GitHub.

---

## 6. Memory files (stan współpracy z Claude Code)

Obecne memory: `C:\Users\pwyso\.claude\projects\C--Users-pwyso-drone-tracker-system\memory\`

Zawiera 20 plików — pełną wiedzę o użytkowniku, preferencjach, projekcie, bazie empirycznej, planie C++. Memory **nie jest w repo** (jest poza nim, w profilu usera).

### 6.1 Przeniesienie memory na nowy komputer

Założenie: na nowym komputerze też będziesz userem `pwyso` (lub równoważnym).

**Najprościej — zip całego katalogu `memory/`**:

```powershell
# na obecnym komputerze:
cd C:\Users\pwyso\.claude\projects\C--Users-pwyso-drone-tracker-system
Compress-Archive -Path memory -DestinationPath memory_backup.zip

# przenieś memory_backup.zip na nowy komputer (USB/OneDrive/email)

# na nowym komputerze:
cd C:\Users\pwyso\.claude\projects\
# path może nie istnieć, utwórz jeśli trzeba:
mkdir -p "C--Users-pwyso-drone-tracker-system"
cd "C--Users-pwyso-drone-tracker-system"
Expand-Archive -Path memory_backup.zip -DestinationPath .
```

**Jeśli user na nowym komputerze ma inną nazwę** (np. `pauli`): ścieżka Claude Code
się zmieni na `C--Users-pauli-...`. Skopiuj zawartość `memory/` do nowej lokalizacji
ręcznie. Linki wewnątrz memory są relatywne, więc zadziała.

**Weryfikacja po przeniesieniu**: uruchom Claude Code w katalogu `drone-tracker-system`
i spytaj "Co wiesz o tym projekcie?" — powinno wyciągnąć MEMORY.md.

---

## 7. Build C++ port

Po Sekcjach 2-5 masz repo + third_party + dane. Teraz build.

### 7.1 `cpp/` jest już w git (od 2026-04-25)

Branch `feature/velocity-tracking` ma pełny C++ port: ~5000 LOC, 57/57 testów pass,
HAL Phase 1+2+3, full parity z Python. Po `git checkout feature/velocity-tracking`
masz wszystko.

**`cpp/build/`** jest gitignored — wygenerowany lokalnie przez CMake build.

### 7.3 CMake configure + build

W **Developer Command Prompt for VS 2022** (nie zwykły cmd — potrzeba MSVC env):

```bat
cd C:\Users\pwyso\drone-tracker-system\cpp

set CMAKE=C:\Users\pwyso\drone-tracker-system\third_party\cmake-3.31.2-windows-x86_64\bin\cmake.exe

REM Auto-detect: jeśli istnieje opencv_contrib_install/, CMake użyje contrib (CSRT enabled)
REM Inaczej fallback na opencv/ prebuilt (CSRT disabled, local_tracker w stub mode)
%CMAKE% -S . -B build -G "Visual Studio 17 2022" -A x64

%CMAKE% --build build --config Release
```

Build produkuje `cpp/build/Release/`:
- `dtracker_main.exe` — pełen pipeline (default exe do uruchamiania)
- `dtracker_tests.exe` — test suite (57 testów, all pass)
- `poc_inference.exe` — proste benchmark
- `dtracker_lib.lib`
- DLL: `onnxruntime.dll`, `opencv_world4100.dll`, `opencv_videoio_ffmpeg4100_64.dll`, `DirectML.dll`

### 7.4 Smoke test — uruchomienie pełnego pipeline'u

```bat
cd C:\Users\pwyso\drone-tracker-system\cpp\build\Release

REM Default: v4 yolov8s @ 640 FP16 + video_test_wide_short.mp4
dtracker_main.exe --max-frames 30 --no-gui --no-record
```

**Oczekiwane output (v4 default)**:
```
Wide:   1920x1080 @ 50 fps, 8800 frames (codec=h264)
MODE:   single-camera (narrow = virtual crop wide)
Run: ../../../artifacts/runs\<ts>
Init detector (DirectML=yes)... OK
frame 30/8800  lock=LOCKED  owner=1  tracks=1  inf=37ms

=== DONE ===
Frames: 30 / 1.5s = 20 fps
Avg inference: 37 ms
```

Test suite:
```bat
dtracker_tests.exe
REM Oczekiwane: === 57 passed, 0 failed ===
```

Jeśli `DirectML=no` albo inference >100 ms — sprawdź `DirectML.dll` obok exe
i sterownik AMD/Intel/NVIDIA.

---

## 8. Python pipeline smoke test

```powershell
cd C:\Users\pwyso\drone-tracker-system

# szybki eksport ONNX z .pt (jeśli nie masz v3_best.onnx):
python tools\export_v3_to_onnx.py
# ⚠ wymaga: opset=12, simplify=False, imgsz=960 (hardcoded w skrypcie)

# pełny pipeline (~15 min, z recording):
$env:PYTHONPATH="src"
python src\main.py --config config\config.yaml
# powinno otworzyć 3 okna (wide/narrow/dashboard), zapisać artifacts/runs/<ts>/
```

Kontrolka że działa:

```powershell
$latest = (Get-ChildItem artifacts\runs\ | Sort-Object LastWriteTime -Desc | Select -First 1).FullName
cat "$latest\run_summary.json"
# powinno mieć session_duration_frames > 0, final_lock_phase itd.
```

---

## 9. CVAT Docker (labellowanie v4)

Szczegóły w `docs/cvat_setup.md`. Szybko:

```bash
cd C:\Users\pwyso
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d

# jednorazowo:
docker exec -it cvat_server bash -c "python ~/manage.py createsuperuser"

# web UI:
# http://localhost:8080
```

Import `artifacts/cvat_import.zip` jako nowy task — pre-labels z v3 się wczytają.

---

## 10. Lista rzeczy do fizycznego przeniesienia

**Auto pakowanie**: `powershell -File tools\_migrate_pack.ps1` → tworzy
`migration_bundle/` z minimum stackiem (~800 MB w 4 ZIP-ach). Dla pełnego transferu
(z trening data + opcv prebuilt fallback): `powershell -File tools\_migrate_pack.ps1 -Full`.

**Rozmiar łączny: ~5-9 GB** (snapshot 2026-04-26, real measurements). Plan transferu (USB / network):

| Źródło na obecnym kompie | Rozmiar | Metoda transferu |
|---|---|---|
| `drone-tracker-system/` (git clone) | ~80 MB | `git clone` + `git checkout feature/velocity-tracking` |
| `cpp/build/` | ~150 MB | NIE przenoś — odbuduj lokalnie przez CMake |
| `third_party/opencv_contrib_install/` | **101 MB** | zip + USB — gotowy build z CSRT, oszczędza 1-2h rebuild |
| `third_party/onnxruntime-directml/` | **43 MB** | zip + USB |
| `third_party/cmake-3.31.2-windows-x86_64/` | ~80 MB | zip + USB (jeśli nie chcesz instalować systemowo) |
| `third_party/opencv/` (prebuilt) | **887 MB** | tylko fallback gdy contrib_install zawiedzie — pomiń jeśli masz contrib |
| `data/weights/` | **312 MB** | zip + USB (v4 + v3 ONNX+PT) |
| `data/test.mp4` + `video.mp4` | ~20 MB | zip + USB (drone-over-sea + Mavic) |
| `data/cvat_exports/` | 175 MB | zip + USB (jeśli planujesz re-train v4) |
| `data/ext_rgb_drone/`, `data/ext_sea/`, `data/roboflow_drone_v1/` | ~1.4 GB | tylko jeśli re-train; inaczej zostaw |
| `artifacts/test_videos/` | **341 MB** | zip + USB (test + train videos) |
| `artifacts/runs/` | ~500 MB-2 GB | NIE PRZENOŚ — historyczne wyniki, regenerowalne |
| `training/v4/` (4032 obrazki) | 1.2 GB | tylko jeśli re-train; inaczej zostaw |
| `C:\Users\pwyso\.claude\projects\...\memory\` | ~150 KB (~25 plików) | zip + USB — KRYTYCZNE dla kontynuacji z Claude |

**Minimum absolutne** (~800 MB, żeby uruchomić tracker — automat przez `_migrate_pack.ps1`):
- repo (`git clone`)
- memory (`memory_backup.zip` ~150 KB)
- `data/weights/` (~312 MB, v4 + v3)
- `third_party/onnxruntime-directml/` (43 MB) + `third_party/opencv_contrib_install/` (101 MB)
- `artifacts/test_videos/` (~341 MB) — można obciąć do 1 video jeśli zależy

**Pełny stack do re-train + dev** (~5-9 GB): + opencv prebuilt + ext_* + cvat_exports + training/v4/.

Reszta (`ext_rgb_drone`, `roboflow`, `antiuav_probe`) — potrzebne tylko do re-train,
można pobrać z Google Drive na nowym komputerze według potrzeby.

---

## 11. Gotchas — rzeczy które gryzą po cichu

### 11.1 ONNX export opset

`tools/export_v3_to_onnx.py` ma **`opset=12, simplify=False`**. Użycie `opset=17`
albo `simplify=True` tworzy model, który ładuje się bez błędu, ale output tensor
ma inne rozmiary i decode YOLOv8 zwraca 0 detekcji. Nie zmieniaj tych parametrów.

### 11.2 imgsz=960 (nie 640)

v3 YOLOv8m trenowany na imgsz=960. Inference @ 640 daje drastyczny spadek recall
na małych dronach (test: `conf max = 0.003` vs 0.17 @ 960). W `cpp/app/poc_inference.cpp`
i `tools/export_v3_to_onnx.py` **imgsz=960 jest hardcoded**.

### 11.3 opencv-contrib-python vs opencv-python

Python `local_target_tracker.py` używa `cv2.legacy.TrackerCSRT_create()` — dostępne
tylko w **opencv-contrib-python** wariancie. Zwykły `opencv-python` cicho nie ma.
Zrzut z pip freeze weryfikuje: `opencv-contrib-python==4.13.0.92`.

### 11.4 OpenCV C++ contrib — RESOLVED 2026-04-25

OpenCV rebuilt z contrib (CSRT/KCF tracking) — `third_party/opencv_contrib_install/`
zawiera gotowy build. CMakeLists auto-detect: jeśli `opencv_contrib_install/` istnieje,
użyje go; inaczej fallback na `opencv/` prebuilt (z `DTRACKER_NO_OPENCV_CONTRIB` define
= `local_tracker` w stub mode).

**Jeśli musisz rebuild od zera** (nowy komputer bez `opencv_contrib_install/`):

```powershell
cd third_party
git clone https://github.com/opencv/opencv.git opencv_src
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_src && git checkout 4.10.0 && cd ..
cd opencv_contrib && git checkout 4.10.0 && cd ..

mkdir opencv_contrib_build && cd opencv_contrib_build
cmake -G "Visual Studio 17 2022" -A x64 `
  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules `
  -DBUILD_LIST=core,imgproc,imgcodecs,highgui,dnn,tracking,video,videoio `
  -DBUILD_opencv_world=ON `
  -DCMAKE_INSTALL_PREFIX=../opencv_contrib_install `
  ../opencv_src
cmake --build . --config Release --target INSTALL
# 1-2h kompilacji
```

Czas oszczędności: na nowym komputerze **przenieś `third_party/opencv_contrib_install/`**
(~1.2 GB) zamiast rebuilować.

### 11.5 Docker Desktop WSL backend

CVAT Docker compose wymaga WSL2 backend w Docker Desktop (nie Hyper-V). Na nowym
komputerze: Docker Desktop → Settings → General → "Use the WSL 2 based engine" ✓.

### 11.6 Ścieżki Windows w git bash

Git Bash używa `/c/Users/...`, PowerShell `C:\Users\...`. W skryptach trzymaj
Unix style dla `git` komend, Windows style dla `cmake`, `cv::VideoCapture`.

### 11.7 `.venv/` nie istnieje

Nie ma virtualenva — projekt używa globalnego Pythona 3.11.9. Jeśli wolisz venv,
utworzysz sam; `requirements.txt` jest gotowy do `pip install -r`.

### 11.8 GMKtec DirectML — AMD Radeon 8060S iGPU

DirectML wymaga aktualnego sterownika AMD. Jeśli `poc_inference.exe` przełącza
się na CPU mimo `DirectML: TAK` w output, sprawdź `dxdiag` czy GPU jest widoczne.
Aktualizuj sterownik z AMD Adrenalin.

---

## 12. Sanity checklist — "wszystko działa" przed kontynuacją pracy

Wykonaj po kolei na nowym komputerze. Wszystkie musza zwrócić ✓:

- [ ] `python --version` → `Python 3.11.9`
- [ ] `python -c "import ultralytics, cv2, onnxruntime; print('ok')"` → `ok`
- [ ] `git log --oneline -1` w repo → ostatni commit z `feature/velocity-tracking`
- [ ] `ls third_party/onnxruntime-directml/runtimes/win-x64/native/onnxruntime.dll` istnieje
- [ ] `ls third_party/opencv/build/x64/vc16/bin/opencv_world4100.dll` istnieje
- [ ] `ls data/weights/v4_best_fp16_imgsz640.onnx` istnieje (default model)
- [ ] `ls data/weights/v3_best_fp16_imgsz960.onnx` istnieje (regression test)
- [ ] `ls artifacts/test_videos/video_test_wide_short.mp4` istnieje
- [ ] `ls third_party/opencv_contrib_install/` istnieje (CSRT enabled)
- [ ] `ls cpp/CMakeLists.txt` istnieje
- [ ] `ls ~/.claude/projects/C--Users-*-drone-tracker-system/memory/MEMORY.md` istnieje
- [ ] `docker ps` → bez errora (daemon up)
- [ ] `cmake --build cpp/build --config Release` → succeeds, message "OpenCV tracking module header found (CSRT enabled)"
- [ ] `cpp/build/Release/dtracker_tests.exe` → `=== 57 passed, 0 failed ===`
- [ ] `cpp/build/Release/dtracker_main.exe --max-frames 30 --no-gui --no-record` → LOCKED, ~37 ms inference
- [ ] `PYTHONPATH=src python src/main.py --config config/config.yaml` → otwiera okna (Python pipeline jako reference)
- [ ] Claude Code → "Co wiesz o tym projekcie?" → wyciąga MEMORY.md (projekt anti-drone, deadline 2026-06-01, branch feature/velocity-tracking itd.)

---

## 13. Po przeniesieniu — plan kontynuacji (snapshot 2026-04-26)

C++ port + v4 model **DONE**. Zostaje:

### Track A — model fine-tune (jeśli realne nagrania z klienta)
1. Dolabelowanie real-world footage z głowicy (CVAT, jak v4)
2. v5 fine-tune: yolov8s @ 640 + dodatkowe domain data
3. Re-export ONNX FP16, podmień `data/weights/v4_*` w `cpp/app/main.cpp`

### Track B — production hardware migration
1. Jetson Orin cross-compile (ARM build, TensorRT export ONNX → engine)
2. Performance push do ≤19 ms/klatka (DirectML diagnoza, lazy CSRT już jest)
3. Vendor adapter Phase 4 (PelcoD/ONVIF/VISCA — gdy znamy konkretny PTZ + kamera)

### Track C — demo 30.04 prep (T-3 dni)
1. Multi-video script (przełączanie scenariuszy klawiszem)
2. README runbook dla operatora
3. Real-world walidacja na nowych nagraniach z klienta (jeśli są)

Memory current state — zobacz `MEMORY.md` w `~/.claude/projects/.../memory/`.
Najświeższe wpisy:
- `project_v4_training_results.md` (87% LOCKED, 4.4× speedup)
- `project_v4_real_world_validation.md` (sea+Mavic > v3, Anti-UAV gap expected)
