# Środowisko pracy — odtwarzanie na nowej maszynie

Cel: przeniesienie pełnego środowiska `drone-tracker-system` na nowy komputer
(zakładany: Windows 11 AMD, docelowo GMKtec EVO-X2). Po tych krokach musi
działać: Python pipeline, C++ port build + inference na ONNX/DirectML, CVAT
Docker, trening v4 lokalnie/Colab.

**Snapshot stanu: 2026-04-24.** Weryfikowane wersje są konkretne — nie używaj
"latest", bo cicho łamią się eksporty ONNX i CSRT.

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
| `data/weights/best.pt` | 52 MB | Google Drive `drone_tracker/runs/v3_drone_m_imgsz960/best.pt` | v3 YOLOv8m custom drone detector |
| `data/weights/v3_best.onnx` | 104 MB | generowany przez `python tools/export_v3_to_onnx.py` | Inference w C++ przez DirectML |
| `artifacts/test_videos/video_test_wide.mp4` | ~180 MB | Google Drive `drone_tracker/videos/CSTN-*.mp4` | Testy pipeline |
| `artifacts/cvat_import.zip` | 152 MB | wygenerowany przez `tools/prelabel_v3_for_cvat.py` | Import do CVAT dla labellowania v4 |

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

### 7.1 Jeśli przeniosłeś `cpp/` z Opcji A

Katalog już jest, przejdź do **7.3**.

### 7.2 Jeśli `cpp/` nie istnieje

`cpp/` jest obecnie nietknięty w git (untracked). Musi być skopiowane ręcznie
z obecnego komputera (z `C:\Users\pwyso\drone-tracker-system\cpp\`).

**Alternatywa** — zcommituj `cpp/` na branch `feature/velocity-tracking` **przed**
przesiadką (nie jest `.gitignore`, tylko untracked, więc `git add cpp/` zadziała):

```bash
# na obecnym komputerze przed przeprowadzką:
git add cpp/
git commit -m "Snapshot C++ port for laptop migration"
git push origin feature/velocity-tracking
```

Wtedy na nowym komputerze `git pull` wciągnie `cpp/`.

### 7.3 CMake configure + build

W **Developer Command Prompt for VS 2022** (nie zwykły cmd — potrzeba MSVC env):

```bat
cd C:\Users\pwyso\drone-tracker-system\cpp

set CMAKE=C:\Users\pwyso\drone-tracker-system\third_party\cmake-3.31.2-windows-x86_64\bin\cmake.exe

%CMAKE% -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DOpenCV_DIR="C:/Users/pwyso/drone-tracker-system/third_party/opencv/build"

%CMAKE% --build build --config Release
```

Build produkuje `cpp/build/Release/`:
- `poc_inference.exe`
- `dtracker_lib.lib`
- `onnxruntime.dll` (copy z third_party)
- `opencv_world4100.dll` (copy z third_party)

### 7.4 Smoke test — uruchomienie POC

```bat
cd C:\Users\pwyso\drone-tracker-system\cpp\build\Release

poc_inference.exe ^
  C:\Users\pwyso\drone-tracker-system\artifacts\cvat_import\obj_train_data\frame_000150.jpg ^
  C:\Users\pwyso\drone-tracker-system\data\weights\v3_best.onnx
```

**Oczekiwane output**:
```
Model:   ...v3_best.onnx
Image:   ...frame_000150.jpg
DirectML: TAK
Image size: 1920x1080
Init detector... OK
Warmup... OK

=== BENCHMARK ===
Avg inference time: ~140 ms (~7 fps)   # GMKtec Radeon 8060S

=== DETECTIONS (1) ===
  [0] cls=0 conf=0.076 bbox=(870,480,892,492) area=~260

Zapisano: poc_inference_out.jpg
```

Jeśli `DirectML: NIE` albo CPU-only — sprawdź czy `third_party/onnxruntime-directml/runtimes/win-x64/native/DirectML.dll` jest obok exe (powinien być skopiowany przez post-build).

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

**Rozmiar łączny: ~7 GB**. Plan transferu (USB / network):

| Źródło na obecnym kompie | Rozmiar | Metoda transferu |
|---|---|---|
| `drone-tracker-system/` (cały) | 78 MB (bez .gitignore) + 7 GB z .gitignore | git clone (dla wersji lean) + zip cała reszta |
| `cpp/` | ~5 MB src + 78 MB build | `git add cpp/ && git push` przed przesiadką **ALBO** zip ręcznie |
| `third_party/` | 2.8 GB | zip + USB / wypakować na docelu |
| `data/` | 3.1 GB | zip + USB **ALBO** Google Drive sync |
| `artifacts/` | 533 MB | zip + USB (zawiera test videos i cvat_import) |
| `C:\Users\pwyso\.claude\projects\...\memory\` | ~100 KB | zip + USB — krytyczne dla kontynuacji pracy z Claude |

**Minimum absolutne** (żeby coś odpalić):
- repo (`git clone`)
- memory (`memory_backup.zip`)
- `data/weights/best.pt` + `v3_best.onnx`
- `third_party/onnxruntime-directml/` + `third_party/opencv/`
- 1 test video

Reszta (ext_rgb_drone, roboflow, antiuav_probe) — potrzebne tylko do trenowania,
można zostawić i pobrać z Google Drive na nowym komputerze według potrzeby.

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

### 11.4 OpenCV C++ prebuilt bez contrib

C++ `cpp/src/local_target_tracker.cpp` — stub, bo OpenCV prebuilt nie ma modułu
`opencv_tracking`. Rozwiązanie (post-demo): rebuild OpenCV z contrib z source:

```powershell
cd third_party
git clone https://github.com/opencv/opencv.git opencv_src
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_src
git checkout 4.10.0
cd ..\opencv_contrib
git checkout 4.10.0
cd ..

mkdir opencv_contrib_build
cd opencv_contrib_build
cmake -G "Visual Studio 17 2022" -A x64 `
  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules `
  -DBUILD_LIST=core,imgproc,imgcodecs,highgui,dnn,tracking,video `
  -DBUILD_opencv_world=ON `
  -DCMAKE_INSTALL_PREFIX=../opencv_contrib_install `
  ../opencv_src
cmake --build . --config Release --target INSTALL
# 1-2h kompilacji
```

Potem w `cpp/CMakeLists.txt` zmień `OpenCV_DIR` na `third_party/opencv_contrib_install/`.

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
- [ ] `ls data/weights/best.pt` istnieje
- [ ] `ls data/weights/v3_best.onnx` istnieje
- [ ] `ls artifacts/test_videos/video_test_wide.mp4` istnieje
- [ ] `ls cpp/CMakeLists.txt` istnieje
- [ ] `ls ~/.claude/projects/C--Users-*-drone-tracker-system/memory/MEMORY.md` istnieje
- [ ] `docker ps` → bez errora (daemon up)
- [ ] `cmake --build cpp/build --config Release` → succeeds
- [ ] `cpp/build/Release/poc_inference.exe <jpg> <onnx>` → znajduje 1 detekcję, `DirectML: TAK`
- [ ] `PYTHONPATH=src python src/main.py --config config/config.yaml` → otwiera okna, klatki lecą
- [ ] Claude Code → "Co wiesz o tym projekcie?" → wyciąga MEMORY.md (projekt anti-drone, deadline 2026-06-01, branch feature/velocity-tracking itd.)

---

## 13. Po przeniesieniu — plan kontynuacji

Zgodnie z memory `project_cpp_port_plan.md` i `feedback_cpp_ambitious_plan.md`:

Priorytet dzisiaj (wieczór 2026-04-24 / D0):
1. `cpp/app/main.cpp` — end-to-end pipeline
2. Upgrade TargetManager do identity anchor + continuity guard
3. Upgrade NarrowTracker do PID + adaptive velocity feedforward
4. Upgrade LockPipeline do 5-state FSM z REFINE
5. `cpp/tests/` parity testy vs Python
6. OpenCV rebuild z contrib → LocalTargetTracker real (zamiast stub)

Plan 6-dniowy do demo 30.04 w memory `project_cpp_port_plan.md`. Po przesiadce
zaczynamy od 1.
