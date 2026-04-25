# Drone Tracker System

System śledzenia dronów z dwukanałową architekturą (wide + narrow) dla
zastosowań **anti-drone / surveillance z głowicy EO/IR**.

- **Wide channel** — globalna detekcja i wybór celu
- **Narrow channel** — precyzyjny lock + zoom na wybranym dronie
- **Recovery channel** — CSRT wizualny fallback gdy detekcja zawodzi
- **Walidowana telemetria** — mierzalne metryki correctness i end-state

Szczegółowa architektura i kontekst projektu: **[CLAUDE.md](CLAUDE.md)**
Plan 6-tygodniowy delivery: **[docs/drone_detection_resources.md](docs/drone_detection_resources.md)**

---

## Wymagania

- **Python 3.11+**
- **OS**: Windows / Linux / macOS
- **Zalecane**: GPU (NVIDIA CUDA) dla real-time performance.
  Pipeline działa też na CPU — ale 60 fps video będzie laggy podczas display.
- **RAM**: min. 8 GB (YOLOv8s waży ~40 MB, ale inference + OpenCV zużyje ~2-4 GB)
- **Dysk**: ~5 GB (Python + zależności + YOLOv8 weights pobrane automatycznie)
- **Opcjonalne**: webcam albo plik wideo do testów

---

## Instalacja

### 1. Sklonuj repo

```bash
git clone <repo-url>
cd drone-tracker-system
```

### 2. Utwórz virtual environment (zalecane)

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

To zainstaluje:
- `ultralytics==8.4.30` (YOLOv8 + ByteTrack)
- `opencv-contrib-python==4.13.0.92` (OpenCV **z** trackerami CSRT/KCF)
- `numpy==2.4.3`, `pyyaml==6.0.3`

**Ważne**: używamy **opencv-contrib-python** (nie plain `opencv-python`).
Standardowy opencv-python NIE ma CSRT/KCF trackerów, które są niezbędne dla
recovery channel. Jeśli masz już zainstalowany plain opencv-python:

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python==4.13.0.92
```

### 4. Zweryfikuj instalację

```bash
python -c "import cv2, ultralytics, numpy, yaml; print('OK')"
python -c "import cv2; print('CSRT:', hasattr(cv2, 'TrackerCSRT_create') or hasattr(cv2, 'legacy_TrackerCSRT_create'))"
```

Oczekiwany output:
```
OK
CSRT: True
```

### 5. YOLOv8 weights

Przy pierwszym uruchomieniu ultralytics **automatycznie pobierze** `yolov8s.pt`
(~22 MB) z serwera. Jeśli chcesz pobrać manualnie:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

Model zostanie zapisany w bieżącym katalogu.

---

## Szybki start — pierwszy run

### 1. Przygotuj plik wideo

Pipeline czyta plik `video.mp4` z katalogu głównego projektu.

```bash
# Skopiuj swój plik jako video.mp4
cp ~/moje_nagranie_drona.mp4 video.mp4
```

**Nie masz video?** Pobierz darmowe drone footage z:
- [Pexels](https://www.pexels.com/search/videos/drone/) — bez rejestracji
- [Pixabay](https://pixabay.com/videos/search/drone/) — bez rejestracji

Wybierz krótki film (10-30 s) z widocznym dronem.

### 2. Uruchom pipeline

```bash
# Windows (z venv aktywnym)
set PYTHONPATH=src
python src/main.py --config config/config.yaml

# Linux / macOS
PYTHONPATH=src python src/main.py --config config/config.yaml
```

### 3. Co zobaczysz

Otworzy się okno z dashboardem (3 panele):

- **Wide panel** (górny) — pełny kadr z ramkami wykrytych obiektów
- **Wide debug** (środkowy) — wszystkie tracki z etykietami `[1] ID X, [2] ID Y...`
- **Narrow panel** (dolny) — krop na wybranym dronie z informacjami:
  - `TARGET ID X` — który track jest śledzony
  - `PAN ERR`, `TILT ERR` — odchylenie centrum od celu
  - `ZOOM Nx` — poziom zoomu
  - `CENTER LOCK ON/OFF` — status locka

### 4. Zamknij pipeline

- **Q** lub **Esc** — wyjście
- Film kończy się sam na ostatniej klatce

---

## Controls operatora

| Klawisz | Akcja |
|---|---|
| **1-9** | Manual lock na N-tym widocznym tracku (z wide_debug panel) |
| **0** | Auto mode (zdjęcie manual lock) |
| **,** / **.** | Poprzedni / następny track w manual lock |
| **R** | Toggle recording (dashboard do MP4) |
| **T** | Toggle telemetry logging |
| **S** | Screenshot (dashboard + wide + narrow) |
| **Q** / **Esc** | Wyjście |

### Tryb manual lock (zalecany dla stabilnego śledzenia)

Gdy pipeline widzi wiele obiektów (dron + ptaki / odbicia / chmury),
operator może wymusić śledzenie konkretnego:

1. Spójrz na **wide debug panel** — każdy track ma żółtą etykietę `[N] ID X`
2. Znajdź dron → zapamiętaj jego numer `[N]`
3. Naciśnij klawisz z tym numerem (1-9)
4. Pipeline **zamraża** ten track jako cel, nie przełączy się na inny

Gdy dron zniknie / chcesz wybrać inny: naciśnij **0** (auto mode).

---

## Wyniki — artifacts

Każdy run generuje katalog w `artifacts/runs/<timestamp>/`:

```
artifacts/runs/2026-04-19_123456/
├── telemetry.jsonl           # per-frame telemetria (JSON Lines)
├── run_summary.json          # agregowane metryki end-state
├── images/                   # screenshoty kluczowych zdarzeń
│   ├── dashboard_*.png
│   ├── wide_*.png
│   └── narrow_*.png
└── video/
    └── tracker_analysis.mp4  # pełny dashboard recording (gdy record=on)
```

### Sprawdź wyniki

```bash
# Ostatni run
LATEST=$(ls -t artifacts/runs/ | head -1)
cat artifacts/runs/$LATEST/run_summary.json
```

Przykładowy output:
```json
{
  "session_duration_frames": 871,
  "final_narrow_owner_id": 1,
  "final_lock_phase": "LOCKED",
  "end_state_verdict": "LOCKED",
  "total_lock_loss_events": 2,
  "total_time_in_locked_frames": 718
}
```

**`end_state_verdict = "LOCKED"`** = tracker utrzymał cel do końca.
**`total_time_in_locked_frames / session_duration_frames`** = % czasu w locku.

---

## Konfiguracja — `config/config.yaml`

Najważniejsze parametry:

```yaml
yolo:
  model: yolov8s.pt          # model YOLO (yolov8n/s/m/l.pt)
  conf: 0.20                 # próg confidence (0.05 = liberalne, 0.30 = surowe)
  imgsz: 960                 # rozdzielczość inference (960 szybkie, 1280 dokładne)
  classes: [4, 14, 33]       # COCO: airplane, bird, kite (drony często tu trafiają)
  inference_every: 2         # co N-ta klatka (2 dla szybkiego video)

video:
  source: video.mp4          # ścieżka do pliku wideo
  record_on_start: true      # czy zapisywać dashboard do MP4

narrow_control:
  display_size_alpha: 0.50   # tempo dopasowania rozmiaru ramki
  display_center_alpha: 0.78 # tempo dopasowania pozycji ramki

tracker:
  max_missed_frames: 36      # po ilu klatkach track umiera
  max_center_distance: 220.0 # maksymalne przesunięcie w match (px)
```

**Tuning per-scenario**: różne typy materiału (close drone, far drone, formacja, scena z ptakami) wymagają różnych wartości. Patrz **[CLAUDE.md](CLAUDE.md)** — sekcja "Parametry config per-scenario".

---

## Troubleshooting

### `ImportError: No module named 'cv2'` lub brak CSRT

```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python==4.13.0.92
```

### Pipeline nic nie wykrywa na moim video

1. Sprawdź czy dron jest wystarczająco duży w kadrze (min ~15x15 px).
2. Uruchom baseline eval na kilku klatkach żeby zobaczyć co YOLO widzi:
   ```bash
   mkdir -p data/drone_baseline
   # skopiuj kilka klatek jako JPG do data/drone_baseline/
   PYTHONPATH=src python tools/eval_yolo_baseline.py --conf 0.05
   ```
3. Jeśli YOLO daje detekcje ale pipeline ignoruje — sprawdź `yolo.classes` w config.
   Drony są różnie klasyfikowane w COCO: `4=airplane, 14=bird, 33=kite`.
4. Dla małych dronów (<10 px) obecny YOLO nie wystarczy — potrzebny custom
   fine-tune, patrz **[training/README.md](training/README.md)**.

### Pipeline tnie / laggy

1. Zmniejsz `yolo.imgsz` (1280 → 960 → 640)
2. Zwiększ `yolo.inference_every` (1 → 2 → 4)
3. Wyłącz `record_on_start` — zapis MP4 też kosztuje
4. Ostatecznie: używaj GPU (NVIDIA CUDA). Ultralytics automatycznie wykryje.

### Manual lock nie trzyma drona

Jeśli pipeline widzi dron **i jego odbicie** (nad wodą) — obydwa z wysokim
conf, MTT matcher może pełznąć między nimi. Obecne rozwiązanie to manual_lock,
ale nie jest 100% niezawodne. **Custom YOLO** w planie 6-tygodniowym
rozwiązuje to fundamentalnie.

### MP4 dashboard recording jest uszkodzone (moov atom missing)

Pipeline musi zakończyć naturalnie (EOF video albo Q/Esc), nie hard-kill.
Jeśli ubijasz proces Ctrl+C lub `kill`, plik MP4 nie dostaje zamkniętego moov
atom i jest nieczytelny. Poprzez naturalne zakończenie to się nie zdarza.

---

## Struktura projektu

```
drone-tracker-system/
├── src/core/                 # główny kod pipeline
│   ├── app.py                # orchestrator + input parsing
│   ├── target_manager.py     # wide channel: wybór owner'a
│   ├── multi_target_tracker.py # agregacja detekcji w tracki
│   ├── narrow_tracker.py     # narrow channel: lock + zoom
│   ├── lock_pipeline.py      # FSM: ACQUIRE / LOCKED / HOLD / ...
│   ├── local_target_tracker.py # CSRT recovery fallback
│   ├── dashboard.py          # renderowanie paneli
│   └── telemetry.py          # logowanie metryk
├── src/main.py               # entry point
├── config/
│   ├── config.yaml           # główny config
│   └── bytetrack_*.yaml      # konfiguracja trackera ByteTrack
├── docs/
│   └── drone_detection_resources.md  # plan 6-tygodniowy
├── training/
│   └── README.md             # instrukcja fine-tune YOLO
├── artifacts/runs/           # wyniki runów (generowane)
├── video.mp4                 # plik wejściowy (Ty dostarczasz)
├── CLAUDE.md                 # dokumentacja architektury + workflow
├── README.md                 # ten plik
└── requirements.txt
```

---

## Dalsza dokumentacja

- **[CLAUDE.md](CLAUDE.md)** — pełna architektura, kontrakty, telemetria,
  empiryczne rezultaty baseline, znane ograniczenia
- **[docs/drone_detection_resources.md](docs/drone_detection_resources.md)** —
  publiczne datasety, pretrained modele, plan 6-tygodniowy do production
- **[training/README.md](training/README.md)** — GPU options (Colab / RunPod),
  training commands dla yolov8m z small-object augmentations

---

## Licencja

(uzupełnij wg potrzeb projektu)

---

## Kontakt

(uzupełnij wg potrzeb projektu)
