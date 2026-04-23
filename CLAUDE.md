# Drone Tracker System

System śledzenia dronów z dwukanałową architekturą dla zastosowań
**anti-drone / surveillance z głowicy EO/IR**.

- **Docelowe scenariusze**: 1-5 dronów jednocześnie, odległość 500-1500 m,
  mix tła (niebo / krajobraz / urban)
- **Typy dronów**: Mavic-class quadcopters, FPV racing, Shahed-class loitering
  munitions, płatowe UAV
- **Deadline delivery**: 2026-06-01

---

## Architektura — dual-channel tracker z recovery

### Trzy warstwy pipeline

1. **Wide channel** (`target_manager.py`, `multi_target_tracker.py`)
   - Pełnoklatkowa detekcja YOLO + ByteTrack
   - `MultiTargetTracker` agreguje detekcje w stabilne `track_id` z własnym
     Kalmanem per track
   - `TargetManager` wybiera globalnego ownera (`selected_id`) z ograniczeniem
     churn: identity anchor, continuity guard w switch gate, manual_lock,
     selection_freeze

2. **Narrow channel** (`narrow_tracker.py`, `lock_pipeline.py`)
   - Committed lock na jednym ownerze
   - Stan: `lock_phase` = UNLOCKED / RECOVERING / WARMUP / LOCKED
   - PID `_step_towards` sterujący smooth_center z **adaptacyjnym** velocity
     feedforward (scale 0.2-1.0 zależnie od |velocity|)
   - `desired_center` = `owner_track.center_xy + ff_scale * velocity_xy`
     - |v| < 2 px/frame: scale=0.2 (drone stacjonarny, mały feedforward)
     - |v| >= 5 px/frame: scale=1.0 (szybki manewr, pełny feedforward)
     - pomiędzy: liniowa interpolacja
   - Stany: TRACKING / HOLD / REACQUIRE / SOFT_REACQUIRE

3. **Recovery channel** (`local_target_tracker.py`)
   - CSRT/KCF wizualny tracker fallback (wymaga `opencv-contrib-python`)
   - Aktywowany gdy wide+narrow stracą owner'a
   - Mostkuje krótkie luki detekcji

### Kontrakt między kanałami

- **Wide → narrow**: wide wskazuje ownera globalnego przez `requested_track`,
  narrow trzyma go lokalnie mimo krótkich dropoutów detektora
- **Narrow → wide (back-contract)**: narrow raportuje jakość lock_measurement
  do wide po każdej klatce dashboardu (`report_lock_measurement`) — pozwala
  narrow'owi fizycznie transitować do stanu LOCKED

### Lock pipeline FSM

`lock_pipeline.py` zarządza wysokopoziomowym stanem:
- `STATE_ACQUIRE` → wybór celu
- `STATE_REFINE` → dopinanie detekcji
- `STATE_LOCKED` → stabilne śledzenie
- `STATE_HOLD` → krótka luka detekcji
- `STATE_REACQUIRE` → odzyskiwanie utraconego celu

---

## Drone-specific tuning (`config/config.yaml` + `app.py`)

### YOLO

- `model: yolov8s.pt` — baseline COCO, **nie dotrenowany na drony** (plan wk 2)
- `classes: [4, 14, 33]` — airplane + bird + kite. Drony często trafiają do
  tych klas. Dla scen z prawdziwymi ptakami (nad morzem) lepiej zostawić `[4]`.
- `conf: 0.20` — kompromis: 0.05 daje propellery ale dużo noise (chmury/liście
  widziane jako bird/kite), 0.30 odrzuca propellery. 0.20 punkt balansu.
- `imgsz: 960` — CPU-friendly. Wyższe 1280+ dla GPU.
- `inference_every: 2` — co druga klatka (dla 60 fps video).
- `search_fallback: false` — wyłączone dla wydajności.

### Drone bbox padding (`app.py:parse_tracks`)

YOLO przy conf≥0.2 wykrywa solidny korpus drona ale wycina propellery (niski
conf). Padding przy konstrukcji Track:
```python
pad_w = (x2 - x1) * 0.15   # ~15% horizontal
pad_h = (y2 - y1) * 0.20   # ~20% vertical (propellery głównie po bokach)
```

Również area filter: `if area < 200.0: continue` (~14x14 px) — odrzuca noise.

### Tracker

- `max_center_distance: 220` — tolerancja matching. Mniejsze (60-80) zabija
  track przy szybkich ruchach, większe ryzykuje mismatch w gęstej scenie.
- `max_missed_frames: 36` — track umiera po 36 bez detekcji

### Narrow display

- `display_size_alpha: 0.50` — tempo dopasowania rozmiaru ramki do drona
  rosnącego/malejącego (było 0.82, zbyt wolne dla zbliżającego się drona)
- `display_max_size_step: 50` — max przyrost rozmiaru per klatka
- `display_center_alpha: 0.78` — tempo dopasowania pozycji ramki

---

## Warstwy telemetryczne

### Per-klatka (`artifacts/runs/<run>/telemetry.jsonl`)

Każda klatka generuje rekord JSON z ~60 polami. Kluczowe grupy:

1. **Tracking podstawowy**
   - `frame_idx`, `selected_id`, `active_track_id`, `active_raw_id`
   - `active_track_bbox`, `active_track_conf`, `active_track_vx/vy`
   - `multi_tracks`

2. **Correctness validation** (commit `4c475ad`)
   - `reference_established`, `reference_raw_id`, `reference_frame_idx`
   - `on_reference_target`, `off_reference_streak`, `wrong_neighbor_event`
   - `candidate_window_frames`, `area_ratio_to_reference`
   - Walidowana referencja: ustanawiana dopiero po N frames stabilnej,
     izolowanej tożsamości (nie z initial-acquire)

3. **End-state** (commit `5e282d6`)
   - `narrow_lock_phase`, `narrow_lock_state`, `narrow_hold_count`
   - `lock_loss_event`, `reacquire_start_event`, `reacquire_success_event`
   - `hold_frames_in_progress`, `reacquire_frames_in_progress`

4. **Neighbor analysis**
   - `nearest_neighbor`: `{track_id, raw_id, conf, distance_px, area_ratio_to_owner}`
   - `neighbor_count`, `owner_teleport_px`

### Per-run summary (`artifacts/runs/<run>/run_summary.json`)

Agregowane metryki end-state per run, nadpisywane co klatkę (survive kill):
```json
{
  "session_duration_frames": N,
  "final_narrow_owner_id": int | null,
  "final_lock_phase": "LOCKED" | "RECOVERING" | ...,
  "end_state_verdict": "LOCKED" | "REACQUIRE" | "NO_OWNER" | "HOLD",
  "total_lock_loss_events": N,
  "total_reacquire_starts": N,
  "total_reacquire_successes": N,
  "reacquire_success_rate": 0..1,
  "total_time_in_locked_frames": N,
  "total_time_in_recovering_frames": N,
  "total_time_in_hold_frames": N
}
```

### Screenshoty (`artifacts/runs/<run>/images/`)

Automatyczne zrzuty dashboard + narrow + wide przy kluczowych zdarzeniach:
`owner_switch`, `center_lock_off`, `reacquire`.

### Video recording (`artifacts/runs/<run>/video/tracker_analysis.mp4`)

Zapisywane gdy `record_on_start: true`. **Uwaga**: przy hard-kill MP4 traci
moov atom — pipeline musi zakończyć naturalnie (EOF albo Q/Esc).

---

## Controls operatora

- **1-9**: manual lock na N-tym widocznym tracku (z `visible_sorted`).
  Etykiety `[1] ID X, [2] ID Y...` pokazują się na wide_debug.
- **0**: auto mode (zdejmuje manual lock)
- **,** / **.**: poprzedni / następny track w manual lock
- **R**: toggle recording
- **T**: toggle telemetry
- **S**: screenshot
- **Q** / **Esc**: exit

Manual lock **zamraża** `selected_id` w target_manager, ale **nie chroni
fizycznie przed migracją** track_id pod wpływem MTT matchera. Scenariusz
z dronem + jego odbiciem w wodzie: YOLO widzi oba jako airplane z conf 0.9+,
`track_id=owner` może "pełznąć" na odbicie. Wymaga custom YOLO z odfiltrowaniem
odbić (plan wk 2-3).

---

## Git workflow

### Branch

- `main` — stable baseline
- `feature/velocity-tracking` — aktywne prace

### Conventional commit style

Opisuj **dlaczego** i mierzalny **impact**, nie tylko **co**:
```
Relax aspect ratio filter to 0.10..10.00 for acrobatic maneuvers

Close-up passes of aerobatic aircraft produce extreme bbox aspect
ratios (barrel rolls, vertical climbs). The previous 0.20..5.00
window discarded YOLO detections in those frames...

Full-run impact: end_state REACQUIRE/NO_OWNER -> LOCKED, locked
frames 54% -> 75%, lock_loss events 8 -> 1.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

### Przed każdym commitem

1. Pokaż `git diff --stat` przed zmianami
2. Smoke test (20s) albo pełny run
3. Porównaj `run_summary.json` z baseline
4. `py_compile` na zmienionych plikach
5. **NIE commituj bez explicit zgody użytkownika**

### Workflow zmian

1. **Diagnoza z telemetrii** przed tuningiem — każdy parametr zmieniany
   na podstawie konkretnego pomiaru, nie intuicji
2. **Smoke test** (20s, `record_on_start: false` tymczasowo) — sanity check
3. **Full run** (~15 min) — full comparison vs baseline
4. **Decyzja commit** — tylko jeśli mierzalna poprawa, no regresji

---

## Preferencje użytkownika

- **Język komunikacji**: polski. Kod, diffy, nazwy plików, commit messages,
  polecenia terminala — zostają w oryginale (angielski).
- **Format propozycji zmian**: Co znalazłem / Dlaczego to ważne / Co proponuję
  / Jak to sprawdzę.
- **Format decyzji**: polski wstęp + oznaczone opcje 1/2/3 + rekomendacja.
  Gdy CLI pokazuje listę Yes/No/Don't ask again — zawsze wyjaśnij co wybrać.
- **Werdykty**: `commitować` / `nie commitować` / `testować dalej`.
- **Nie commituj proaktywnie**. Zawsze pytaj o zgodę.
- **Nie strzelaj heurystykami bez mierzalnej hipotezy** — preferuj metodyczne
  pomiary nad intuicyjnym tuningiem.
- **Konfrontuj się z wizualnymi faktami** — gdy użytkownik mówi "pływa", nie
  zasłaniaj się telemetrią. Telemetria może mierzyć niewłaściwą rzecz.
  Generuj PNG z dashboardu / klatek i oglądaj **empirycznie**.

---

## Plan delivery do 2026-06-01

Szczegóły w `docs/drone_detection_resources.md` + `training/README.md`
+ `training/colab_setup.ipynb`:

| Tydzień | Daty | Faza | Deliverable |
|---|---|---|---|
| **1** | 20-26.04 | **Discovery** | Anti-UAV / Drone-vs-Bird / Det-Fly pobrane, baseline eval YOLOv8s, decyzja GPU (Colab/RunPod) |
| 2 | 27.04-3.05 | Fine-tune | yolov8m custom drone model, mAP@0.5 > 0.8 |
| 3 | 4-10.05 | Small objects | P2 head, imgsz=1920, augmentations dla dronów 2-30 px |
| 4 | 11-17.05 | Integration | Custom YOLO + pipeline end-to-end na realnych scenariuszach |
| 5 | 18-24.05 | User footage | Domain adaptation na realnym EO/IR gdy dostępny |
| 6 | 25.05-1.06 | Polish | Delivery ready |

### GPU options (z `training/README.md`)

| Opcja | Koszt | Trening yolov8m |
|---|---|---|
| **Colab Free** | $0 | T4, limit 12h sesji — wystarczy dla PoC |
| **Colab Pro** | $10/mies | T4 priority, ~6-8h dla 100 epok |
| **RunPod RTX 4090** | $0.34/h | ~5-6h → ~$2 total |
| **RunPod A100 80GB** | $1.89/h | ~2-3h → ~$4-6 total |
| **Własny RTX 3060+** | ~2000 PLN one-time | zależy od modelu |

### Colab setup — `training/colab_setup.ipynb`

Gotowy notebook z 11 krokami:
1. `nvidia-smi` weryfikacja GPU
2. Instalacja `ultralytics==8.4.30`
3. Mount Google Drive
4. Baseline eval na drone images
5. Pobranie Det-Fly dataset
6. Template konwersji annotacji (PASCAL VOC → YOLO format)
7. Basic training yolov8m
8. Advanced training z small-object augmentations
9. Eval mAP metrics
10. Inference na własnym video
11. Download weights i integracja z pipeline

---

## Znane ograniczenia (wymagają custom modelu, nie tuningu)

1. **YOLO COCO nie jest dotrenowany na drony** — klasa `airplane` to najbliższe
   co jest, daje niedokładne bboxy
2. **Odbicia w wodzie** — YOLO widzi drona i jego odbicie jako dwa airplane
   z conf 0.9+. Pipeline downstream nie odróżnia
3. **Małe drony (2-10 px na 1500m)** — obecny YOLO ich nie wykryje
4. **Performance CPU-only** — real-time wymaga GPU

---

## Empiryczne rezultaty baseline

### Video stress test (8 identycznych samolotów akrobatycznych, 560 klatek)

- LOCKED frames: **418 / 560 (75 %)**
- `lock_loss_events`: **1**
- `end_state_verdict`: **LOCKED**
- Poprawa vs baseline sprzed sesji: **+21 pp LOCKED, 8× mniej lock_loss,
  REACQUIRE → LOCKED**

### Video drone Mavic nad łąką (1054 klatek, 60 fps, pionowe)

- LOCKED frames: **718 / 871 (82 %)**
- `lock_loss_events`: **2**
- `end_state_verdict`: **LOCKED**
- **Pierwszy pełny run z realistycznym drone scenariuszem**

### Video drone nad morzem

Problemowy — YOLO widzi drona + odbicie jako dwa airplane. Wymaga custom
YOLO z odfiltrowaniem odbić lub dedykowanej klasy "water reflection".

---

## Custom YOLO milestone — v3 (2026-04-23)

Tydzień 2 planu delivery dostarczony. Wytrenowany dedykowany detektor
drone'ów `v3_drone_m_imgsz960` (yolov8m @ imgsz=960, 40 epok AdamW cos LR).

### Dataset v3 (2134 obrazków)

| split | dji (CSRT-labelled) | fp+pex (hard neg) | Roboflow public | razem |
|---|---|---|---|---|
| train | 326 | 225 | 1156 | **1707** |
| val | 82 | 56 | 289 | **427** |

- **dji**: 3 klipy DJI (`dji0002/0003/0005`) labellowane własnym narzędziem
  `tools/label.sh` (CSRT propagator z click-to-zoom), chronologiczny split
  80/20 (`tools/_split_train_val.py`) — brak leaków klatek sąsiadujących
- **fp+pex**: 281 hard negatives (fale, pianki, horyzont bez drona)
  z poprzedniej iteracji hard-neg mining
- **Roboflow**: 1445 publicznych obrazków drone-as-target
  ([drone-yolov5-b4787 by UAV Detection](https://universe.roboflow.com/uav-detection/drone-yolov5-b4787), CC BY 4.0) — zapewnia dywersyfikację kolorów drona
  (biały Phantom + czarny Mavic), oświetlenia (słoneczne + pochmurne), tła

Merge + split robi `tools/_build_v3_dataset.py`. Trening `training/train_v3.py`.

### Wyniki

- **val mAP@0.5 = 0.956** (v2 było 0.941)
- val mAP@0.5:0.95 = 0.693 (v2 było 0.728 — lekki spadek precyzji bboxów
  przez luźniejsze annotacje Roboflow vs ciasne CSRT dji)
- **test.mp4 inference**: biały DJI Phantom 3 na morzu, conf 0.77-0.85
  przez 428 klatek, ciasne bboxy, **zero FP na falach**. v2 na tym video
  nie łapał drona nawet przy conf=0.01 — domain shift naprawiony
  dywersyfikacją datasetu (nie wymagało fine-tune'a na test.mp4, co
  zachowuje niezależność hold-out na przyszłość).

### Wnioski

- Tydzień 2 (yolov8m custom, mAP>0.8) przekroczony 2× marginesem
- Known limit "odbicia w wodzie" częściowo zaadresowany przez hard negs
  `fp_*` i `pex*` — widać na klatkach test.mp4 z falami
- Small objects <10 px (known limit #3) wciąż nieadresowane — wymaga
  P2 head + imgsz=1280+ (plan tyg 3)

### Narzędzia workflow (tooling)

- `tools/label.sh <clip_id> [start_frame] [step] [end_frame]` — wrapper,
  resolves `data/ext_rgb_drone/DJI_*_<id>_V.MP4` automatycznie
- `tools/_label_track_init.py` — CSRT propagator z click-to-zoom
  two-stage ROI (klik na scaled preview → 1:1 crop 600x450 → ROI).
  ESC fallback dla dużych dronów (scaled ROI). Argumenty:
  `<video> <tag> <img_dir> <lbl_dir> <review_dir> [step] [start_frame] [end_frame]`
- `tools/_split_train_val.py` — stratyfikowany chronologicznie split
  v2 positives 80/20 (ostatnie klatki każdego klipu do val, unika leaku)
- `tools/_build_v3_dataset.py` — merge v2 + Roboflow do `training/v3/`
- `training/train_v2.py`, `training/train_v3.py` — training configs

### Artefakty na Google Drive

- `drone_tracker/datasets/v2_dataset.zip` (523 MB, 689 obrazków)
- `drone_tracker/datasets/v3_dataset.zip` (1.1 GB, 2134 obrazków)
- `drone_tracker/runs/v2_drone_m_imgsz960/` (best.pt + metrics)
- `drone_tracker/runs/v3_drone_m_imgsz960/` (best.pt + metrics)

---

## Quick references

### Uruchomienie pipeline

```bash
# Aktywuj virtual environment (zalecane)
source .venv/bin/activate        # Linux/macOS
# albo
.venv\Scripts\Activate.ps1       # Windows PowerShell

# Uruchom pipeline (z recording + telemetria)
PYTHONPATH=src python src/main.py --config config/config.yaml

# Szybki smoke test — tymczasowo zmień record_on_start: false w config
```

### Analiza wyników

```bash
LATEST=$(ls -t artifacts/runs/ | head -1)
cat artifacts/runs/$LATEST/run_summary.json
wc -l artifacts/runs/$LATEST/telemetry.jsonl
```

### Struktura artifacts

```
artifacts/runs/<timestamp>/
├── telemetry.jsonl           # per-frame telemetria
├── run_summary.json          # end-state summary (aktualizowany co klatkę)
├── images/                   # screenshoty kluczowych eventów
│   ├── dashboard_*.png
│   ├── wide_*.png
│   └── narrow_*.png
└── video/tracker_analysis.mp4  # full dashboard recording
```

### Tools diagnostyczne (lokalne, `.gitignore`)

- `tools/test_optical_flow.py` — Lucas-Kanade smoke test
- `tools/test_csrt.py` — CSRT smoke test na track_id
- `tools/test_csrt_hybrid.py` — CSRT + YOLO reinit hybrid (nieużyteczny)
- `tools/eval_yolo_baseline.py` — YOLO baseline na drone images

---

## Dependencies (`requirements.txt`)

- `ultralytics==8.4.30` — YOLOv8 detection + ByteTrack
- `opencv-contrib-python==4.13.0.92` — zawiera CSRT/KCF trackers (standardowe
  `opencv-python` NIE ma, cicho fail'uje w `local_target_tracker.py`)
- `numpy==2.4.3`
- `pyyaml==6.0.3`

Python 3.11+.

---

## Struktura repo

```
drone-tracker-system/
├── src/core/                      # główny kod pipeline
│   ├── app.py                     # orchestrator + parse_tracks (filter+padding)
│   ├── target_manager.py          # wide: wybór owner'a, identity anchor
│   ├── multi_target_tracker.py    # agregacja detekcji w tracki (Kalman)
│   ├── narrow_tracker.py          # narrow: lock + zoom + adaptive feedforward
│   ├── lock_pipeline.py           # FSM: ACQUIRE / LOCKED / HOLD / ...
│   ├── local_target_tracker.py    # CSRT recovery fallback
│   ├── dashboard.py               # renderowanie paneli
│   └── telemetry.py               # logowanie metryk + run_summary
├── src/main.py                    # entry point
├── config/
│   ├── config.yaml                # główny config (drone-tuned)
│   └── bytetrack_*.yaml           # ByteTrack config
├── docs/
│   └── drone_detection_resources.md  # katalog datasetów + pretrained
├── training/
│   ├── README.md                  # GPU setup + training commands
│   └── colab_setup.ipynb          # 11-step Colab notebook
├── artifacts/runs/                # wyniki runów (generowane)
├── data/drone_baseline/           # lokalne drone images do eval (ignored)
├── tools/                         # lokalne diagnostyki (ignored)
├── video.mp4                      # plik wejściowy
├── README.md                      # instrukcja dla użytkownika
├── CLAUDE.md                      # ten plik — architektura + kontekst
└── requirements.txt
```
