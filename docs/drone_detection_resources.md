# Drone Detection — Resources Catalogue

Docelowy system: anti-drone / EO/IR surveillance, 500–1500 m, 1–5 dronów,
mix tła (niebo / krajobraz / urban). Typy: Mavic 4, FPV, Shahed-type,
płatowe UAV.

## Rozmiar drona w kadrze — matematyka

FOV 30°, rozdzielczość 1920×1080, kamera EO/IR:

| Dron | Rozp. | 500 m | 1000 m | 1500 m |
|---|---|---|---|---|
| Mavic 4 | 25 cm | ~6 px | ~3 px | ~2 px |
| FPV | 20 cm | ~5 px | ~2-3 px | ~1-2 px |
| Shahed-136 | 2.5 m | ~60 px | ~30 px | ~20 px |
| Typ. płatowy UAV | 1 m | ~25 px | ~12 px | ~8 px |

**Wniosek**: docelowe cele to 2-60 px. COCO-trained YOLO pracuje optymalnie
na obiektach > 30 px. **Małe drony < 10 px wymagają dedykowanego detektora
z P2 head i wysokim imgsz (1920+).**

---

## Publiczne datasety

### 1. Anti-UAV Challenge (CVPR workshop)
- **URL**: https://anti-uav.github.io/
- **Rozmiar**: ~160k klatek, ~100 sekwencji
- **Modalności**: RGB + IR synchronized
- **Rozdzielczość**: 1920×1080 (RGB), 640×512 (IR)
- **Drony**: mix quadcopters + fixed-wing
- **Odległości**: short, medium, long range
- **Dostęp**: **wymaga rejestracji** (formularz na stronie)
- **Licencja**: research-only
- **Najbardziej relevantne** dla naszego zastosowania

### 2. Drone-vs-Bird Challenge (IEEE AVSS)
- **URL**: https://wosdetc2023.wordpress.com/
- **Rozmiar**: ~11k klatek, ~70 videos
- **Modalności**: RGB
- **Main challenge**: odróżnianie drona od ptaka (krytyczne dla anti-drone)
- **Dostęp**: publiczny GitHub po zgłoszeniu challenge
- **Typ obiektów**: głównie quadcopters + ptaki

### 3. DUT Anti-UAV (Dalian University)
- **URL**: https://github.com/wangdongdut/DUT-Anti-UAV
- **Rozmiar**: 10h footage, ~580 sequences
- **Modalności**: RGB
- **Scene variation**: sky, tree, building, sea, multi-scale
- **Dostęp**: GitHub, prośba mailem
- **Licencja**: academic

### 4. Det-Fly
- **URL**: https://github.com/Jake-WU/Det-Fly
- **Rozmiar**: ~13k obrazków
- **Typ**: DJI drones w zróżnicowanych tłach
- **Dostęp**: direct download
- **Licencja**: MIT

### 5. USC Drone Dataset
- **URL**: https://github.com/CenekAlbl/drone-tracking-datasets
- **Typ**: single drone multiple cameras
- **Modalność**: RGB

### 6. VisDrone (mniej relevantne)
- **URL**: https://github.com/VisDrone/VisDrone-Dataset
- **Uwaga**: to widoki Z DRONA, nie NA DRONA — tylko pomocniczo

### 7. Roboflow Universe — Drone Detection
- **URL**: https://universe.roboflow.com/ (search "drone detection")
- **Setki datasetów** z preview + sample (100-1000 obrazków)
- **Wymaga**: darmowe konto, API key dla pełnego download
- **Zaleta**: szybka integracja z Ultralytics

---

## Pretrained Drone Detection Models

Zanim trenujemy od zera, warto spróbować pretrained.

### Hugging Face
- Search "yolo drone" na https://huggingface.co/models
- Przykładowe: `keremberke/yolov8n-drone-detection`, wiele community variants
- **Sposób użycia**: download weights → użyj z ultralytics YOLO API

### Roboflow Universe
- Wiele deployed modeli z API inference
- Często pretrained na ~1000-10000 samples

### Eagle-Vision (open-source anti-drone)
- **URL**: https://github.com/Eagle-Vision-AI (jeśli istnieje)
- Full anti-drone pipeline

---

## Papers + Techniques

### Small Object Detection techniques
- **YOLOv8 P2 Head**: dodatkowa warstwa detekcji dla mniejszych obiektów (stride 4 zamiast 8)
- **SAHI** (Slicing Aided Hyper Inference): tile inference na dużych rozdzielczościach
  - GitHub: https://github.com/obss/sahi
- **Anchor-free + higher imgsz (1920, 2048)**

### Key papers
- "A Survey of Drone Detection, Identification and Tracking" (2021)
- "Deep Learning for Drone Detection" (2022)
- "Small Drones Detection" challenges na IEEE AVSS

---

## Plan akcji

### Tydzień 1 — Discovery + Baseline
1. **Download sample** (Anti-UAV sample lub Drone-vs-Bird) do `data/drone_baseline/`
2. **Uruchom** `python tools/eval_yolo_baseline.py` — baseline obecnego yolov8s
3. **Udokumentuj**: ile drone'ów wykrywa, z jakim conf, czy klasyfikuje jako
   airplane/bird/kite

### Tydzień 2 — Pretrained test
1. **Pobierz** najlepiej oceniany drone model z Hugging Face
2. **Uruchom** eval na tym samym eval set
3. **Porównaj** z baseline yolov8s

### Tydzień 3+ — Fine-tune
Jeśli pretrained nie wystarczy, dotrenowanie na domain data.

---

## Decision checkpoints

- **Po tygodniu 1**: czy baseline yolov8s wystarcza na Mavic 4 @ 500 m (bbox ~6 px)?
  - **NIE** (prawdopodobne) → tydzień 2
- **Po tygodniu 2**: czy pretrained drone model wystarcza?
  - **NIE** → fine-tune (tydzień 3)
  - **TAK** → przejście do integracji w pipeline
