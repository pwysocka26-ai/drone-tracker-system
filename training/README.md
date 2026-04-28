# Training YOLOv8 dla drone detection

## Schemat training vN (uniwersalny dla v3/v4/v5+)

Spojny, powtarzalny flow dla kazdej iteracji modelu. Ta sama konwencja sciezek
i komend, tylko inny `N`.

### Konwencja sciezek

| Element | Sciezka |
|---|---|
| Build script | `tools/_build_vN_dataset.py` |
| Local dataset dir | `training/vN/` (`images/{train,val}` + `labels/{train,val}` + `data.yaml`) |
| Local zip (gitignored) | `training/vN_dataset.zip` |
| Drive zip (upload target) | `MyDrive/drone_tracker/vN_dataset.zip` |
| Colab notebook | `training/colab_vN_train.ipynb` |
| Colab unpack dir | `/content/vN_work/training/vN/` |
| Colab run output | `/content/runs/<run_name>/` |
| Drive run output (po treningu) | `MyDrive/drone_tracker/vN/<run_name>/` (caly folder) |
| Local download dir | `data/weights/<run_name>/` |
| Pipeline runtime model | `data/weights/<run_name>/weights/<modelname>_fp16_imgszN.onnx` |

### Flow (zawsze ten sam)

```bash
# 1. LOKALNIE: build dataset + zip
python tools/_build_vN_dataset.py

# 2. UPLOAD: drag&drop training/vN_dataset.zip -> MyDrive/drone_tracker/

# 3. COLAB:
#    File -> Open notebook -> training/colab_vN_train.ipynb
#    Runtime -> Change runtime type -> GPU (A100/V100/T4)
#    Run all cells

# 4. DOWNLOAD: Drive -> MyDrive/drone_tracker/vN/<run_name>/
#    -> data/weights/<run_name>/

# 5. USE:
./cpp/build/Release/dtracker_main.exe \
    --model data/weights/<run_name>/weights/<modelname>_fp16_imgszN.onnx \
    --imgsz N
```

### Konkrety per wersja

| Wersja | Source | run_name | Base | imgsz | Epochs | Zip name |
|---|---|---|---|---|---|---|
| v3 | Roboflow + v2 | `v3_drone_m_imgsz960` | yolov8m | 960 | 40 | `v3_dataset.zip` |
| v4 | CVAT + v3 | `v4_drone_s_imgsz640`, `v4_drone_s_imgsz960` | yolov8s | 640, 960 | 50 | `v4_dataset.zip` |
| v5 | v4 (kopia) | `v5_drone_m_imgsz1280` | yolov8m | 1280 | 60 | `v5_dataset.zip` |

### Reguly

1. **Jeden zip per wersja**. Nie reuse'uj `vN_dataset.zip` dla `vN+1`. Nawet jak data identyczna —
   buduj nowy `vN+1_dataset.zip`. Powod: czytelnosc i debug — Drive sciezka jednoznacznie wskazuje wersje.

2. **Caly folder runu na Drive**, nie tylko ONNX. `best.pt` potrzebny do re-export
   (INT8, inny imgsz) bez powtarzania kilkugodzinnego treningu. `results.csv` + plots
   do post-mortem analizy konwergencji.

3. **`data.yaml` zawsze pisany przez build script**. Nie kopiowany — z poprawnym `path: training/vN`.

4. **Walidacja przed treningiem**: build script printuje `train=N val=M total=X`. Sprawdz
   ze liczby zgadzaja sie z oczekiwaniami (porownaj z poprzednia wersja).

---

## GPU infrastructure

Potrzebny GPU dla fine-tune. Opcje:

### Opcja 1: Google Colab Pro ($10/mies)
- **Zaleta**: najtańsze, brak setup, przeglądarka
- **Wada**: ograniczenie czasu (24h sesja), T4 GPU (wolniejszy)
- **Użycie**: kopiujesz notebook do Drive, wrzucasz dataset do Drive, trenujesz
- **Setup**:
  1. Zakup Colab Pro
  2. Upload dataset do Google Drive
  3. Otwórz `colab_train_yolo.ipynb` (stworzymy w tygodniu 1)
  4. Trening yolov8m: ~6-8h dla 100 epok, ~$0 (w ramach Pro)

### Opcja 2: RunPod (pay-per-hour)
- **Zaleta**: silne GPU (A100, RTX 4090), elastyczność
- **Wada**: wymaga rejestracji + karty, setup SSH
- **Cennik**:
  - RTX 4090: $0.34/h (Community), $0.69/h (Secure Cloud)
  - A100 40GB: $1.19/h
  - A100 80GB: $1.89/h
- **Użycie**: wynajmujesz pod na 20-40h, trenujesz, płacisz ~$10-40

### Opcja 3: Lokalny GPU
Jeśli masz RTX 3060+ (8GB+ VRAM), działa natywnie.

## Setup Anti-UAV Dataset

Zakładając że pobrałeś Anti-UAV:

```bash
# Struktura oczekiwana przez ultralytics
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/   # format YOLO: class x_center y_center width height (normalized)
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

`data.yaml`:
```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test
names:
  0: drone
```

Anti-UAV dostarcza annotacje w formacie custom JSON — potrzeba
konwersji na YOLO format. Skrypt konwersji: `training/convert_antiuav_to_yolo.py`
(do stworzenia w tygodniu 2).

## Training Command

### Bazowa komenda (yolov8m, 100 epok)

```bash
yolo train \
    model=yolov8m.pt \
    data=path/to/data.yaml \
    epochs=100 \
    imgsz=1920 \
    batch=8 \
    device=0 \
    name=drone_v1 \
    patience=20 \
    save=True \
    plots=True
```

### Small object detection tuning (drony 2-30 px)

```bash
yolo train \
    model=yolov8m.pt \
    data=path/to/data.yaml \
    epochs=150 \
    imgsz=1920 \
    batch=4 \
    device=0 \
    name=drone_small_v1 \
    patience=25 \
    # Augmentations dla małych obiektów:
    mosaic=1.0 \
    mixup=0.15 \
    copy_paste=0.3 \
    scale=0.5 \
    close_mosaic=10
```

### Custom architecture dla bardzo małych obiektów

Należy zmodyfikować `yolov8m.yaml` dodając P2 head:
```yaml
# ... (standard yolov8 architecture)
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 3, C2f, [128]]
  # P2 head (stride 4, dla obiektów < 16 px):
  - [-1, 1, Conv, [128, 3, 2]]
  # ... connect to Detect
```

Szablon dokończymy w tygodniu 3.

## Monitoring training

```bash
# W innym terminalu, podczas trainingu:
tensorboard --logdir runs/train/drone_v1
```

Metryki do śledzenia:
- **mAP@0.5** — main metric, cel >0.85
- **mAP@0.5-0.95** — strict, cel >0.55
- **precision / recall** — balance
- **box_loss / cls_loss** — convergence

## Inference benchmark

Po trainingu:

```bash
yolo predict \
    model=runs/train/drone_v1/weights/best.pt \
    source=path/to/test/images \
    conf=0.25 \
    save=True \
    imgsz=1920
```

## Integracja w pipeline

Gdy model gotowy:
1. Skopiuj `best.pt` jako `yolov8_drone.pt` do `src/core/` lub `models/`
2. Edit `config/config.yaml`:
   ```yaml
   yolo:
     model: yolov8_drone.pt
     classes: [0]  # 'drone' teraz class 0 w custom modelu
   ```
3. Uruchom pipeline jak zwykle, verify w telemetrii

## Timeline

| Tydzień | Task | Wynik |
|---|---|---|
| 1 | Download datasets + baseline eval | Raport: jak YOLOv8s wypada na dronach |
| 2 | Pretrained drone model eval + convert Anti-UAV labels | Ground truth dla fine-tune |
| 3 | Fine-tune yolov8m (first run) | Custom model v1, mAP@0.5 metric |
| 4 | Integration + test na oryginal video | Pipeline z custom YOLO |
| 5 | User footage integration (gdy dostępne) | Domain adaptation |
| 6 | Polish + delivery | Production-ready |
