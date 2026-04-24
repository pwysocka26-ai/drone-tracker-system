# Dataset discovery v4 — publiczne źródła

**Data**: 2026-04-24
**Cel**: znaleźć publiczne materiały dla 4 klas detektora v4 (`tarcza`, `pilka`, `dron_duzy`, `dron_maly`) z licencją **commercial-friendly** (CC BY 4.0, MIT, Apache 2.0, Pexels, Unsplash, public domain).

**Wyłączone**: CC BY-NC (non-commercial), CC BY-NC-ND, GPL, restricted, OSINT war footage.

---

## TL;DR — status per klasa

| klasa | public datasets commercial-safe | status | plan |
|---|---|---|---|
| `tarcza` | **brak** (niszowe) | 🔴 | **syntetyczna generacja** 3D render + composite |
| `pilka` | balony (CC BY) + COCO sports ball | 🟡 | mix proxy + ewent. syntetyka |
| `dron_duzy` | Anti-UAV MIT (ograniczone Shahed), Roboflow scattered | 🟡 | mix + syntetyka dla Shahed-specific |
| `dron_maly` | v3 lokalnie + Anti-UAV (MIT) + Roboflow CC BY | 🟢 | mamy materiał |

---

## Klasa 1: `tarcza` (2×3 m z kołem, na morzu)

### Publiczne datasety

**Nie znaleziono** żadnego dataset-u z **tarczami morskimi 2×3 m** o permisywnej licencji. Istnieją pokrewne:

| dataset | obrazki | licencja | ocena |
|---|---|---|---|
| SeaBuoys (5751 buoy images, 6 klas) | 5751 | **niejasna** | 🔴 buoy ≠ tarcza; licencja do potwierdzenia u autorów |
| M²SODAI (multi-modal maritime, NeurIPS 2023) | ~6000/kat. | niejasna | 🔴 ships + floating, nie tarcze |
| MASS-LSVD (large-scale ship detection) | 64k | niejasna | 🔴 ships, nie tarcze |

### Wniosek

**Syntetyczna generacja jest konieczna** (user zaakceptowała). Pipeline:

1. **3D model tarczy** (Blender / Open3D) — prostokąt 2×3 m, koło w środku, tekstura wojskowa biało-czerwona
2. **Tła morskie** z Pexels / Unsplash (CC0 / Pexels License) — różne warunki: sztil / fale / słońce / chmury / zmierzch
3. **Compositing**: tarcza w różnych pozycjach (kąt kamery 0°-15° elewacji, dystanse symulujące 500-1500 m = 6-19 px w kadrze)
4. **Augmentacja**: motion blur, kompresja JPEG, noise EO/IR, refleksy na wodzie

**Minimalna objętość**: ~1500-3000 syntetycznych obrazków żeby model nauczył się rozpoznawać tarczę na różnych tłach.

---

## Klasa 2: `pilka` (90 cm średnicy, podwieszona pod dronem)

### Publiczne datasety

| dataset | obrazki | licencja | adekwatność |
|---|---|---|---|
| [Roboflow **Balloon Finder**](https://universe.roboflow.com/ftc-object-detection/balloon-finder) | 52 | **CC BY 4.0** ✓ | 🟡 małe, ale realne balony |
| [Roboflow **ballon-sqpif**](https://universe.roboflow.com/ballon-ocjlg/ballon-sqpif) | 1842 | CC BY 4.0 (typowo dla RF) | 🟢 dobra baza |
| [Roboflow **Balloon Dataset**](https://universe.roboflow.com/balloons-rtzsy/balloon-dataset-mp9i3) | 178 | CC BY 4.0 | 🟡 |
| [Roboflow **Balloon live**](https://universe.roboflow.com/balloon-d7vcz/balloon-live) | 99 | CC BY 4.0 | 🟡 |
| **COCO 2017** — klasa `sports ball` (id 37) | ~8000 instancji | **CC BY 4.0** ✓ | 🟢 najliczniejsza |

### Wniosek

**Balony są dobrym proxy dla piłki 90 cm** (podobny kształt kuli, rozmiar zbliżony). COCO sports ball dostarcza dodatkowej różnorodności pozycji/oświetleń. Balony w powietrzu są naturalnym setup-em (większość zdjęć balonów to balony na niebie = identyczne tło jak nasze scenariusze).

**Plan**:
- Balony Roboflow (wszystkie CC BY) → 2000-3000 obrazków
- COCO sports ball → ~3000 obrazków
- Syntetyka: piłka 3D compositing na zdjęcia dronów w locie (kluczowe dla "piłka **pod** dronem" scenario — to niszowe, nie ma publicznie)

---

## Klasa 3: `dron_duzy` (heavy-lift / Shahed-class, rozpiętość ~2.5 m)

### Publiczne datasety

| dataset | obrazki | licencja | adekwatność |
|---|---|---|---|
| [**Anti-UAV** (ZhaoJ9014)](https://github.com/ZhaoJ9014/Anti-UAV) | 300+410+600 video seq | **MIT** ✓ | 🟢 multi-scale UAV, niejasne czy Shahed |
| [**MMFW-UAV**](https://www.nature.com/articles/s41597-025-04482-2) | 147,417 fixed-wing | ❌ **CC BY-NC-ND** | 🔴 non-commercial, odpada |
| [**Kaggle Fixed Wing UAV**](https://www.kaggle.com/datasets/nyahmet/fixed-wing-uav-dataset) | ? | do sprawdzenia | 🟡 potencjalnie OK |
| [**Kaggle Drone Dataset (UAV)**](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav) | ? | do sprawdzenia | 🟡 |
| [Roboflow **class:shahed search**](https://universe.roboflow.com/search?q=class:shahed) | ? | per-dataset | 🟡 wymaga weryfikacji |

### Wniosek

**Trudna klasa** — większość datasetów fixed-wing UAV jest non-commercial (MMFW-UAV) albo niejasnej licencji. **Anti-UAV** jest jedynym dużym MIT-licensed ale nie wiemy dokładnie jakie typy UAV zawiera (dokumentacja mówi tylko "multi-scale" i "tiny-scale").

**Plan**:
1. Pobrać Anti-UAV (MIT) — sprawdzić ile z tego to heavy-lift / fixed-wing
2. Przeszukać Roboflow per-dataset (query `class:shahed`) — wybierać tylko te CC BY 4.0
3. **Syntetyka Shahed**: jeśli publicznie za mało, wyrenderować 3D modele Shahed-136 (dostępne na sketchfab / blenderkit, uwaga na licencje) + compositing na niebo

**Ryzyko**: klasa `dron_duzy` może mieć dużo mniej obrazków niż pozostałe → class imbalance w treningu → potrzebne class weighting albo oversampling.

---

## Klasa 4: `dron_maly` (Mavic-class, ~30 cm)

### Mamy już lokalnie

- **v3 dataset** `training/v3/` (2134 obrazków, 1707 train + 427 val) — dji + fp+pex + Roboflow drone-yolov5-b4787
- **CSTN fragmenty** `artifacts/mkv/CSTN-*.mkv` + `artifacts/test_videos/video_test_wide.mp4` — do zlabellowania w CVAT

### Dodatkowe publiczne

| dataset | obrazki | licencja | adekwatność |
|---|---|---|---|
| [**Anti-UAV**](https://github.com/ZhaoJ9014/Anti-UAV) | multi | **MIT** ✓ | 🟢 zawiera tiny-scale UAVs |
| [Roboflow **drone-yolov5-b4787**](https://universe.roboflow.com/uav-detection/drone-yolov5-b4787) | 1445 | CC BY 4.0 (już w v3) | 🟢 używany |
| [Roboflow **Drone Detection YOLOv7**](https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7) | 1300 | CC BY 4.0 | 🟢 |
| [Roboflow **Drones (6 modeli)**](https://universe.roboflow.com/colleage-7thf7/drone-dataset-pw8lv) | 1900 | CC BY 4.0 | 🟢 |
| [Det-Fly](https://github.com/Jake-WU/Det-Fly) | 13,271 | **niejasna** (academic citation only) | 🔴 do potwierdzenia |
| Drone-vs-Bird Challenge | ~700k video frames | ❌ **non-commercial** | 🔴 odpada |

### Wniosek

Mamy dużo. **Priorytet** w iteracji v4: dodaj **Anti-UAV (MIT)** do v3 bazy. Det-Fly odłóż do czasu potwierdzenia licencji.

---

## Podsumowanie i kolejne kroki

### Globalny status

- **Commercial-safe materiał**: **tak, wystarczający** dla klas 2 (balony) i 4 (drony małe)
- **Braki**: klasa 1 (tarcza) → **syntetyka**, klasa 3 (duży dron / Shahed) → **częściowo syntetyka**

### Do potwierdzenia (user / research)

1. **Licencja Det-Fly** — WebFetch [github.com/Jake-WU/Det-Fly](https://github.com/Jake-WU/Det-Fly) do repo licensing
2. **Licencja Kaggle fixed-wing UAV** — per-dataset sprawdzenie na kaggle.com
3. **Anti-UAV** — pobrać i obejrzeć klasy / typy dronów
4. **Roboflow `class:shahed`** — manually przeglądać per-dataset licencję

### Plan pobrania (po potwierdzeniu)

1. **Anti-UAV (MIT)** — jeden `git clone` + download weights zgodnie z README (repozytorium ZhaoJ9014)
2. **Roboflow balony (CC BY 4.0)** — każdy dataset przez Roboflow API / direct download z atrybucją w produkcie
3. **COCO 2017 sports_ball** — filtered subset (~3k obrazków), CC BY 4.0
4. **Syntetyka tarczy + syntetyka piłki pod dronem + syntetyka Shahed** — plan w `docs/synthetic_data_plan.md` (TBD)

### Atrybucja (wymagana przez CC BY 4.0)

Każdy dataset trzeba zaatrybuować w dokumentacji produktu. W repo: plik `docs/attributions.md` z listą źródeł i autorów. Budżet annotacyjny: ~2-3h na zebranie atrybucji zanim produkt pójdzie do klienta.
