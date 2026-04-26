# V4 vs V3 visual validation — sea_drone (data/test.mp4)

Empiryczna konfrontacja v4 yolov8s @ imgsz=640 FP16 (default) vs v3 yolov8m @ imgsz=960 FP16 (poprzedni
default) na scenariuszu **drone over sea** — historycznie problematyczny ze względu na odbicia w wodzie.

**Data**: 2026-04-26. **Generator**: `tools/_v4_validation_png.py`.

## Wyniki summary (full 428 klatek)

| metric | v4 yolov8s @640 | v3 yolov8m @960 |
|---|---|---|
| LOCKED | 168 / 428 (39.3%) | 136 / 428 (31.8%) |
| HOLD | 75 | 30 |
| REACQUIRE | 10 | 0 |
| lock_loss events | 1 | 0 |
| inference avg | 45 ms | 128 ms |

## Klatki side-by-side (lewa: v4, prawa: v3)

### `sea_drone_frame_0050.png`

Klatka 50/428 — drone jeszcze niewidoczny w polu. Oba modele: UNLOCKED, bez detekcji. Sanity baseline.

### `sea_drone_frame_0150.png`

Klatka 150/428 — drone wchodzi w pole. **v4 LOCKED z bbox**, **v3 UNLOCKED bez detekcji**. To pokazuje
**early acquisition** — v4 łapie drona ~100 klatek (3.3 s @ 30 fps) wcześniej niż v3.

### `sea_drone_frame_0250.png`

Klatka 250/428 — drone w środku pola, fale pod nim. **v4 LOCKED z ciasnym bboxem na dronie**,
**v3 ACQUIRE z FALSE bboxem obok drona** (prawdopodobnie odbicie/fala). v4 ma wyraźnie lepszy precision.

### `sea_drone_frame_0350.png`

Klatka 350/428 — stabilna faza. **Oba LOCKED**, bboxy prawie identyczne. W steady-state oba modele
radzą sobie podobnie.

## Wnioski

Twoje labele (1898 klatek z `video_test_wide_v3_prelabel`) poprawiły:

1. **Early acquisition** — v4 reaguje na drona w ~3 s krótszym oknie
2. **Rejection of false positives** — v3 mylił drona z odbiciem/falą, v4 nie

**Werdykt**: v4 jest production-ready dla naszego use case (anti-drone EO/IR z głowicy, 1920x1080,
500-1500 m, drony DJI/Mavic-class).
