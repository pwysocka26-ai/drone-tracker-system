# Labelling guide — D2-D3 (2026-04-26 do 2026-04-27)

Cheatsheet dla labellowania drone'ów w CVAT. Cel D2-D3: 1500-3000
zlabellowanych klatek `dron_maly` + ewentualne hard negatives.

## Klasy

W projekcie `drone_v4_csstn` są 4 klasy (curriculum, future-proof):

| label | kolor | kiedy używać |
|---|---|---|
| **`dron_maly`** | niebieski | **GŁÓWNY label** — drone w typowej skali anti-drone (5-50 px, dystans 500-1500 m). Pre-label v3 daje to domyślnie. |
| `dron_duzy` | pomarańczowy | drone zbliska (>50 px), gdy widać szczegóły kadłuba/śmigieł. Rzadko w obecnym video. |
| `tarcza` | czerwony | tarcza/cel statyczny. Brak w obecnym video, klasa rezerwowa. |
| `pilka` | zielony | piłka/balon. Brak w obecnym video, klasa rezerwowa. |

**99% pracy** = `dron_maly`. Pozostałe klasy gdy nagrasz nowe materiały.

## Co liczyć jako bbox — zasada

**Bbox = solidny korpus drona + śmigła (jeśli widać)**

- ✅ Cały dron z śmigłami: bbox obejmuje wszystko
- ✅ Korpus widoczny + zamazane śmigła: bbox wokół korpusu + zamazania
- ❌ Tylko sam korpus, śmigła wycięte: za ciasny bbox → trening ucieka do "samolot bez śmigieł"

**Margines**: 1-3 px wokół widocznego drona. **Nie więcej** — duży margines uczy model że drone jest większy niż jest naprawdę.

## Edge cases — co robić

### 1. Częściowa okluzja (chmura, drzewo)

| % drona widoczne | akcja |
|---|---|
| ≥50% widoczne | bbox tylko widocznej części |
| 25-50% | nadal bbox widocznej części, ale **nie martw się ostrością** |
| <25% | **POMIŃ** — usuń pre-label jeśli istnieje |

### 2. Dron na granicy kadru

- Jeśli >50% drona widoczne: bbox po widocznej części, krawędź = krawędź obrazu
- Jeśli <50% widoczne: pomiń (usuń pre-label)

### 3. Wiele dronów w kadrze

- Każdy dron osobny bbox
- Pre-label v3 prawdopodobnie złapał tylko jeden — dorysuj resztę
- **Nie łącz** dronów w jeden bbox nawet jeśli blisko siebie

### 4. Ruch (motion blur)

- Bbox wokół całej smugi, zatem szerszy niż zwykle
- Lepiej zlabellować szerszą smugę niż przegapić drona

### 5. Drone vs odbicie/cień

- **Tylko realny dron** dostaje bbox
- Odbicie w wodzie / cień na ziemi — **POMIŃ** (usuń pre-label jeśli istnieje)
- To jest ten sam problem co v3 miał na morzu — uczymy v4 że odbicie ≠ dron

### 6. Klatki bez drona (negative samples)

- **Zostaw bez bbox** — to są negative samples, model uczy się że tu nie ma drona
- **NIE OZNACZAĆ jako "skip"** — pusta klatka z brakiem labela = poprawnym labelem
- Pre-label v3 może dać fałszywy bbox w pustej klatce (chmura, ptak) → **usuń go**

### 7. Drone bardzo mały (<8 px)

- Jeśli widać 4-7 px kropkę i jesteś pewna że to drone (z kontekstu kilku klatek w okolicy):
  - Zlabelluj bbox wokół kropki (nawet 4×4)
  - Marker niski recall → trening v4 może podnieść
- Jeśli nie pewna (mogłoby być kropką szumu, ptaszkiem, etc.) — **pomiń**

## Workflow

### Tempo

- **Pierwsze 50-100 klatek**: 30-60 klatek/h (nauka skrótów + podejmowanie decyzji)
- **Po rozkręceniu**: 80-150 klatek/h
- **D2 cel**: 800-1500 klatek
- **D3 cel**: zamknąć do 2000+ klatek (combined z D2)

### Skróty (najważniejsze)

| klawisz | akcja |
|---|---|
| **D** / **F** | następna klatka |
| **A** / **S** | poprzednia |
| **W** | tryb edycji (move/resize bbox) |
| **N** | nowy bbox |
| **Delete** | usuń zaznaczony bbox |
| **Ctrl+Z** / **Ctrl+Shift+Z** | undo / redo |
| **Ctrl+S** | save (autosave też leci) |
| **F1** | full lista skrótów (CVAT pomoc) |
| spacja | play/pause (przy video tasks) |

### Backup co 200 klatek

Co 200 klatek zrób export jako safety:
- Menu **Actions** (3 kropki obok task #3) → **Export task dataset**
- Format: **YOLO 1.1**
- Zapisz ZIP w `~/cvat_backups/v4_progress_<data>.zip`

CVAT autosave działa, ale bezpieczniej mieć eksport gdyby coś się rozjechało.

### Skip-strategy gdy widzisz że jest źle

Jeśli pre-label masowo ma fałszywe bboxy (np. v3 zatrzymał się na chmurze):
1. **Zaznacz wszystkie bboxy klatki** (`Ctrl+A` w view)
2. **Delete**
3. Dodaj poprawne (jeśli istnieją)

Lepiej szybko odrzucić zły pre-label niż się nim irytować.

## Eksport po skończeniu — D4 morning

Gdy uznasz że masz dość labelek (1500-3000):

1. **Actions** (przy task #3) → **Export task dataset**
2. Format: **YOLO 1.1**
3. **Save images**: ✓ (włącz — chcemy obrazy + labels w jednym ZIP)
4. **Submit**

Wynik: ZIP z strukturą:
```
obj.data
obj.names
train.txt
obj_train_data/
  frame_NNNNNN.jpg
  frame_NNNNNN.txt   <- jeden line per bbox: "<class_id> <cx> <cy> <w> <h>"
```

Ten ZIP idzie do `training/v4/` na D4 (przed treningiem). Patrz
`training/train_v4.py` — wymaga `training/v4/data.yaml` z:
```yaml
path: training/v4
train: images/train
val: images/val
nc: 1
names: ['dron_maly']
```

(Pomimo że w CVAT są 4 klasy, dla v4 trening używamy tylko `dron_maly` bo
to jedyna klasa z danymi. Pozostałe klasy w CVAT są na przyszłość.)

## Self-check przed eksportem

Przed eksportem przejrzyj:

- [ ] **20 random klatek** — czy bboxy są ciasne wokół dronów?
- [ ] **Klatki bez drona** — czy nie ma zbędnych pre-labels?
- [ ] **Częściowe okluzje** — konsekwentnie zastosowane <25%/25-50%/≥50% rule?
- [ ] **Save** — `Ctrl+S` jeszcze raz przed eksportem
- [ ] **Liczba klatek** — minimum 1500, dobry cel 2500+

Jeśli któryś check'box "może lepiej zrobić więcej", wracaj — zła labelka wpływa
na cały trening v4, demo i delivery.

## Pytania — gdy utkniesz

Pytania które warto zadać Claude'owi:
- "Czy [opisz scenariusz] to dron czy nie?" — Claude pomoże zdecydować
- "Eksport YOLO 1.1 ma <jaką> strukturę?" — sprawdzimy
- "Trening v4 nie startuje, error <co>" — debug

Klucz: **konsekwencja** > perfekcja. Jeśli zdecydujesz "częściowe okluzje
≥40% labelluję", trzymaj się tego we wszystkich klatkach.
