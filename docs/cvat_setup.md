# CVAT setup + workflow labellowania (Paulina)

**Cel**: uruchomić CVAT lokalnie w Dockerze, zaimportować pre-label z v3 YOLO, anotować dron_maly na `video_test_wide.mp4`.

Data utworzenia: 2026-04-24.

---

## 1. Pobranie i uruchomienie CVAT (raz, ~10 min)

Otwórz terminal (PowerShell lub Git Bash) **poza katalogiem drone-tracker-system**, np. w `C:\Users\pwyso\`.

```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
```

To pobierze ~2-3 GB obrazów i uruchomi serwis w tle. Pierwsze uruchomienie trwa ok. 5-10 min (zależy od łącza).

**Sprawdź działanie**:
```bash
docker compose ps
```
Powinieneś zobaczyć wszystkie serwisy `Up` (cvat_server, cvat_ui, cvat_db, itd.).

## 2. Utwórz superusera (raz)

```bash
docker exec -it cvat_server bash -c "python ~/manage.py createsuperuser"
```

Podaj:
- username: `paulina` (albo dowolny)
- email (opcjonalnie)
- password (dowolne, silne)

## 3. Zaloguj się

Otwórz w przeglądarce: http://localhost:8080

Zaloguj się danymi z kroku 2.

---

## 4. Import pre-label z v3

**W międzyczasie** ja (Claude) puszczam `tools/prelabel_v3_for_cvat.py` który generuje:
```
artifacts/cvat_import/
    obj.names
    obj.data
    train.txt
    obj_train_data/
        frame_000050.jpg
        frame_000050.txt
        frame_000100.jpg
        ...
```

Kiedy zakończy (~10-15 min na CPU), spakuj do zip:

```bash
cd artifacts
# Git Bash:
zip -r cvat_import.zip cvat_import/
# lub PowerShell:
Compress-Archive -Path cvat_import -DestinationPath cvat_import.zip
```

**Workflow w CVAT**:

1. **Projects** → **Create new project**
   - Name: `drone_v4_csstn`
   - Labels: dodaj klasę `dron_maly` (niebieski kolor)
   - Submit
2. **Tasks** → **Create new task** (w projekcie `drone_v4_csstn`)
   - Name: `video_test_wide_prelabel`
   - Files → **Upload**: `artifacts/cvat_import.zip`
     - CVAT automatycznie wykryje format YOLO 1.1 i zaimportuje obrazki + annotacje
   - Advanced configuration:
     - Use cache: **tak** (szybsze)
     - Image quality: 90+
   - Submit
3. Po utworzeniu task kliknij **Job #1** → wejdziesz w edytor

## 5. Workflow labellowania

**Kluczowe shortcuty CVAT**:

| klawisz | akcja |
|---|---|
| `F` / `G` | next / prev frame |
| `N` | new shape (rysuj bbox) |
| `Esc` | anuluj |
| `W` | tryb wybierania bbox |
| `Del` | usuń bbox |
| `Shift+R` | duplikuj bbox do następnej klatki |
| `Ctrl+Z` / `Ctrl+Shift+Z` | undo / redo |
| `Ctrl+S` | save |

**Szybki workflow dla każdej klatki**:

1. Sprawdź pre-label (zielone bboxy z v3)
2. Jeśli OK → `F` (następna klatka)
3. Jeśli bbox trochę obok → `W`, kliknij bbox, resize / drag
4. Jeśli brak drona a v3 narysowało → `Del`
5. Jeśli jest dron a v3 pominęło → `N`, narysuj bbox
6. **Ctrl+S** co ~50 klatek (CVAT ma autosave, ale bezpieczniej)

**Tempo**: celuj 30-60 klatek/godzinę początkowo, potem 60-120/h gdy złapiesz rytm.

**Target dla datasetu v4**: ~1500-3000 zanotowanych klatek z `video_test_wide.mp4`.

## 6. Export annotacji dla treningu

Gdy skończysz (albo nawet co 500 klatek dla backup):

1. **Actions** → **Export job annotations**
2. Format: **YOLO 1.1**
3. Zapisz ZIP
4. Rozpakuj do `training/v4/annotations/`

Stamtąd wchodzi do trenowania v4.

## 7. Backup / recovery

**Backup bazy**: `docker exec cvat_db pg_dumpall -U root > cvat_backup.sql`  
**Backup annotacji**: **Actions** → **Export job dataset** (cały z obrazkami i labels)

**Stop CVAT**: `docker compose stop` (zostaje stan)  
**Usuń zupełnie**: `docker compose down -v` (usuwa też bazę — UWAGA)

---

## Rozwiązywanie problemów

**CVAT nie startuje**: sprawdź `docker compose logs -f` — zwykle port 8080 zajęty przez inny proces albo brak pamięci.

**Upload się wiesza**: zip za duży. Zmniejsz jakość JPG w `prelabel_v3_for_cvat.py` (linia `IMWRITE_JPEG_QUALITY`, obecnie 92).

**Bboxy się "rozjeżdżają" po interpolacji**: dodaj keyframes co ~5-10 klatek (podczas labellowania ustaw klatkę jako keyframe przyciskiem gwiazdki na panelu bbox).
