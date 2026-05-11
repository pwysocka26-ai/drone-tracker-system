"""Konwertuj YOLO labels z multi-class (0/1/2) na single-class (0).

Decyzja z 2026-05-10: anti-drone EO/IR nie wymaga rozróżnienia
maly/duzy/pilka — wszystkie etykiety to po prostu 'dron'.

Skrypt:
- Walks katalog labels/
- Każdy .txt: zmienia class_id na 0
- Zachowuje backup w labels_backup_multiclass/
- Idempotent: druga uruchomienie nic nie zmieni

Usage:
    python tools/convert_to_single_class.py <labels_dir> [<labels_dir>...]
"""
import shutil
import sys
from pathlib import Path


def convert(lbl_dir: Path):
    if not lbl_dir.exists():
        print(f"skip {lbl_dir} (missing)")
        return
    backup = lbl_dir.parent / f"{lbl_dir.name}_backup_multiclass"
    if not backup.exists():
        print(f"backup -> {backup}")
        shutil.copytree(lbl_dir, backup)

    n_files = 0
    n_lines = 0
    n_changed = 0
    for f in lbl_dir.iterdir():
        if not f.is_file() or f.suffix != ".txt":
            continue
        n_files += 1
        content = f.read_text(encoding="utf-8").strip()
        if not content:
            continue
        new_lines = []
        for line in content.split("\n"):
            parts = line.split()
            if len(parts) != 5:
                new_lines.append(line)
                continue
            n_lines += 1
            old = parts[0]
            if old != "0":
                n_changed += 1
                parts[0] = "0"
            new_lines.append(" ".join(parts))
        f.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"{lbl_dir}: files={n_files} lines={n_lines} changed={n_changed}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: convert_to_single_class.py <labels_dir> [<labels_dir>...]")
        sys.exit(2)
    for d in sys.argv[1:]:
        convert(Path(d))
