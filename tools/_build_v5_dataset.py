"""Build v5 dataset z v4 source (data identyczna; v5 zmienia model+imgsz, nie data).

Schemat (parity z v3/v4 flow):

    1. tools/_build_vN_dataset.py        -> tworzy training/vN/ + vN_dataset.zip
    2. drag&drop vN_dataset.zip          -> MyDrive/drone_tracker/
    3. Colab notebook (colab_vN_train)   -> trening + zapis na Drive
    4. download MyDrive/drone_tracker/vN -> data/weights/<run_name>/

v5 motywacja (memory: project_v4_dataset_size_distribution_2026_04_27):
- Dataset == v4 (4032 obrazki, 64.5% bbox <30 px)
- Zmienia sie model (yolov8s -> yolov8m) + imgsz (640 -> 1280)
- Wiec train/val identyczne, tylko inny base + augmentations + epochs

Layout po run:
    training/v5/
        images/{train,val}/
        labels/{train,val}/
        data.yaml                 (path: training/v5)
    training/v5_dataset.zip       (~1 GB, internal paths: training/v5/...)

Uzycie:
    python tools/_build_v5_dataset.py
    # albo z innym source:
    python tools/_build_v5_dataset.py --source training/v4
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


SOURCE_DIR_DEFAULT = Path("training/v4")
OUT_DIR = Path("training/v5")
OUT_ZIP = Path("training/v5_dataset.zip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=SOURCE_DIR_DEFAULT,
                   help=f"Source dataset dir (default {SOURCE_DIR_DEFAULT})")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help=f"Output dataset dir (default {OUT_DIR})")
    p.add_argument("--out-zip", type=Path, default=OUT_ZIP,
                   help=f"Output zip (default {OUT_ZIP})")
    p.add_argument("--no-zip", action="store_true",
                   help="Skip zip generation (zostaw tylko training/v5/)")
    p.add_argument("--force", action="store_true",
                   help="Nadpisz training/v5/ + zip jesli juz istnieja")
    return p.parse_args()


def validate_source(source: Path) -> tuple[int, int]:
    """Sanity check source dataset. Zwraca (n_train, n_val)."""
    if not source.exists():
        print(f"ERROR: source {source} nie istnieje", file=sys.stderr)
        print(f"  Najpierw zbuduj v4: python tools/_build_v4_dataset.py <cvat.zip> --include-v3",
              file=sys.stderr)
        sys.exit(1)

    required = [
        source / "images" / "train",
        source / "images" / "val",
        source / "labels" / "train",
        source / "labels" / "val",
        source / "data.yaml",
    ]
    for path in required:
        if not path.exists():
            print(f"ERROR: brak {path} w source", file=sys.stderr)
            sys.exit(1)

    img_exts = {".jpg", ".jpeg", ".png"}
    n_train = sum(1 for p in (source / "images" / "train").iterdir()
                  if p.suffix.lower() in img_exts)
    n_val = sum(1 for p in (source / "images" / "val").iterdir()
                if p.suffix.lower() in img_exts)

    if n_train == 0 or n_val == 0:
        print(f"ERROR: source ma train={n_train} val={n_val}, oczekuje >0", file=sys.stderr)
        sys.exit(1)

    return n_train, n_val


def copy_dataset(source: Path, out_dir: Path) -> tuple[int, int]:
    """Skopiuj source -> out_dir (rekurencyjnie). Zwraca (n_train, n_val)."""
    if out_dir.exists():
        print(f"[v5] usuwam istniejace {out_dir}...")
        shutil.rmtree(out_dir)

    print(f"[v5] kopiuje {source} -> {out_dir} (1.2 GB, ~30-60s na NTFS)...")
    shutil.copytree(source, out_dir)

    img_exts = {".jpg", ".jpeg", ".png"}
    n_train = sum(1 for p in (out_dir / "images" / "train").iterdir()
                  if p.suffix.lower() in img_exts)
    n_val = sum(1 for p in (out_dir / "images" / "val").iterdir()
                if p.suffix.lower() in img_exts)
    return n_train, n_val


def write_data_yaml(out_dir: Path) -> None:
    """Nadpisz data.yaml z poprawna sciezka path: training/v5."""
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['dron_maly']\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"[v5] data.yaml: path={out_dir.as_posix()}")


def build_zip(out_dir: Path, out_zip: Path) -> int:
    """Spakuj out_dir do zipa. Internal paths: training/v5/...

    Returns: zip size in bytes.
    """
    if out_zip.exists():
        print(f"[v5] usuwam istniejacy {out_zip}...")
        out_zip.unlink()

    print(f"[v5] pakuje {out_dir} -> {out_zip} (ETA 1-2 min)...")
    n_files = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in out_dir.rglob("*"):
            if path.is_file():
                # arcname: training/v5/images/train/foo.jpg
                # — zawsze "training/<out_dir.name>/..." niezaleznie gdzie fizycznie jest out_dir.
                # Notebook unpacker oczekuje {WORK_DIR}/training/v5/...
                rel_to_out = path.relative_to(out_dir).as_posix()
                arcname = f"training/{out_dir.name}/{rel_to_out}"
                z.write(path, arcname=arcname)
                n_files += 1
                if n_files % 500 == 0:
                    print(f"  ... {n_files} plikow")

    size = out_zip.stat().st_size
    print(f"[v5] zip: {n_files} plikow, {size/1e6:.1f} MB")
    return size


def main() -> int:
    args = parse_args()

    print(f"=== Build v5 dataset (source: {args.source}) ===\n")

    # 1. Validate source
    src_train, src_val = validate_source(args.source)
    print(f"[v4 source] train={src_train}  val={src_val}  total={src_train + src_val}")

    # 2. Refuse overwrite unless --force
    if (args.out_dir.exists() or args.out_zip.exists()) and not args.force:
        print(f"\nERROR: {args.out_dir} albo {args.out_zip} juz istnieje.", file=sys.stderr)
        print(f"  Uzyj --force zeby nadpisac, albo usun recznie.", file=sys.stderr)
        return 1

    # 3. Copy v4 -> v5
    out_train, out_val = copy_dataset(args.source, args.out_dir)
    if (out_train, out_val) != (src_train, src_val):
        print(f"WARN: copy mismatch — src=({src_train},{src_val}) out=({out_train},{out_val})",
              file=sys.stderr)

    # 4. Rewrite data.yaml (path: training/v5)
    write_data_yaml(args.out_dir)

    # 5. Zip (unless --no-zip)
    if args.no_zip:
        print(f"\n[v5] --no-zip, pomijam pakowanie")
    else:
        build_zip(args.out_dir, args.out_zip)

    # 6. Summary + next steps
    print(f"\n[v5 TOTAL] train={out_train}  val={out_val}  grand={out_train + out_val}")
    print(f"[v5]       layout: {args.out_dir}")
    if not args.no_zip:
        print(f"[v5]       zip:    {args.out_zip}")
    print(f"\n=== Next steps ===")
    print(f"  1. Upload {args.out_zip} -> MyDrive/drone_tracker/")
    print(f"  2. Open training/colab_v5_train.ipynb w Colab Pro")
    print(f"  3. Runtime -> Change runtime type -> GPU (A100/V100/T4)")
    print(f"  4. Run all cells")
    print(f"  5. Po treningu: download MyDrive/drone_tracker/v5/v5_drone_m_imgsz1280/")
    print(f"     -> data/weights/v5_drone_m_imgsz1280/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
