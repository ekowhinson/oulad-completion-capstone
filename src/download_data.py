"""Download and extract the Open University Learning Analytics Dataset (OULAD).

Fetches the official archive from the UCI Machine Learning Repository (dataset
id 349, CC BY 4.0) and extracts the seven CSV tables into data/raw/. The raw
data is git-ignored; this script is the reproducible access path required by
the capstone rubric.

Usage:
    python src/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open+university+learning+analytics+dataset.zip"
)

EXPECTED_TABLES = [
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = ROOT / "data" / "oulad.zip"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[download] {url}")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n[ok] saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"[ok] extracted to {out_dir}")


def verify(out_dir: Path) -> bool:
    ok = True
    print("\nExtracted tables:")
    for name in EXPECTED_TABLES:
        path = out_dir / name
        if path.exists():
            print(f"  {name:26} {path.stat().st_size / 1e6:8.1f} MB")
        else:
            print(f"  {name:26} MISSING")
            ok = False
    return ok


def main() -> int:
    download(UCI_ZIP_URL, ZIP_PATH)
    extract(ZIP_PATH, RAW_DIR)
    if not verify(RAW_DIR):
        print("\n[error] some expected tables are missing; inspect the archive.")
        return 1
    print("\nDone. Raw OULAD tables are in data/raw/ (git-ignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
