"""Ingest every supported file in a folder (default: the 15 sample documents).

Usage:  python -m scripts.ingest_folder [folder]
"""
import sys
from pathlib import Path

from app.backend.container import get_ingestion
from app.config import ROOT_DIR
from app.ingestion.parsers import SUPPORTED_EXTENSIONS


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR / "data" / "documents"
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        sys.exit(f"No supported files in {folder}")
    pipeline = get_ingestion()
    total = 0
    for path in files:
        try:
            r = pipeline.ingest_file(path)
            total += r["chunks"]
            print(f"  ok   {path.name:45} {r['chunks']:5} chunks  {r['seconds']:7.1f}s")
        except Exception as exc:  # keep going; report the bad file
            print(f"  FAIL {path.name:45} {exc}")
    print(f"Done: {len(files)} files, {total} chunks in collection '{pipeline.store.collection}'.")


if __name__ == "__main__":
    main()
