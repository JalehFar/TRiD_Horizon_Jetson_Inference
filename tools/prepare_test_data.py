from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy or symlink final test videos into samples/ from an external dataset root.")
    parser.add_argument("--manifest", type=Path, default=Path("samples/test_videos.csv"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--symlink", action="store_true")
    args = parser.parse_args()
    with args.manifest.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        dst = Path(row["relative_input_path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        candidates = list(args.source_root.rglob(dst.name))
        if not candidates:
            print(f"MISSING {dst.name}")
            continue
        src = candidates[0]
        if dst.exists():
            continue
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        print(f"{'linked' if args.symlink else 'copied'} {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
