"""Sanity-check a dataset folder before training.

    python check_dataset.py "C:/Users/hp340/OneDrive/Desktop/archive/dataset"

Understands:  <root>/<class>/*.png   and   <root>/{training,testing}/<class>/*.png
Reports image counts, corrupt/tiny files, byte-identical duplicates and whether
the shipped training/testing folders share the same parts (data leakage).
"""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from common import dataset_roots, part_id

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if len(sys.argv) != 2:
    sys.exit(__doc__)

roots = dataset_roots(sys.argv[1])
print("Split folders found:", [r.name for r in roots], "\n")

classes = sorted({p.name for r in roots for p in r.iterdir() if p.is_dir()})
if len(classes) < 2:
    sys.exit(f"Found classes {classes}; need at least 2.")

hash_to_files = defaultdict(list)
parts_by_root = defaultdict(set)
totals = defaultdict(int)
bad_total = 0

for cls in classes:
    for r in roots:
        d = r / cls
        if not d.is_dir():
            continue
        n = bad = tiny = 0
        for f in d.rglob("*"):
            if f.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    tiny += min(im.size) < 64
            except Exception:
                bad += 1
                print(f"  corrupt: {f}")
                continue
            n += 1
            hash_to_files[hashlib.md5(f.read_bytes()).hexdigest()].append(f)
            parts_by_root[r.name].add(part_id(f, cls))
        totals[cls] += n
        bad_total += bad
        print(f"{r.name:<10} {cls:<14} {n:>6} images   corrupt={bad}  tiny={tiny}")

print("\nTotal per class:", dict(totals))
dupes = sum(len(v) - 1 for v in hash_to_files.values() if len(v) > 1)
print(f"Byte-identical duplicate files: {dupes}")

names = list(parts_by_root)
if len(names) >= 2:
    shared = set.intersection(*(parts_by_root[n] for n in names[:2]))
    print(f"Parts appearing in BOTH '{names[0]}' and '{names[1]}': {len(shared)} "
          f"of {len(parts_by_root[names[1]])}")
    if shared:
        print("  -> The shipped split LEAKS (same parts on both sides). "
              "train.py ignores it and re-splits by part.")

if min(totals.values()) < 100:
    print("\nSome class has < 100 images - collect more data.")
if max(totals.values()) > 3 * max(1, min(totals.values())):
    print("Classes are imbalanced (train.py compensates with class weights).")
print("\nDataset check finished.", "Corrupt files found!" if bad_total else "No corrupt files.")
