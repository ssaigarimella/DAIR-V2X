#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone GT counter for DAIR-V2X (native format).

- Reads: <ROOT>/cooperative/data_info.json
- For each frame pair, opens cooperative label JSON (world frame) at "cooperative_label_path"
- Counts objects (optionally filtered by class names)
- Can filter frames by your split file (expects "cooperative_split": {"train":[...], "val":[...], "test":[...]})

Usage (example):
  python3 check_gt_counts.py \
    --root /workspace/datasets/dair_v2x_synth_TEST1/cooperative-vehicle-infrastructure \
    --split-file /workspace/DAIR-V2X/data/split_datas/synth-async-split.json \
    --split val \
    --classes car pedestrian

If --classes is omitted, all objects are counted.
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def parse_world_label(path: Path):
    """
    Returns a list of dicts with at least a 'type' field.
    Accepts a few common shapes:
      - [{"type": "...", ...}, ...]
      - {"labels": [ {...}, {...} ]}
      - anything with a top-level list is treated as the list
    """
    if not path.exists():
        return []
    J = load_json(path)
    if isinstance(J, list):
        recs = J
    elif isinstance(J, dict):
        if "labels" in J and isinstance(J["labels"], list):
            recs = J["labels"]
        else:
            # some tools store one object per file; normalize to list
            recs = [J]
    else:
        recs = []
    out = []
    for g in recs:
        # normalize class/type key
        t = g.get("type") or g.get("class") or g.get("category") or "unknown"
        out.append({"type": str(t).lower(), **g})
    return out

def resolve_split_ids(split_file: Path, split_name: str):
    if not split_file or not split_file.exists():
        return None
    S = load_json(split_file)
    # prefer cooperative_split if present
    if "cooperative_split" in S:
        return set(S["cooperative_split"][split_name])
    # fallback: plain split dict with "train"/"val"/"test" lists of vehicle ids
    if split_name in S:
        return set(S[split_name])
    raise KeyError(f"Unrecognized split file format: {split_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to cooperative-vehicle-infrastructure")
    ap.add_argument("--split-file", default="", help="Optional split json (with cooperative_split)")
    ap.add_argument("--split", default="val", choices=["train","val","test"], help="Split to use if split-file given")
    ap.add_argument("--classes", nargs="*", default=[], help="Class names to count (case-insensitive). If omitted, count all.")
    args = ap.parse_args()

    root = Path(args.root)
    coop_info = root / "cooperative" / "data_info.json"
    if not coop_info.exists():
        raise FileNotFoundError(f"Missing {coop_info}")

    frame_pairs = load_json(coop_info)
    split_ids = None
    if args.split_file:
        split_ids = resolve_split_ids(Path(args.split_file), args.split)

    wanted = set([c.lower() for c in args.classes]) if args.classes else None

    n_frames = 0
    total = 0
    per_frame_counts = []
    per_class_total = Counter()
    missing = 0
    empty = 0

    for fp in frame_pairs:
        # choose vehicle image stem as the “id” for split matching (this matches VIC code)
        veh_img = fp.get("vehicle_image_path","")
        stem = Path(veh_img).stem  # e.g., 000123
        if split_ids is not None and stem not in split_ids:
            continue

        lbl_rel = fp.get("cooperative_label_path","")
        lbl_path = root / lbl_rel
        if not lbl_rel or not lbl_path.exists():
            missing += 1
            per_frame_counts.append(0)
            n_frames += 1
            continue

        objs = parse_world_label(lbl_path)
        if wanted:
            objs = [o for o in objs if o["type"] in wanted]
        n = len(objs)
        if n == 0:
            empty += 1
        n_frames += 1
        total += n
        per_frame_counts.append(n)
        per_class_total.update([o["type"] for o in objs])

    head = per_frame_counts[:10]
    tail = per_frame_counts[-10:] if len(per_frame_counts) > 10 else []

    print("==== GT COUNT SUMMARY ====")
    print(f"Root: {root}")
    if split_ids is not None:
        print(f"Split: {args.split}  (frames in split: {len(split_ids)})")
    if wanted:
        print(f"Classes filtered: {sorted(wanted)}")
    print(f"Frames scanned: {n_frames}")
    print(f"Total GT objects: {total}")
    if per_class_total:
        print("By class:", dict(per_class_total))
    print(f"Frames with missing label file: {missing}")
    print(f"Frames with empty label list:  {empty}")
    if head:
        print(f"Per-frame counts (first up to 10): {head}")
    if tail and tail is not head:
        print(f"Per-frame counts (last up to 10):  {tail}")

if __name__ == "__main__":
    main()
