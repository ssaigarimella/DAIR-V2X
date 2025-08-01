import json
import argparse
from pathlib import Path
from math import gcd
from functools import reduce

# <-- edit this if the split JSON lives somewhere else
JSON_PATH = Path("/workspace/DAIR-V2X/data/split_datas/cooperative-split-data.json")

def compute_simplified_ratio(counts):
    g = reduce(gcd, counts)
    if g == 0:
        return [0] * len(counts)
    return [c // g for c in counts]

def analyze_split(d, name):
    # expects d to have 'train','val','test' keys each mapping to a list
    parts = ["train", "val", "test"]
    counts = [len(d.get(k, [])) for k in parts]
    total = sum(counts)
    if total == 0:
        print(f"Section '{name}': all splits empty")
        return
    simplified = compute_simplified_ratio(counts)
    percents = [c / total * 100 for c in counts]

    print(f"\nSection '{name}':")
    print(f"  counts: train={counts[0]}, val={counts[1]}, test={counts[2]}, total={total}")
    print(f"  simplified train:val:test = {simplified[0]}:{simplified[1]}:{simplified[2]}")
    print(f"  percentages: train={percents[0]:.2f}%, val={percents[1]:.2f}%, test={percents[2]:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Report train:val:test counts, ratios, and percentages from a split JSON."
    )
    parser.add_argument(
        "--section",
        help="Top-level section to inspect (e.g., 'batch_split' or 'vehicle_split'); if omitted, all sections with train/val/test are processed",
        default=None,
    )
    args = parser.parse_args()

    if not JSON_PATH.is_file():
        raise SystemExit(f"Error: JSON file not found at {JSON_PATH}")

    data = json.loads(JSON_PATH.read_text())

    targets = [args.section] if args.section else list(data.keys())
    found = False
    for section in targets:
        if section not in data:
            continue
        val = data[section]
        if not isinstance(val, dict):
            continue
        if all(k in val for k in ("train", "val", "test")):
            analyze_split(val, section)
            found = True

    if not found:
        print("No section with train/val/test splits found in the provided JSON.")

if __name__ == "__main__":
    main()
