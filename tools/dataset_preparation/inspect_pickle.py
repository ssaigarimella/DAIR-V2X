import pickle
from pathlib import Path

base_dir = Path("/workspace/DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side")

splits = ["kitti_infos_train.pkl", "kitti_infos_val.pkl", "kitti_infos_test.pkl"]

for split in splits:
    pkl_path = base_dir / split
    if not pkl_path.exists():
        print(f"[WARN] {split} not found at {pkl_path}")
        continue

    try:
        with open(pkl_path, "rb") as f:
            info = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {split}: {e}")
        continue

    print(f"\n{split} — num samples: {len(info)}")
    if not info:
        print("  (empty list)")
        continue

    first = info[0]
    print("  Example keys:", list(first.keys()))

    # Smart sample display: prefer img_prefix or img_info if available
    sample_display = first.get("img_prefix") or first.get("img_info") or first
    print("  Sample:", sample_display)
