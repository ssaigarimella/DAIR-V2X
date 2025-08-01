import torch
from pathlib import Path

# <<< edit this to point to whichever .pth checkpoint you want to inspect
CHECKPOINT_PATH = Path(
    "/workspace/DAIR-V2X/configs/vic3d/late-fusion-image/imvoxelnet/"
    "vic3d_latefusion_inf_imvoxelnet_973cefc0b2c14fee1b8775aa996ac779.pth"
)

# Decide where to load the tensors. Using CPU is safest for inspection (no CUDA required).
# If you intend to resume training on GPU and CUDA is available, you could set this to cuda.
load_device = torch.device("cpu")  # keep on CPU for metadata inspection
# Example: to load directly to GPU if available, uncomment below
# if torch.cuda.is_available():
#     load_device = torch.device("cuda:0")

# Load the checkpoint, remapping tensors to the chosen device.
ckpt = torch.load(CHECKPOINT_PATH, map_location=load_device)

meta = ckpt.get("meta", {})

print(f"Checkpoint: {CHECKPOINT_PATH}")
print("Saved epoch:", meta.get("epoch"))
print("Saved iter:", meta.get("iter"))
print("Has optimizer state:", "optimizer" in ckpt)
# Some versions store scheduler info under different keys
has_scheduler = bool(meta.get("param_schedulers") or meta.get("lr_schedulers"))
print("Has lr scheduler info:", has_scheduler)

# Optional: inspect an example parameter tensor to see what device it lives on
state_dict = ckpt.get("state_dict") or ckpt.get("model")  # depending on how it was saved
if isinstance(state_dict, dict):
    name, tensor = next(iter(state_dict.items()))
    # tensor might be a Tensor or nested; guard accordingly
    if hasattr(tensor, "device"):
        print(f"Example param '{name}' device: {tensor.device}")
