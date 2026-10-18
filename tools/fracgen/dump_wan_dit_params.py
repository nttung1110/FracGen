#!/usr/bin/env python
"""
Dump all parameter (and optionally buffer) keys + shapes + a small capped value preview
for Wan DiT / JointWanTransformer3DModel.

Why this works:
- `WanVideoPipeline` exposes the model as `pipe.dit`
- The actual class can be `WanModel` or `JointWanTransformer3DModel` depending on the checkpoint
- This script avoids building the full pipeline and instead uses `ModelPool.auto_load_model`,
  which auto-detects the correct model class from the checkpoint hash.

Example:
  python tools/dump_wan_dit_params.py \\
    --model-file /abs/path/to/diffusion_pytorch_model.safetensors \\
    --output /abs/path/to/dit_param_dump.txt \\
    --max-values 32 \\
    --include-buffers

Multi-shard safetensors:
  python tools/dump_wan_dit_params.py \\
    --model-file /abs/path/to/diffusion_pytorch_model-00001-of-00005.safetensors \\
    --model-file /abs/path/to/diffusion_pytorch_model-00002-of-00005.safetensors \\
    ... \\
    --output /abs/path/to/dit_param_dump.jsonl --format jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        action="append",
        required=True,
        help="Path to the DiT/JointGen checkpoint file (.safetensors/.pth). Repeat for multi-shard checkpoints.",
    )
    parser.add_argument("--output", required=True, help="Output path (.txt or .jsonl).")
    parser.add_argument("--format", default="txt", choices=["txt", "jsonl"], help="Output format.")
    parser.add_argument("--max-values", type=int, default=32, help="Max flattened values to record per tensor.")
    parser.add_argument("--include-buffers", action="store_true", help="Also dump buffers from named_buffers().")
    parser.add_argument("--name-regex", default=None, help="Only include tensors where name matches this regex.")
    parser.add_argument("--device", default="cpu", help="Load model onto this device (default: cpu).")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Model dtype.")
    parser.add_argument("--vram-limit", type=float, default=None, help="Optional VRAM limit passed to loader.")

    args = parser.parse_args()

    # Ensure repo is importable.
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # For consistency with examples: these env vars affect downloader behavior.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from diffsynth.models.model_loader import ModelPool
    from diffsynth.utils.param_dump import dump_wan_dit_params
    import torch

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    model_path: str | list[str]
    if len(args.model_file) == 1:
        model_path = args.model_file[0]
    else:
        model_path = args.model_file

    # Load only the models present in the given file(s).
    pool = ModelPool()
    vram_config = pool.default_vram_config()
    vram_config["computation_device"] = args.device
    vram_config["computation_dtype"] = torch_dtype
    # Keep everything on computation_device; no explicit offload by default.
    vram_config["offload_device"] = None
    vram_config["offload_dtype"] = None

    pool.auto_load_model(model_path, vram_config=vram_config, vram_limit=args.vram_limit)

    dit = pool.fetch_model("wan_video_dit")
    if dit is None:
        raise RuntimeError(
            "Could not find a 'wan_video_dit' model in the provided checkpoint file(s). "
            "Double-check that --model-file points to the DiT/JointGen weights."
        )

    dump_wan_dit_params(
        dit,
        output_path=args.output,
        include_buffers=args.include_buffers,
        max_values=args.max_values,
        name_regex=args.name_regex,
        format=args.format,
    )

    print(f"Wrote dump to: {args.output}")
    print(f"Model class: {dit.__class__.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


