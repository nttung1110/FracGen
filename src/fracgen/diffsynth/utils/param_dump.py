
""""
Usage example in the debugiing:


from diffsynth.utils.param_dump import dump_wan_dit_params


step_id = "aft_load"
# dit is pipe.dit (JointWanTransformer3DModel or WanModel)
out_path = dump_wan_dit_params(
    self.pipe.dit,
    tag=f"step_{step_id}",
    max_values=8,
    include_buffers=False,
    # name_regex=r"blocks\.0\.",  # optional filter
)
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional, TextIO, Tuple


def dump_wan_dit_params(
    model,
    output_path: str | Path | None = None,
    output_dir: str | Path = "outputs/param_dumps",
    tag: str = "dit",
    include_buffers: bool = False,
    max_values: int = 32,
    name_regex: Optional[str] = None,
    format: str = "py",
) -> Path:
    """
    Dump Wan DiT / JointWanTransformer3DModel parameters (and optionally buffers) to a file.

    Intended for **in-process debugging** (e.g., call this from training code).

    Args:
        model: `pipe.dit` or any torch module (e.g. JointWanTransformer3DModel).
        output_path: If provided, write exactly here. Otherwise auto-generate a path.
        output_dir: Directory used when `output_path` is None.
        tag: Added to the auto-generated filename.
        include_buffers: Also dump `named_buffers()`.
        max_values: Max flattened values to record per tensor.
        name_regex: Only dump tensors whose names match this regex.
        format: "txt" or "jsonl".

    Returns:
        Path to the written file.
    """
    if output_path is None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%m%d_%H%M%S")   
        suffix = "py" if format.lower().strip() == "py" else "txt"
        output_path = out_dir / f"wan_{tag}_params_{ts}.{suffix}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    dump_module_tensors(
        model,
        output_path,
        include_buffers=include_buffers,
        max_values=max_values,
        name_regex=name_regex,
        format=format,
    )
    print(f"Wan DiT params dumped to: {output_path}")
    return Path(output_path)



@dataclass(frozen=True)
class TensorPreview:
    name: str
    kind: str  # "param" | "buffer"
    shape: Tuple[int, ...]
    dtype: str
    device: str
    requires_grad: Optional[bool]
    numel: int
    preview_flat: list
    stats: Optional[dict]


def _safe_stats(t):
    # Keep this cheap: compute stats only for floating/complex tensors, and only on CPU.
    try:
        import torch
    except Exception:
        return None
    if not isinstance(t, torch.Tensor):
        return None
    if t.numel() == 0:
        return None
    if not (torch.is_floating_point(t) or torch.is_complex(t)):
        return None
    tt = t.detach()
    if tt.is_cuda:
        tt = tt.cpu()
    tt = tt.float()
    # Avoid NaNs exploding summary.
    tt = torch.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "min": float(tt.min().item()),
        "max": float(tt.max().item()),
        "mean": float(tt.mean().item()),
        "std": float(tt.std(unbiased=False).item()),
    }


def _tensor_preview(
    name: str,
    kind: str,
    t,
    max_values: int,
) -> TensorPreview:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("This utility requires torch at runtime.") from e

    if t is None:
        return TensorPreview(
            name=name,
            kind=kind,
            shape=tuple(),
            dtype="(None)",
            device="(None)",
            requires_grad=None,
            numel=0,
            preview_flat=[],
            stats=None,
        )

    if not isinstance(t, torch.Tensor):
        # Some weird buffers might not be tensors; serialize as string.
        s = str(t)
        return TensorPreview(
            name=name,
            kind=kind,
            shape=tuple(),
            dtype=f"(non-tensor: {type(t).__name__})",
            device="(n/a)",
            requires_grad=None,
            numel=0,
            preview_flat=[s[:256]],
            stats=None,
        )

    is_meta = getattr(t, "is_meta", False)
    shape = tuple(t.shape)
    numel = int(t.numel())
    dtype = str(t.dtype)
    device = str(t.device)
    requires_grad = bool(getattr(t, "requires_grad", False)) if kind == "param" else None

    if is_meta:
        return TensorPreview(
            name=name,
            kind=kind,
            shape=shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            numel=numel,
            preview_flat=["<meta tensor: no data>"],
            stats=None,
        )

    # Copy only the first few values to CPU to keep this light.
    flat = t.detach().reshape(-1)
    n = min(max_values, flat.numel())
    if n == 0:
        preview = []
    else:
        snippet = flat[:n]
        if snippet.is_cuda:
            snippet = snippet.cpu()
        preview = snippet.tolist()

    return TensorPreview(
        name=name,
        kind=kind,
        shape=shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        numel=numel,
        preview_flat=preview,
        stats=_safe_stats(t),
    )


def iter_module_tensor_previews(
    module,
    *,
    include_buffers: bool = False,
    max_values: int = 32,
    name_regex: Optional[str] = None,
    sort: bool = True,
):
    """
    Yield TensorPreview for a module's parameters (and optionally buffers).
    Values are capped by `max_values` (flattened).
    """
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError("This utility requires torch at runtime.") from e

    pat = re.compile(name_regex) if name_regex else None

    def _want(n: str) -> bool:
        return True if pat is None else (pat.search(n) is not None)

    params = list(module.named_parameters(recurse=True))
    if sort:
        params.sort(key=lambda kv: kv[0])
    for n, p in params:
        if not _want(n):
            continue
        yield _tensor_preview(n, "param", p, max_values=max_values)

    if include_buffers:
        bufs = list(module.named_buffers(recurse=True))
        if sort:
            bufs.sort(key=lambda kv: kv[0])
        for n, b in bufs:
            if not _want(n):
                continue
            yield _tensor_preview(n, "buffer", b, max_values=max_values)


def dump_module_tensors(
    module,
    output_path: str | Path,
    include_buffers: bool = False,
    max_values: int = 32,
    name_regex: Optional[str] = None,
    sort: bool = True,
    format: str = "txt",  # "txt" | "jsonl"
):
    """
    Dump module parameters/buffers to a file.

    - **txt**: one line per tensor
    - **jsonl**: one JSON object per line (easy to parse)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower().strip()
    if fmt not in {"txt", "py"}:
        raise ValueError(f"Unsupported format={format!r}. Use 'txt' or 'py'.")

    with output_path.open("w", encoding="utf-8") as f:
        _write_header(f, module, include_buffers=include_buffers, max_values=max_values, name_regex=name_regex)
        for item in iter_module_tensor_previews(
            module,
            include_buffers=include_buffers,
            max_values=max_values,
            name_regex=name_regex,
            sort=sort,
        ):
            if fmt == "py":
                f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
            else:
                stats = ""
                if item.stats is not None:
                    stats = (
                        f" | min={item.stats['min']:.6g} max={item.stats['max']:.6g}"
                        f" mean={item.stats['mean']:.6g} std={item.stats['std']:.6g}"
                    )
                f.write(
                    f"{item.kind}:{item.name} | shape={tuple(item.shape)} dtype={item.dtype} device={item.device}"
                    + (f" requires_grad={item.requires_grad}" if item.requires_grad is not None else "")
                    + f" numel={item.numel}{stats} | first_{len(item.preview_flat)}={item.preview_flat}\n"
                )


def _write_header(
    f: TextIO,
    module,
    *,
    include_buffers: bool,
    max_values: int,
    name_regex: Optional[str],
):
    cls_name = module.__class__.__name__
    try:
        import torch

        n_params = sum(p.numel() for p in module.parameters())
        n_trainable = sum(p.numel() for p in module.parameters() if getattr(p, "requires_grad", False))
        approx_bytes = 0
        for p in module.parameters():
            if getattr(p, "is_meta", False):
                continue
            approx_bytes += p.numel() * torch.tensor([], dtype=p.dtype).element_size()
        approx_mb = approx_bytes / (1024 ** 2) if approx_bytes > 0 else 0.0
    except Exception:
        n_params = None
        n_trainable = None
        approx_mb = None

    f.write(f"# module_class: {cls_name}\n")
    if n_params is not None:
        f.write(f"# total_params: {n_params}\n")
        f.write(f"# trainable_params: {n_trainable}\n")
    if approx_mb is not None:
        f.write(f"# approx_param_size_mb: {approx_mb:.2f}\n")
    f.write(f"# include_buffers: {include_buffers}\n")
    f.write(f"# max_values_per_tensor(flat): {max_values}\n")
    f.write(f"# name_regex: {name_regex!r}\n")
    f.write("\n")







