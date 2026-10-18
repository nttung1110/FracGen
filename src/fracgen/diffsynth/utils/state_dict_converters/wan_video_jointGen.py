import re
from collections import defaultdict
import torch


def _strip_model_prefix(name: str) -> str:
    return name[len("model."):] if name.startswith("model.") else name


_BLOCK_RE = re.compile(r"^(blocks|geo_blocks)\.(\d+)\.(.+)$")


def _parse_block_key(name: str):
    """
    Returns (kind, idx, suffix) or None.
      kind: "blocks" | "geo_blocks"
      idx: int
      suffix: str (everything after idx + '.')
    """
    m = _BLOCK_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def WanVideoDiTStateDictConverter(state_dict, model=None):
    """
    Converter for loading a WAN video DiT checkpoint into JointGen models.

    Key behaviors:
      - Drops unrelated modules (pose/face/motion).
      - Strips leading "model." prefix when present.
      - If checkpoint has no `geo_blocks.*`, initializes `geo_blocks.*` by copying from
        `blocks.*` using the JointGen model's `vace_layers` mapping when available.
      - Optionally initializes `geo_patch_embedding.*` and `geo_head.*` in a shape-aware
        way when `model` is provided and source weights exist.
      - Does NOT auto-copy any `after_proj*` params between streams.
    """
    # normalized_name -> original_name (so we can index `state_dict[original]`)
    norm_to_orig = {}
    block_params = defaultdict(list)  # block_id -> list[(suffix, original_name)]
    geo_present = False

    for orig_name in state_dict:
        norm = _strip_model_prefix(orig_name)
        norm_to_orig[norm] = orig_name

        parsed = _parse_block_key(norm)
        if parsed is not None:
            kind, idx, suffix = parsed
            if kind == "blocks":
                block_params[idx].append((suffix, orig_name))
            elif kind == "geo_blocks":
                geo_present = True

    # 1) First: copy through all existing keys (after stripping prefixes / excluding roots)
    state_dict_ = {}
    for norm, orig in norm_to_orig.items():
        state_dict_[norm] = state_dict[orig]

    # 2) Create geo_blocks from blocks if missing in checkpoint
    if not geo_present:
        # `geo_blocks` are stored compactly, so geo_idx corresponds to vace_layers position.
        vace_layers = list(getattr(model, "vace_layers"))
        geo_idx_to_block_idx = {geo_idx: int(block_id) for geo_idx, block_id in enumerate(vace_layers)}

        # Copy params
        for geo_idx, src_block_idx in geo_idx_to_block_idx.items():
            assert src_block_idx in block_params, f"Cannot find source block {src_block_idx} for geo_block {geo_idx}"
            for suffix, orig in block_params[src_block_idx]:
                # PS: keep after_proj params as-is (don't auto-copy them into geo stream)
                if "after_proj" in suffix:
                    continue
                geo_key = f"geo_blocks.{geo_idx}.{suffix}"
                # don't override if somehow present
                if geo_key in state_dict_:
                    continue
                state_dict_[geo_key] = state_dict[orig]

        # 3) Optional: head -> geo_head, patch_embedding -> geo_patch_embedding (shape-aware).
        #    Only do this when we have a model instance to know target shapes.
        if model is not None:
            # 2.5) Ensure any `after_proj*` params that exist in the JointGen model but are
            # missing from the checkpoint get explicit zero tensors 
            for name, param in model.named_parameters(recurse=True):
                if "after_proj" not in name:
                    continue
                assert name not in state_dict_, f"Unexpectedly found '{name}' in checkpoint."
                state_dict_[name] = torch.zeros(tuple(param.shape), dtype=param.dtype, device="cpu")

            # 3a) geo_patch_embedding
            # Only copy when shapes match exactly; otherwise abort this copy.
            assert hasattr(model, "geo_patch_embedding"), "Model missing expected geo_patch_embedding module."
            can_copy_geo_patch = True
            src_w = state_dict_.get("patch_embedding.weight", None)
            src_b = state_dict_.get("patch_embedding.bias", None)
            tgt_w = getattr(model.geo_patch_embedding, "weight", None)
            tgt_b = getattr(model.geo_patch_embedding, "bias", None)

            if not (hasattr(src_w, "shape") and tgt_w is not None and hasattr(tgt_w, "shape")):
                can_copy_geo_patch = False
            elif tuple(src_w.shape) != tuple(tgt_w.shape):
                can_copy_geo_patch = False

            # Bias must match too (when both exist)
            if can_copy_geo_patch:
                if (src_b is None) != (tgt_b is None):
                    can_copy_geo_patch = False
                elif src_b is not None:
                    if not (hasattr(src_b, "shape") and hasattr(tgt_b, "shape")):
                        can_copy_geo_patch = False
                    elif tuple(src_b.shape) != tuple(tgt_b.shape):
                        can_copy_geo_patch = False

            if can_copy_geo_patch:
                state_dict_["geo_patch_embedding.weight"] = src_w
                if src_b is not None:
                    state_dict_["geo_patch_embedding.bias"] = src_b

            # 3b) geo_head
            assert hasattr(model, "geo_head"), "Model missing expected geo_head module."
            # Only copy when shapes match exactly; otherwise abort this copy.
            can_copy_geo_head = True
            src_mod = state_dict_.get("head.modulation", None)
            src_w = state_dict_.get("head.head.weight", None)
            src_b = state_dict_.get("head.head.bias", None)
            tgt_mod = getattr(model.geo_head, "modulation", None)
            tgt_head = getattr(model.geo_head, "head", None)
            tgt_w = getattr(tgt_head, "weight", None)
            tgt_b = getattr(tgt_head, "bias", None)

            # modulation (must exist + match)
            if not (hasattr(src_mod, "shape") and tgt_mod is not None and hasattr(tgt_mod, "shape")):
                can_copy_geo_head = False
            elif tuple(src_mod.shape) != tuple(tgt_mod.shape):
                can_copy_geo_head = False

            # linear weight (must exist + match)
            if can_copy_geo_head:
                if not (hasattr(src_w, "shape") and tgt_w is not None and hasattr(tgt_w, "shape")):
                    can_copy_geo_head = False
                elif tuple(src_w.shape) != tuple(tgt_w.shape):
                    can_copy_geo_head = False

            # linear bias must match too (when both exist)
            if can_copy_geo_head:
                if (src_b is None) != (tgt_b is None):
                    can_copy_geo_head = False
                elif src_b is not None:
                    if not (hasattr(src_b, "shape") and hasattr(tgt_b, "shape")):
                        can_copy_geo_head = False
                    elif tuple(src_b.shape) != tuple(tgt_b.shape):
                        can_copy_geo_head = False

            if can_copy_geo_head:
                state_dict_["geo_head.modulation"] = src_mod
                state_dict_["geo_head.head.weight"] = src_w
                if src_b is not None:
                    state_dict_["geo_head.head.bias"] = src_b

    return state_dict_

# with open("temp_weight.py", "w") as f:
#     print(list(state_dict_.keys()), file=f)
#     print(f"len: {len(list(state_dict_.keys()))}", file=f)