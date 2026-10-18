from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict
import torch
from typing import Union


def _materialize_meta_tensors_(model: torch.nn.Module, device: Union[str, torch.device]):
    """
    Some checkpoints intentionally omit keys (e.g., optional heads / adapters).
    When we instantiate under `skip_model_initialization()`, those missing params/buffers
    stay on the `meta` device. Calling `module.to(...)` would then raise:
      "Cannot copy out of meta tensor; no data! ..."

    We only materialize tensors that are still `meta` (so we don't clobber already-loaded
    weights that were assigned from the checkpoint via `assign=True`).
    """

    # "disk" isn't a real torch device; fall back to CPU for materialization.
    if device == "disk":
        device = "cpu"

    def _resolve_parent_module(root: torch.nn.Module, qualified_name: str):
        parts = qualified_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    # Parameters
    for name, param in model.named_parameters(recurse=True):
        if param is None or not getattr(param, "is_meta", False):
            continue
        parent, attr = _resolve_parent_module(model, name)
        old = getattr(parent, attr)
        # Preserve Parameter subclass + any custom kwargs stored in __dict__
        param_cls = type(old)
        kwargs = dict(getattr(old, "__dict__", {}))
        kwargs["requires_grad"] = old.requires_grad
        new_data = torch.empty(old.shape, device=device, dtype=old.dtype)
        setattr(parent, attr, param_cls(new_data, **kwargs))

    # Buffers
    for name, buf in model.named_buffers(recurse=True):
        if buf is None or not getattr(buf, "is_meta", False):
            continue
        parent, attr = _resolve_parent_module(model, name)
        old = getattr(parent, attr)
        new_buf = torch.empty(old.shape, device=device, dtype=old.dtype)
        # Keep buffer registration/persistence behavior unchanged.
        parent._buffers[attr] = new_buf
        setattr(parent, attr, new_buf)


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None, strict=True):
    config = {} if config is None else config
    # Why do we use `skip_model_initialization`?
    # It skips the random initialization of model parameters,
    # thereby speeding up model loading and avoiding excessive memory usage.
    # with skip_model_initialization():     #HACK:
    model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                # Support both `converter(state_dict)` and `converter(state_dict, model)`.
                try:
                    state_dict = state_dict_converter(state_dict, model)
                except TypeError:
                    state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            m, p = model.load_state_dict(state_dict, strict=strict, assign=True)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            # Support both `converter(state_dict)` and `converter(state_dict, model)`.
            try:
                state_dict = state_dict_converter(state_dict, model)
            except TypeError:
                state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        m, p = model.load_state_dict(state_dict, strict=strict, assign=True)

        # Materialize any remaining meta tensors (typically missing params/buffers)
        # before we try to initialize them or move the module.
        # _materialize_meta_tensors_(model, device)

        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        model = model.to(dtype=torch_dtype, device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    if len(m) > 0 or len(p) > 0:
        print(f"WARNING: when loading model {model_class}, \n#### Missing keys {len(m)}: {m}, \n#### Unexpected keys {len(p)}: {p}")
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model
