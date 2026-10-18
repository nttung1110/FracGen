import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from accelerate import Accelerator
from diffsynth.core.loader.file import load_state_dict as _load_state_dict


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """
    Resolved checkpoint info.
    - path: full path to the checkpoint file
    - kind: typically "epoch" or "step" (based on filename)
    - index: integer suffix parsed from filename (e.g. epoch-12 -> 12)
    """
    path: str
    kind: str
    index: int


_DEFAULT_CKPT_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("epoch", re.compile(r"^epoch-(\d+)\.safetensors$")),
    ("step", re.compile(r"^step-(\d+)\.safetensors$")),
)


def _normalize_resume_spec(resume_from_checkpoint: Optional[str]) -> Optional[str]:
    if resume_from_checkpoint is None:
        return None
    s = str(resume_from_checkpoint).strip()
    if s == "":
        return None
    if s.lower() in {"none", "null", "false", "0"}:
        return None
    return s


def list_numbered_checkpoints(
    output_dir: str,
    patterns: Sequence[Tuple[str, re.Pattern]] = _DEFAULT_CKPT_PATTERNS,
) -> List[ResolvedCheckpoint]:
    """
    List checkpoint files directly under `output_dir` that match numbered patterns.
    """
    if output_dir is None or not os.path.isdir(output_dir):
        return []

    out: List[ResolvedCheckpoint] = []
    for name in os.listdir(output_dir):
        for kind, pat in patterns:
            m = pat.match(name)
            if m:
                out.append(
                    ResolvedCheckpoint(
                        path=os.path.join(output_dir, name),
                        kind=kind,
                        index=int(m.group(1)),
                    )
                )
                break
    return out


def resolve_resume_checkpoint(
    output_dir: str,
    resume_from_checkpoint: Optional[str],
    patterns: Sequence[Tuple[str, re.Pattern]] = _DEFAULT_CKPT_PATTERNS,
    prefer_kinds: Sequence[str] = ("epoch", "step"),
) -> Optional[ResolvedCheckpoint]:
    """
    Resolve a resume checkpoint based on a user-specified spec and `output_dir`.

    - resume_from_checkpoint = "auto"/"latest": picks the newest checkpoint under `output_dir`
      (by kind preference then by largest index).
    - resume_from_checkpoint = filename: tries `<output_dir>/<filename>`
    - resume_from_checkpoint = path: tries that path directly
    - resume_from_checkpoint = None/"none": returns None
    """
    spec = _normalize_resume_spec(resume_from_checkpoint)
    if spec is None:
        return None

    spec_l = spec.lower()
    if spec_l in {"auto", "latest"}:
        ckpts = list_numbered_checkpoints(output_dir, patterns=patterns)
        if not ckpts:
            return None
        best: Optional[ResolvedCheckpoint] = None
        for kind in prefer_kinds:
            candidates = [c for c in ckpts if c.kind == kind]
            if candidates:
                best = max(candidates, key=lambda x: x.index)
                break
        return best or max(ckpts, key=lambda x: (x.kind, x.index))

    candidate_paths = [spec]
    if output_dir is not None:
        candidate_paths.append(os.path.join(output_dir, spec))
        candidate_paths.append(os.path.join(output_dir, os.path.basename(spec)))

    for p in candidate_paths:
        if p and os.path.isfile(p):
            base = os.path.basename(p)
            for kind, pat in patterns:
                m = pat.match(base)
                if m:
                    return ResolvedCheckpoint(path=p, kind=kind, index=int(m.group(1)))
            return ResolvedCheckpoint(path=p, kind="file", index=0)

    return None


def infer_resume_counters(
    ckpt: ResolvedCheckpoint,
    num_update_steps_per_epoch: int,
) -> Dict[str, int]:
    """
    Infer start counters (best-effort) from a checkpoint filename.

    Returns:
      - start_epoch: epoch index to start training loop from
      - global_step: optimizer-step counter (best-effort)
      - logger_num_steps: ModelLogger.num_steps (best-effort)
    """
    if ckpt.kind == "epoch":
        # epoch-K is treated as "finished epoch K"; next epoch is K+1
        start_epoch = ckpt.index + 1
        global_step = start_epoch * int(num_update_steps_per_epoch)
        return {"start_epoch": start_epoch, "global_step": global_step, "logger_num_steps": global_step}
    if ckpt.kind == "step":
        global_step = int(ckpt.index)
        return {"start_epoch": 0, "global_step": global_step, "logger_num_steps": global_step}
    return {"start_epoch": 0, "global_step": 0, "logger_num_steps": 0}


def trainer_state_path_for_weights(weights_path: str) -> str:
    """
    For weights file `<name>.safetensors`, use `<name>.trainer_state.pt`.
    """
    if weights_path.endswith(".safetensors"):
        return weights_path[: -len(".safetensors")] + ".trainer_state.pt"
    return weights_path + ".trainer_state.pt"


def load_trainer_state(state_path: str) -> Optional[Dict[str, Any]]:
    if state_path is None or not os.path.isfile(state_path):
        return None
    try:
        return torch.load(state_path, map_location="cpu")
    except Exception:
        return None


def save_trainer_state(accelerator: Optional[Accelerator], state: Dict[str, Any], state_path: str) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    if accelerator is not None and hasattr(accelerator, "save"):
        accelerator.save(state, state_path)
    else:
        torch.save(state, state_path)


def _map_checkpoint_keys_for_model(
    checkpoint_state_dict: Dict[str, torch.Tensor],
    model_state_keys: Iterable[str],
    add_prefix_if_missing: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    model_keys = set(model_state_keys)
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in checkpoint_state_dict.items():
        if k in model_keys:
            mapped[k] = v
            continue
        if add_prefix_if_missing:
            k2 = add_prefix_if_missing + k
            if k2 in model_keys:
                mapped[k2] = v
                continue
        mapped[k] = v
    return mapped


def load_checkpoint_weights_into_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    remove_prefix_in_ckpt: Optional[str] = None,
    strict: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load a weights-only checkpoint (e.g. `epoch-*.safetensors`) into `model`.
    """
    sd = _load_state_dict(checkpoint_path, torch_dtype=torch_dtype, device="cpu")
    mapped = _map_checkpoint_keys_for_model(sd, model.state_dict().keys(), add_prefix_if_missing=remove_prefix_in_ckpt)
    incompatible = model.load_state_dict(mapped, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    return missing, unexpected


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        optimizer_state_dtype: Optional[str] = "bf16",
        max_ckpt_to_keep: Optional[int] = 2,
    ):
        self.output_path = os.path.join(output_path, "ckpts")
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        # If set, optimizer state tensors are cast before saving to reduce disk usage.
        # Examples: "bf16", "fp16", "fp32"/None.
        self.optimizer_state_dtype = optimizer_state_dtype
        self.max_ckpt_to_keep = max_ckpt_to_keep


    def on_step_end(self, accelerator, model, save_steps=None,
                    optimizer=None, scheduler=None, epoch_id=0, global_step=0):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_step_checkpoint(
                accelerator, model, self.num_steps,
                optimizer=optimizer, scheduler=scheduler,
                epoch_id=epoch_id, global_step=global_step,
            )

    def save_step_checkpoint(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        step_id: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch_id: int = 0,
        global_step: int = 0,
    ) -> str:
        """Save `step-{step_id}.safetensors` + sidecar `step-{step_id}.trainer_state.pt`."""
        self.save_model(accelerator, model, f"step-{step_id}.safetensors")
        weights_path = self.weights_path_for_step(step_id)
        state_path = trainer_state_path_for_weights(weights_path)
        trainer_state: Dict[str, Any] = {
            "kind": "step",
            "step_id": int(step_id),
            "epoch_id": int(epoch_id),
            "start_epoch": int(epoch_id),
            "global_step": int(global_step),
            "model_logger_num_steps": int(self.num_steps),
        }
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            dtype = _parse_dtype(self.optimizer_state_dtype)
            if dtype is not None:
                opt_state = _cast_optimizer_state_dict(opt_state, dtype=dtype)
                trainer_state["optimizer_state_dtype"] = str(dtype).replace("torch.", "")
            trainer_state["optimizer"] = opt_state
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            trainer_state["scheduler"] = scheduler.state_dict()
        save_trainer_state(accelerator, trainer_state, state_path)
        return weights_path


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_epoch_checkpoint(accelerator, model, epoch_id)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            self._cleanup_old_step_checkpoints(keep=2)

    def _cleanup_old_step_checkpoints(self, keep: int = 2) -> None:
        pattern = re.compile(r"^step-(\d+)\.safetensors$")
        files = []
        for name in os.listdir(self.output_path):
            m = pattern.match(name)
            if m:
                files.append((int(m.group(1)), os.path.join(self.output_path, name)))
        files.sort(key=lambda x: x[0])
        for _, weights_path in files[:-keep] if len(files) > keep else []:
            state_path = trainer_state_path_for_weights(weights_path)
            if os.path.isfile(weights_path):
                os.remove(weights_path)
            if os.path.isfile(state_path):
                os.remove(state_path)


    # ------------------------
    # Resume + trainer-state IO
    # ------------------------
    def weights_path_for_epoch(self, epoch_id: int) -> str:
        return os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")

    def weights_path_for_step(self, step_id: int) -> str:
        return os.path.join(self.output_path, f"step-{step_id}.safetensors")

    def save_epoch_checkpoint(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        epoch_id: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        global_step: Optional[int] = None,
        extra_trainer_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save `epoch-{epoch_id}.safetensors` plus a sidecar `epoch-{epoch_id}.trainer_state.pt`.
        Returns the weights path.
        """
        self._save_weights_epoch(accelerator, model, epoch_id)
        weights_path = self.weights_path_for_epoch(epoch_id)
        state_path = trainer_state_path_for_weights(weights_path)
        trainer_state: Dict[str, Any] = {
            "kind": "epoch",
            "epoch_id": int(epoch_id),
            "next_epoch": int(epoch_id) + 1,
            "start_epoch": int(epoch_id) + 1,
            "global_step": int(global_step) if global_step is not None else None,
            "model_logger_num_steps": int(self.num_steps),
        }
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            dtype = _parse_dtype(self.optimizer_state_dtype)
            if dtype is not None:
                opt_state = _cast_optimizer_state_dict(opt_state, dtype=dtype)
                trainer_state["optimizer_state_dtype"] = str(dtype).replace("torch.", "")
            trainer_state["optimizer"] = opt_state
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            trainer_state["scheduler"] = scheduler.state_dict()
        if extra_trainer_state:
            trainer_state.update(extra_trainer_state)
        save_trainer_state(accelerator, trainer_state, state_path)
        self._cleanup_old_epoch_artifacts(accelerator)
        return weights_path

    def maybe_resume(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        args: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        num_update_steps_per_epoch: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Resume training from `args.resume_from_checkpoint` within `self.output_path` (or an explicit path).

        Returns (start_epoch, global_step). If no checkpoint found, returns (0, 0).
        """
        start_epoch, global_step = 0, 0
        resume_spec = getattr(args, "resume_from_checkpoint", None)
        ckpt = resolve_resume_checkpoint(self.output_path, resume_spec)
        if ckpt is None:
            return start_epoch, global_step

        accelerator.print(f"[Resume] Loading checkpoint: {ckpt.path}")
        accelerator.wait_for_everyone()

        strict = not bool(getattr(args, "no_resume_strict", False))
        unwrapped = accelerator.unwrap_model(model)
        missing, unexpected = load_checkpoint_weights_into_model(
            unwrapped,
            ckpt.path,
            remove_prefix_in_ckpt=self.remove_prefix_in_ckpt,
            strict=strict,
        )
        accelerator.print(f"[Resume] Loaded with strict={strict}; missing={len(missing)}, unexpected={len(unexpected)}")

        if num_update_steps_per_epoch is None:
            # Only needed to infer global_step from an epoch checkpoint name.
            num_update_steps_per_epoch = 0

        counters = infer_resume_counters(ckpt, num_update_steps_per_epoch=num_update_steps_per_epoch)
        start_epoch = int(counters["start_epoch"])
        global_step = int(counters["global_step"])
        self.num_steps = int(counters["logger_num_steps"])

        # Sidecar state (if exists)
        state_path = trainer_state_path_for_weights(ckpt.path)
        trainer_state = load_trainer_state(state_path)
        if trainer_state is not None:
            start_epoch = int(trainer_state.get("start_epoch", trainer_state.get("next_epoch", start_epoch)))
            global_step = int(trainer_state.get("global_step", global_step))
            self.num_steps = int(trainer_state.get("model_logger_num_steps", self.num_steps))

            if optimizer is not None and "optimizer" in trainer_state:
                try:
                    optimizer.load_state_dict(trainer_state["optimizer"])
                    # Align state with current parameter dtype/device (avoids bf16/fp16 resume errors).
                    _align_optimizer_state_with_params_inplace(optimizer)
                except Exception as e:
                    accelerator.print(f"[Resume] Warning: failed to load optimizer state: {e}")
            if scheduler is not None and "scheduler" in trainer_state and hasattr(scheduler, "load_state_dict"):
                try:
                    scheduler.load_state_dict(trainer_state["scheduler"])
                except Exception as e:
                    accelerator.print(f"[Resume] Warning: failed to load scheduler state: {e}")

            # Keep your current behavior: override LR with args.learning_rate after resume
            desired_lr = getattr(args, "learning_rate", None)
            if optimizer is not None and desired_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = desired_lr
                    if "initial_lr" in param_group:
                        param_group["initial_lr"] = desired_lr
                if scheduler is not None and hasattr(scheduler, "base_lrs"):
                    scheduler.base_lrs = [desired_lr for _ in scheduler.base_lrs]
                if scheduler is not None and hasattr(scheduler, "_last_lr"):
                    scheduler._last_lr = [desired_lr for _ in scheduler._last_lr]

        accelerator.print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}, model_logger.num_steps={self.num_steps}")
        return start_epoch, global_step

    def _save_weights_epoch(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id: int) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = self.weights_path_for_epoch(epoch_id)
            accelerator.save(state_dict, path, safe_serialization=True)

    def _cleanup_old_epoch_artifacts(self, accelerator: Optional[Accelerator]) -> None:
        """
        Keep only the latest N `epoch-*.safetensors` and `epoch-*.trainer_state.pt` files to reduce disk usage.
        """
        n = self.max_ckpt_to_keep
        if n is None:
            return
        if accelerator is not None and (not accelerator.is_main_process):
            return
        os.makedirs(self.output_path, exist_ok=True)
        # Gather epoch indices from epoch-*.safetensors (source of truth)
        pattern = re.compile(r"^epoch-(\d+)\.safetensors$")
        files: List[Tuple[int, str]] = []
        for name in os.listdir(self.output_path):
            m = pattern.match(name)
            if m:
                files.append((int(m.group(1)), os.path.join(self.output_path, name)))
        files.sort(key=lambda x: x[0])
        # Keep only last N
        to_delete_epochs = [e for e, _ in (files[:-n] if len(files) > n else [])]
        for e in to_delete_epochs:
            weights_path = self.weights_path_for_epoch(e)
            state_path = trainer_state_path_for_weights(weights_path)
            if os.path.isfile(weights_path):
                os.remove(weights_path)
            if os.path.isfile(state_path):
                os.remove(state_path)


def _parse_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    s = str(dtype).strip().lower()
    if s in {"none", "null", ""}:
        return None
    if s in {"fp32", "float32", "f32"}:
        return torch.float32
    if s in {"fp16", "float16", "f16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None


def _cast_optimizer_state_dict(opt_state: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
    """
    Cast floating tensors in optimizer state to a smaller dtype to reduce disk usage.
    This only affects the saved checkpoint; we can upcast back to fp32 on resume.
    """
    # Shallow copy
    out: Dict[str, Any] = {"state": {}, "param_groups": opt_state.get("param_groups", [])}
    state = opt_state.get("state", {})
    for k, v in state.items():
        if isinstance(v, dict):
            vv: Dict[str, Any] = {}
            for kk, t in v.items():
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    vv[kk] = t.to(dtype=dtype, device="cpu")
                else:
                    vv[kk] = t
            out["state"][k] = vv
        else:
            out["state"][k] = v
    return out


def _upcast_optimizer_state_inplace(optimizer: torch.optim.Optimizer, target_dtype: torch.dtype = torch.float32) -> None:
    """
    Upcast floating optimizer state tensors in-place (helps stability when saved in bf16/fp16).
    """
    for _, st in optimizer.state.items():
        if isinstance(st, dict):
            for kk, t in list(st.items()):
                if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != target_dtype:
                    st[kk] = t.to(dtype=target_dtype)


def _unwrap_optimizer(optimizer: Any) -> torch.optim.Optimizer:
    """
    Accelerate may wrap optimizers (e.g. AcceleratedOptimizer) and keep the real optimizer at `.optimizer`.
    """
    return getattr(optimizer, "optimizer", optimizer)


def _align_optimizer_state_with_params_inplace(optimizer: Any) -> None:
    """
    Ensure optimizer state tensors match the dtype/device of their corresponding params.

    This avoids dtype/device mismatches on resume, especially when training weights are bf16/fp16.
    """
    opt = _unwrap_optimizer(optimizer)
    if not hasattr(opt, "param_groups") or not hasattr(opt, "state"):
        return

    for group in opt.param_groups:
        for p in group.get("params", []):
            if p is None:
                continue
            st = opt.state.get(p, None)
            if not isinstance(st, dict):
                continue
            for kk, t in list(st.items()):
                if not isinstance(t, torch.Tensor):
                    continue
                if not t.is_floating_point():
                    continue
                target_device = p.device
                target_dtype = p.dtype
                if t.device != target_device or t.dtype != target_dtype:
                    st[kk] = t.to(device=target_device, dtype=target_dtype)
