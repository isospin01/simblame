"""Load a policy from a short string, so common cases need no code.

==============================  ==================================================
spec                            policy
==============================  ==================================================
``zero`` (default)              all-zero actions
``random``                      seeded uniform samples from the action space
``path/to/model.zip``           a stable-baselines3 model (algorithm auto-detected)
``org/repo``                    a stable-baselines3 model from the HuggingFace Hub
                                (the most recently uploaded .zip; pick one with
                                ``org/repo/file.zip``)
``file.py:make`` / ``pkg.mod:make``  your own factory ``make(variant) -> policy``
==============================  ==================================================

Any object with ``get_action(obs)`` / ``act`` / ``predict`` or a plain callable
works as a policy.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import zipfile
from pathlib import Path

import numpy as np

from .base import HarnessError, import_target

SB3_ALGOS = ("SAC", "TD3", "PPO", "A2C", "DDPG", "DQN")
# Training-only objects pickled as Python code; skipping them lets models saved with
# another Python version load without warnings (they are not used for inference).
INFERENCE_ONLY = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0,
                  "clip_range": lambda _: 0.0, "exploration_schedule": lambda _: 0.0}
CONTRIB_ALGOS = ("TQC", "TRPO", "ARS", "CrossQ", "RecurrentPPO", "QRDQN")


class ZeroPolicy:
    def __init__(self, action_shape=None):
        self.action_shape = action_shape

    def get_action(self, obs):
        return np.zeros(self.action_shape or (), dtype=float)


class RandomPolicy:
    """Uniform random actions from a fixed seed (reset per episode for determinism)."""

    def __init__(self, low, high, seed: int = 0):
        self.low, self.high, self.seed = np.asarray(low), np.asarray(high), seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def get_action(self, obs):
        return self.rng.uniform(self.low, self.high)


class SB3Policy:
    """Deterministic actions from a stable-baselines3 model."""

    def __init__(self, model):
        self.model = model

    def get_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def load_policy(spec: str | None, variant: str = "", action_space=None):
    """Turn a policy spec string (see module docstring) into a policy object."""
    if spec in (None, "", "zero", "none"):
        return ZeroPolicy(getattr(action_space, "shape", None))
    if spec == "random":
        if action_space is None:
            raise HarnessError("the 'random' policy needs an environment with an action_space")
        return RandomPolicy(action_space.low, action_space.high)
    target, _, attr = spec.partition(":")
    if attr and (target.endswith(".py") or "." in target):
        return _from_factory(target, attr, variant)
    path = Path(spec).expanduser()
    if path.exists():
        return SB3Policy(_load_sb3(path))
    if spec.count("/") == 1 and not spec.endswith(".zip"):
        return SB3Policy(_load_sb3(_download_hub_model(spec)))
    if spec.count("/") == 2 and spec.endswith(".zip"):
        org, repo, filename = spec.split("/")
        return SB3Policy(_load_sb3(_download_hub_model(f"{org}/{repo}", filename)))
    if path.suffix == ".zip":
        raise HarnessError(f"policy file {spec} not found")
    raise HarnessError(f"don't know how to load policy {spec!r}. Use zero, random, a "
                       "stable-baselines3 .zip, a HuggingFace repo id, or file.py:factory")


def _from_factory(target: str, attr: str, variant: str):
    module = import_target(target)
    if not hasattr(module, attr):
        raise HarnessError(f"{target} has no attribute {attr!r}")
    obj = getattr(module, attr)
    if callable(obj) and not hasattr(obj, "get_action"):
        params = inspect.signature(obj).parameters
        return obj(variant) if params else obj()
    return obj


def _load_sb3(path: Path):
    try:
        import stable_baselines3 as sb3
    except ImportError as e:
        raise HarnessError("loading a .zip policy needs stable-baselines3: "
                           "pip install stable-baselines3") from e
    algo_name = _sb3_algorithm(path)
    candidates = [algo_name] if algo_name else list(SB3_ALGOS + CONTRIB_ALGOS)
    for name in candidates:
        cls = getattr(sb3, name, None)
        if cls is None:
            try:
                import sb3_contrib

                cls = getattr(sb3_contrib, name, None)
            except ImportError:
                cls = None
        if cls is None:
            continue
        try:
            return cls.load(str(path), device="cpu", custom_objects=INFERENCE_ONLY)
        except Exception:
            if algo_name:
                raise
    raise HarnessError(f"could not load {path} with any stable-baselines3 algorithm")


def _sb3_algorithm(path: Path) -> str | None:
    """Read the algorithm name from the policy class module stored in the zip."""
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data"))
        module = data.get("policy_class", {}).get("__module__", "")
    except Exception:
        return None
    # e.g. "stable_baselines3.sac.policies" -> "SAC", "sb3_contrib.tqc.policies" -> "TQC"
    parts = module.split(".")
    if len(parts) >= 2:
        short = parts[1]
        for name in SB3_ALGOS + CONTRIB_ALGOS:
            if name.lower() == short:
                return name
    return None


def _download_hub_model(repo_id: str, filename: str | None = None) -> Path:
    """Download one .zip from a HuggingFace repo (default: the most recently uploaded)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise HarnessError("downloading a policy from the HuggingFace Hub needs huggingface_hub: "
                           "pip install huggingface_hub") from e
    with _quiet_hub():
        if filename is None:
            zips = _hub_zips(repo_id)
            if not zips:
                raise HarnessError(f"no stable-baselines3 .zip found in HuggingFace repo {repo_id}")
            filename = zips[0]
        return Path(hf_hub_download(repo_id, filename))


@contextlib.contextmanager
def _quiet_hub():
    """Hide the Hub's advisory log lines (e.g. 'set HF_TOKEN') during a download."""
    logger = logging.getLogger("huggingface_hub")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


def _hub_zips(repo_id: str) -> list:
    """Top-level .zip files of a repo, newest upload first."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        entries = [e for e in api.list_repo_tree(repo_id, expand=True)
                   if e.path.endswith(".zip") and "/" not in e.path]
        entries.sort(key=lambda e: (getattr(getattr(e, "last_commit", None), "date", None)
                                    or 0, e.path), reverse=True)
        return [e.path for e in entries]
    except Exception:  # older huggingface_hub: no dates, fall back to names
        return sorted(f for f in api.list_repo_files(repo_id) if f.endswith(".zip") and "/" not in f)
