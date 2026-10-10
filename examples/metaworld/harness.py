"""Meta-World examples (scripted expert policies, no training needed).

Install:  pip install "simblame[metaworld]"

1) ``PickPlaceAssetRefresh`` -- a synthetic but realistic regression.
   ``v1.1`` is an "asset refresh" of the stock pick-place scene that edits the
   MJCF files on disk in five places at once:

     - gripper position actuators kp 400 -> 60    (the culprit)
     - finger-pad friction 2.0 -> 1.2             (harmless on its own)
     - object mass 0.75 -> 1.0 kg                 (harmless on its own)
     - table collision box colour                 (rendering only)
     - solver iterations 50 -> 40                 (harmless on its own)

       simblame run examples/metaworld/harness.py:PickPlaceAssetRefresh

2) ``HandInsert`` / ``BoxClose`` -- the *stock* tasks, used to ask which MuJoCo
   release changed their scripted-policy success rate (Meta-World pins
   ``mujoco==3.3.0`` "due to changes in how mujoco represents contacts"):

       simblame versions examples/metaworld/harness.py:HandInsert \
           --mujoco 3.3.0..3.15.0 --seeds 100

Episodes are made reproducible by seeding *explicitly*: seed ``s`` uses task
``s % 50`` of ``metaworld.MT1(task, seed=s // 50)``. (Resetting the gymnasium
``Meta-World/MT1`` wrapper with ``reset(seed=s)`` does not by itself select a
reproducible task -- ``simblame check`` catches that.)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import simblame as sb

warnings.filterwarnings("ignore", module="gymnasium")
warnings.filterwarnings("ignore", module="metaworld")

# (relative file, old text, new text, expected count)
ASSET_REFRESH_V1_1 = [
    ("sawyer_xyz/sawyer_pick_place_v3.xml", 'kp="400"', 'kp="60"', 2),
    ("objects/assets/xyz_base.xml",
     'solref="0.01 1" friction="2 0.1 0.002"', 'solref="0.01 1" friction="1.2 0.1 0.002"', 2),
    ("sawyer_xyz/sawyer_pick_place_v3.xml", 'mass=".75"', 'mass="1.0"', 1),
    ("scene/basic_scene.xml", 'rgba="0.3 0.3 1.0 0.5"', 'rgba="0.45 0.45 0.45 0.5"', 1),
    ("scene/basic_scene.xml", 'iterations="50"', 'iterations="40"', 1),
]


def edited_assets(edits, tag: str) -> Path:
    """Copy Meta-World's asset tree and apply text edits (cached per edit set)."""
    import metaworld

    src = Path(metaworld.__file__).parent / "assets"
    key = hashlib.sha1(repr((str(src), edits)).encode()).hexdigest()[:10]
    dst = Path(tempfile.gettempdir()) / f"simblame-metaworld-{tag}-{key}"
    if (dst / ".ok").exists():
        return dst
    tmp = Path(tempfile.mkdtemp(prefix="simblame-mw-"))
    shutil.copytree(src, tmp / "assets")
    for rel, old, new, expected in edits:
        f = tmp / "assets" / rel
        text = f.read_text()
        n = text.count(old)
        if n != expected:
            raise RuntimeError(f"{rel}: expected {expected}x {old!r}, found {n} "
                               "(different Meta-World version?)")
        f.write_text(text.replace(old, new))
    (tmp / "assets" / ".ok").write_text("ok")
    try:
        os.replace(tmp / "assets", dst)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return dst


class MetaWorldTask(sb.Harness):
    """Base: one Meta-World MT1 task with its scripted expert policy."""

    abstract = True  # not offered by `simblame run examples/metaworld/harness.py`

    task = "pick-place-v3"
    variants = ("stock", "stock")
    max_steps = 500
    stop_on_success = True
    default_seeds = 50
    packages = ("metaworld",)
    # variant name -> list of asset edits (None = stock assets)
    edits: dict = {}

    def __init__(self):
        self._benchmarks = {}

    def _tasks(self, k: int):
        import metaworld

        if k not in self._benchmarks:
            self._benchmarks[k] = metaworld.MT1(self.task, seed=k).train_tasks
        return self._benchmarks[k]

    def make_env(self, variant):
        import metaworld

        cls = metaworld.MT1(self.task, seed=0).train_classes[self.task]
        edits = self.edits.get(variant)
        if edits:
            root = edited_assets(edits, variant)
            xml = Path(cls().model_name).relative_to(Path(metaworld.__file__).parent / "assets")
            path = str(root / xml)
            cls = type(f"{cls.__name__}_{variant.replace('.', '_')}", (cls,),
                       {"model_name": property(lambda self: path)})
        return cls()

    def make_policy(self, variant):
        from metaworld.policies import ENV_POLICY_MAP

        return ENV_POLICY_MAP[self.task]()

    def reset(self, env, seed):
        tasks = self._tasks(seed // 50)
        env.set_task(tasks[seed % 50])
        obs, _ = env.reset(seed=seed)
        return obs

    def get_model(self, env):
        return env.model

    def get_data(self, env):
        return env.data


class PickPlaceAssetRefresh(MetaWorldTask):
    task = "pick-place-v3"
    variants = ("v1.0", "v1.1")
    edits = {"v1.1": ASSET_REFRESH_V1_1}


class HandInsert(MetaWorldTask):
    task = "hand-insert-v3"


class BoxClose(MetaWorldTask):
    task = "box-close-v3"
