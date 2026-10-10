"""Build (and cache) one virtualenv per MuJoCo version for ``simblame versions``.

Default mode **clone**: reproduce the *current* environment exactly (``uv pip
freeze``), changing only ``mujoco``. Nothing else needs to be specified, and
the only difference between two version runs is the simulator.

Mode **fresh**: an empty environment with only ``mujoco`` and the packages
passed with ``--with``.

simblame itself is never reinstalled: a ``.pth`` file points the new
environment at the running copy of the package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .._version import __version__
from ..harness.base import HarnessError

PACKAGE_DIR = Path(__file__).resolve().parents[1]

TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
_LOCAL_TAG = re.compile(r"^torch[a-z_-]*==[^+]+\+(cpu|cu\d+|rocm[\d.]+)$", re.IGNORECASE)


class VersionEnvError(HarnessError):
    """Creating a per-version environment failed (message says why and what to try)."""


def cache_dir() -> Path:
    base = os.environ.get("SIMBLAME_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "simblame")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def frozen_requirements(python: str = sys.executable) -> list:
    """The current environment's packages, minus mujoco and simblame."""
    out = subprocess.run([_uv(), "pip", "freeze", "--python", python],
                         capture_output=True, text=True, check=True).stdout
    keep = []
    for line in out.splitlines():
        line = line.strip()
        name = re.split(r"[=<>@ ]", line.replace("-e ", ""), maxsplit=1)[0].lower()
        if not line or line.startswith("#") or name in ("mujoco", "simblame"):
            continue
        if line.startswith("-e") and "simblame" in line.lower():
            continue
        keep.append(line)
    return keep


def ensure_env(version: str, *, mode: str = "clone", extra: Sequence[str] = (),
               python: str | None = None, status=print) -> Path:
    """Create (or reuse) an environment with ``mujoco==version``; return its python."""
    python = python or f"{sys.version_info.major}.{sys.version_info.minor}"
    requirements = frozen_requirements() if mode == "clone" else []
    requirements += list(extra)
    key = hashlib.sha1(json.dumps([mode, python, sorted(requirements),
                                   __version__]).encode()).hexdigest()[:10]
    path = cache_dir() / "venvs" / f"mujoco-{version}-{mode}-py{python}-{key}"
    exe = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (path / ".simblame-ok").exists() and exe.exists():
        return exe
    status(f"creating {mode} environment for mujoco {version} (python {python}) ...")
    shutil.rmtree(path, ignore_errors=True)
    _run([_uv(), "venv", "-q", "--python", python, str(path)], "create the virtualenv")
    _install(exe, version, requirements)
    _link_simblame(exe)
    (path / ".simblame-ok").write_text(version)
    return exe


def _install(exe: Path, version: str, requirements: list) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        req, override = Path(tmp) / "requirements.txt", Path(tmp) / "override.txt"
        req.write_text("\n".join(requirements + [f"mujoco=={version}"]) + "\n")
        override.write_text(f"mujoco=={version}\n")
        cmd = [_uv(), "pip", "install", "-q", "--python", str(exe), "-r", str(req),
               "--overrides", str(override)]
        tags = {m.group(1).lower() for r in requirements if (m := _LOCAL_TAG.match(r))}
        for tag in sorted(tags):
            cmd += ["--extra-index-url", TORCH_INDEX.format(tag=tag)]
        if tags:
            cmd += ["--index-strategy", "unsafe-best-match"]
        _run(cmd, f"install mujoco {version} with your packages",
             hint="Some package may not support this MuJoCo/Python combination. Try "
                  "`--fresh --with <your-env-package>` to install only what the harness needs.")


def _link_simblame(exe: Path) -> None:
    """Make the running simblame importable in the new environment via a .pth file."""
    shim = cache_dir() / "src" / hashlib.sha1(str(PACKAGE_DIR).encode()).hexdigest()[:10]
    shim.mkdir(parents=True, exist_ok=True)
    link = shim / "simblame"
    if not link.exists():
        link.symlink_to(PACKAGE_DIR, target_is_directory=True)
    site = subprocess.run([str(exe), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (Path(site) / "_simblame_shim.pth").write_text(str(shim) + "\n")


def _uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise VersionEnvError("`simblame versions` needs uv to build per-version environments: "
                                "pip install uv  (https://docs.astral.sh/uv/)")
    return uv


def _run(cmd: list, what: str, hint: str = "") -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        raise VersionEnvError(f"could not {what}:\n{tail}" + (f"\n\n{hint}" if hint else ""))
