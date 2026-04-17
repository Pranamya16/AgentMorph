"""Colab bootstrap for AgentMorph Stage 1.

Run the whole file in a single Colab cell:

    !curl -sL <raw-url>/notebooks/colab_setup.py | python -

Or, if you've already cloned the repo:

    !python notebooks/colab_setup.py

It:
  1. Ensures we're on a T4 (warns otherwise).
  2. Mounts Google Drive at /content/drive so the HF model cache and the
     `runs/` directory survive session kills.
  3. Installs agentmorph and the framework + model extras.
  4. Runs a dry-run smoke test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


DRIVE_MOUNT = Path("/content/drive")
PROJECT_ROOT = Path(os.environ.get("AGENTMORPH_ROOT", "/content/AgentMorph"))
HF_CACHE = Path("/content/drive/MyDrive/AgentMorph/hf_cache")
RUNS_DIR = Path("/content/drive/MyDrive/AgentMorph/runs")


def _run(cmd: list[str], *, check: bool = True) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=check)


def _check_gpu() -> None:
    try:
        import torch
    except ImportError:
        print("torch not yet installed; will install below.")
        return
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible. All 5 primary models need a T4 or better.")
        return
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {name} ({vram:.1f} GB)")
    if "T4" not in name:
        print(f"NOTE: expected a T4, got {name!r}. Should still work at 4-bit.")


def _mount_drive() -> None:
    if DRIVE_MOUNT.exists() and any(DRIVE_MOUNT.iterdir()):
        print("Drive already mounted.")
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        print("Not running on Colab — skipping Drive mount.")
        return
    drive.mount(str(DRIVE_MOUNT))
    print(f"Drive mounted at {DRIVE_MOUNT}")


def _install() -> None:
    if shutil.which("pip") is None:
        raise RuntimeError("pip not found")
    _run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"])
    _run([
        sys.executable, "-m", "pip", "install", "-q",
        "-e", f"{PROJECT_ROOT}[models,smolagents,langgraph,agentdojo]",
    ], check=False)
    # bitsandbytes sometimes needs a CUDA-matched wheel; reinstall if quant fails.
    # (leave to the user if the first install works)


def _prepare_dirs() -> None:
    for p in (HF_CACHE, RUNS_DIR):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTMORPH_HF_CACHE"] = str(HF_CACHE)
    print(f"HF cache: {HF_CACHE}")
    print(f"Runs dir: {RUNS_DIR}")


def _smoketest() -> None:
    # Use the dry-run path so we don't burn a model load here.
    _run([
        sys.executable, "-m", "agentmorph.runner",
        "--model", "Llama-3.2-3B",
        "--framework", "native",
        "--env", "ecommerce",
        "--n-scenarios", "3",
        "--dry-run",
        "--out-dir", str(RUNS_DIR / "stage1_baseline"),
    ])


def main() -> None:
    print("=== AgentMorph Colab bootstrap ===")
    _mount_drive()
    _prepare_dirs()
    _install()
    _check_gpu()
    _smoketest()
    print()
    print("Ready. Kick off the full baseline sweep with:")
    print("    !python -m agentmorph.runner --out-dir", RUNS_DIR / "stage1_baseline")


if __name__ == "__main__":
    main()
