"""What this machine can run: OS, Apple Silicon, NVIDIA GPU, installed environments, downloaded weights."""
from __future__ import annotations

import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from config import settings


@lru_cache(maxsize=1)
def host() -> dict:
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            gpu = out.splitlines()[0] if out else None
        except Exception:
            gpu = None
    return {"os": platform.system(), "machine": platform.machine(), "nvidia_gpu": gpu,
            "apple_silicon": settings.IS_MAC and platform.machine() == "arm64",
            "repo": settings.REPO.is_dir(), "repo_path": str(settings.REPO)}


def env_installed(env: str) -> bool:
    return Path(settings.ENV_PYTHON[env]).exists()


def missing_weights(model: dict) -> list[str]:
    return [w for w in model["weights"] if not (settings.REPO / w).exists()]


def availability(model: dict) -> tuple[str, str]:
    """('ready' | 'restricted' | 'missing', reason)."""
    h = host()
    if not h["repo"]:
        return "missing", "handover repository not found (set HANDOVER_ROOT)"
    if model.get("requires") == "mac" and not h["apple_silicon"]:
        return "restricted", "needs macOS on Apple Silicon (MLX); not available on this machine"
    if model.get("gpu") and not h["nvidia_gpu"]:
        return "restricted", "needs an NVIDIA CUDA GPU"
    if model["env"] != "mlx" and not env_installed(model["env"]):
        return "missing", f"environment '{model['env']}' not installed (python setup/setup_envs.py)"
    if model["env"] == "mlx" and not Path(settings.ENV_PYTHON["mlx"]).exists():
        return "missing", "set MLX_PYTHON to the project-09 environment's python"
    miss = missing_weights(model)
    if miss:
        return "missing", "weights not downloaded (python setup/download_models.py)"
    return "ready", "ready"
