"""Paths and runtime settings for the demo app.

The app does not copy any model. It runs the models inside a clone of github.com/iAakash1/handover, using that
clone's own environments (.venvs/<env>) and downloaded weights. Point HANDOVER_ROOT at the clone if it is not at
one of the default locations.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
DEMO_ROOT = APP_DIR.parent  # Demo_TIH_Handover


def _find_repo() -> Path:
    candidates = [os.environ.get("HANDOVER_ROOT"), r"D:\handover", str(DEMO_ROOT.parent / "handover"),
                  str(Path.home() / "handover")]
    for c in candidates:
        if c and (Path(c) / "projects").is_dir() and (Path(c) / "setup" / "models_manifest.json").is_file():
            return Path(c).resolve()
    return Path(candidates[1] or ".")


REPO = _find_repo()
SAMPLES_DIR = DEMO_ROOT / "Demo_Assets" / "sample_images"
LOG_DIR = APP_DIR / "logs"
UPLOAD_DIR = APP_DIR / ".uploads"
WORKER_SCRIPT = APP_DIR / "workers" / "worker_main.py"

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
BIN = "Scripts" if IS_WINDOWS else "bin"

# Python interpreter of each runtime. The Windows/Linux ones are the clone's .venvs built by setup/setup_envs.py.
# MLX models need the Mac environment of project 09 (conda env "vlm" in its docs); set MLX_PYTHON to its python.
ENV_PYTHON = {
    env: REPO / ".venvs" / env / BIN / ("python.exe" if IS_WINDOWS else "python")
    for env in ("keras", "soynet", "vlm", "florence", "tih")
}
ENV_PYTHON["mlx"] = Path(os.environ.get("MLX_PYTHON", sys.executable if IS_MAC else "mlx-python-not-configured"))

# Environments whose models run on the NVIDIA GPU; the app keeps at most one large model loaded across them.
GPU_ENVS = {"vlm", "florence", "tih"}
# The tomato Keras worker runs on the CPU (TensorFlow 2.20 has no GPU build for Windows). The SoyNet worker
# (TensorFlow 2.10) uses the GPU with memory growth: its Keras models were saved with the mixed_float16 policy and
# crash in TensorFlow's CPU kernels on this machine; on the GPU they run as they were trained.
CPU_ONLY_ENVS = {"keras"}
TF_GROWTH_ENVS = {"soynet"}

MAX_UPLOAD_MB = 20
ALLOWED_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
REQUEST_TIMEOUT_S = 1200  # first load of a 7B model from disk can take several minutes
TIH_LOW_CONFIDENCE = 0.55  # same default as the project-10 app

DISCLAIMER = ("Predictions are experimental decision support from research models trained mostly on lab "
              "photographs (PlantVillage, MangoLeafBD, SoyNet) or a private TIH dataset. They are not a "
              "definitive agricultural diagnosis; confirm with an expert before acting.")
