"""Helpers shared by the adapters (they run inside a worker, in a handover environment)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("HANDOVER_ROOT", r"D:\handover"))


def project(name: str) -> Path:
    return REPO / "projects" / name


def load_script(path: Path, module_name: str):
    """Import a repository script (e.g. projects/01_.../predict.py) by path under a unique module name."""
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def require(paths) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("weights not downloaded: " + "; ".join(
            str(Path(m).relative_to(REPO)) if str(m).startswith(str(REPO)) else m for m in missing)
            + " - run: python setup/download_models.py")


def top_k(classes, probs, k=3):
    order = sorted(range(len(probs)), key=lambda i: -float(probs[i]))[:k]
    return [[classes[i], round(float(probs[i]), 4)] for i in order]
