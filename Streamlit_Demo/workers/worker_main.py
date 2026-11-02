"""Long-lived inference worker, started by the app with one of the handover clone's environments.

Protocol: one JSON request per line on stdin, one JSON reply per line on the original stdout.
    {"op": "predict", "adapter": "tomato.keras_cnn", "args": {...}, "image": "<path>"}
    {"op": "unload"}   free every loaded model (and the GPU memory)
    {"op": "info"}     python version, platform, installed frameworks
    {"op": "quit"}
Everything libraries print goes to stderr (the app writes it to logs/), so stdout carries only replies.
Adapters are loaded by file path, so this worker never imports the app's own packages: the projects' modules
(e.g. project 03's `config`) keep their names.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
sys.path.append(str(MODELS_DIR))  # for models/_repo.py; appended so the projects' own modules win

# keep the protocol channel, then send fd 1 (C-level prints included) to stderr
_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

_adapters: dict[str, object] = {}
_loaded: dict[str, tuple[object, object]] = {}  # cache key -> (adapter module, handle)


def _adapter(name: str):
    if name not in _adapters:
        crop, mod = name.split(".")
        path = MODELS_DIR / crop / f"{mod}.py"
        spec = importlib.util.spec_from_file_location(f"demo_adapter_{crop}_{mod}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _adapters[name] = module
    return _adapters[name]


def _unload_all() -> None:
    for module, handle in _loaded.values():
        unload = getattr(module, "unload", None)
        if unload:
            unload(handle)
    _loaded.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _predict(req: dict) -> dict:
    module = _adapter(req["adapter"])
    args = req.get("args", {})
    key = req["adapter"] + json.dumps(args, sort_keys=True)
    load_s = 0.0
    if key not in _loaded:
        if getattr(module, "HEAVY", False):  # one large model per worker
            _unload_all()
        t = time.perf_counter()
        _loaded[key] = (module, module.load(args))
        load_s = time.perf_counter() - t
    _, handle = _loaded[key]
    t = time.perf_counter()
    result = module.predict(handle, req["image"], args)
    result["infer_seconds"] = round(time.perf_counter() - t, 3)
    result["load_seconds"] = round(load_s, 2)
    result["cached"] = load_s == 0.0
    return result


def _info() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for name in ("torch", "tensorflow", "keras", "transformers", "peft", "bitsandbytes", "mlx_vlm"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "quit":
                _proto.write(json.dumps({"ok": True}) + "\n")
                break
            if op == "unload":
                _unload_all()
                reply = {"ok": True}
            elif op == "info":
                reply = {"ok": True, "result": _info()}
            elif op == "predict":
                reply = {"ok": True, "result": _predict(req)}
            else:
                reply = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:  # report, keep serving
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _proto.write(json.dumps(reply, default=str) + "\n")


if __name__ == "__main__":
    main()
