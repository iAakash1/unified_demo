"""Run one registered model on one stored image through its worker, and record the outcome."""
from __future__ import annotations

import time

from config.model_registry import BY_ID
from core.model_loader import WorkerError, WorkerPool
from core.prediction_formatter import readable
from utils.device_detection import availability
from utils.logging_utils import get_logger

log = get_logger("inference")


def run(pool: WorkerPool, model_id: str, image_path: str) -> dict:
    model = BY_ID[model_id]
    state, reason = availability(model)
    out = {"model_id": model_id, "model": model["name"], "crop": model["crop"], "ok": False}
    if state != "ready":
        out["error"] = reason
        return out
    heavy = bool(model.get("gpu"))
    started = time.perf_counter()
    try:
        result = pool.predict(model["env"], model["adapter"], model["args"], str(image_path), heavy)
        out.update(result)
        out["ok"] = True
        out["display"] = readable(result.get("label"))
    except WorkerError as exc:
        out["error"] = str(exc)
    except Exception as exc:  # never crash the page
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["wall_seconds"] = round(time.perf_counter() - started, 2)
    log.info("%s -> %s (%s s)%s", model_id, out.get("label"), out["wall_seconds"],
             "" if out["ok"] else f" ERROR {out.get('error')}")
    return out
