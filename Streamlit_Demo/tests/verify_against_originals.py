"""Check that the app's adapters give the same answers as the repository's own predict scripts.

For every model the app can run on this machine, the same image goes through
  (a) the app path: worker process + adapter (what the Streamlit page uses), and
  (b) the original command from the handover repository, run with its own environment,
and the predicted label and score (or generated text) are compared. Results go to
Demo_Assets/verification/verification_results.{json,md}.

    <any python with streamlit deps>  tests/verify_against_originals.py [--only id1,id2] [--skip-originals]

Images: one tomato test image bundled in Demo_Assets (PlantVillage), and SoyNet test images read from the local
SoyNet folder (SOYNET_DIR; not redistributed, not copied).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from config import settings  # noqa: E402
from config.model_registry import BY_ID, MODELS  # noqa: E402
from core import inference_engine  # noqa: E402
from core.model_loader import WorkerPool  # noqa: E402
from utils.device_detection import availability  # noqa: E402

REPO = settings.REPO
TOMATO = settings.SAMPLES_DIR / "tomato" / ("late_blight__01425d17-4c97-46e3-b395-c1453b78ab78___GHLB2_Leaf_9100.JPG")
SOY_DIR = Path(os.environ.get("SOYNET_DIR", r"C:\Users\Admin\Desktop\latest\SoyNet\Preprocessing_SoyNet_Data\Camera Clicks_256_256"))
SOY = [SOY_DIR / "Healthy_Preprocessing data" / "soy (23).JPG", SOY_DIR / "Disease_Preprocessing data" / "Soy  (218).JPG"]
OUT = settings.DEMO_ROOT / "Demo_Assets" / "verification"

KERAS_FLAG = {"cnn": "cnn", "mobilenet": "mobilenet", "efficientnet": "efficientnet", "convnext": "convnext"}
VLM_FLAG = {"blip2_march", "blip2_june", "llava", "qwen_april", "plantdx", "plantvlm", "qwen_base", "florence"}


def py(env):
    return str(settings.ENV_PYTHON[env])


def original_cmd(m, image):
    a = m["args"]
    if m["adapter"] == "tomato.keras_cnn":
        return [py("keras"), "predict.py", "--image", image, "--model", a["model"]], REPO / "projects/01_cnn_keras_tomato"
    if m["adapter"] == "tomato.torch_convnext":
        return [py("vlm"), "predict.py", "--image", image], REPO / "projects/02_cnn_pytorch_convnext"
    if m["adapter"] == "soybean.soynet" and not a["model"].startswith("keras"):
        return ([py("soynet"), "predict_soynet.py", "--image", image, "--model", a["model"]],
                REPO / "projects/03_soybean_soynet")
    if m["adapter"] == "tomato.vlm_plantvillage":
        return [py(m["env"]), "vlm_predict.py", "--model", a["model"], "--image", image], REPO / "evaluation"
    if m["adapter"] == "tomato.tih_qlora":
        return ([py("tih"), "-m", "plantdx.cli", "company", "predict", "--config",
                 "configs/train/qwen25vl_tih_tomato_moreaccurate.yaml", "--adapter",
                 "checkpoints/tih_tomato_moreaccurate", "--image", image, "--json"],
                REPO / "projects/10_tih_tomato_qwen_qlora")
    return None, None


def run_original(m, image):
    cmd, cwd = original_cmd(m, image)
    if cmd is None:
        return {"available": False, "note": "no single-image script in the repository for this model"}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"}
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    out = {"available": True, "returncode": p.returncode, "seconds": round(time.perf_counter() - t, 1),
           "command": " ".join(Path(c).name if i == 0 else c for i, c in enumerate(cmd)).replace(str(REPO), "<repo>")}
    text = p.stdout.strip()
    if m["adapter"] == "tomato.tih_qlora":
        try:
            r = json.loads(text[text.index("["):])[0]
            out.update(label=r["label"], score=r["confidence"], text=r["generated"])
        except Exception:
            out["raw"] = text[-500:]
    elif m["adapter"] == "tomato.vlm_plantvillage":
        out["text"] = text  # vlm_predict.py prints only the generated diagnosis on stdout
    else:
        rows = [re.match(r"\s*(\S+)\s+([0-9.]+)\s*$", l) for l in text.splitlines()]
        rows = [r for r in rows if r]
        if rows:
            if m["adapter"] == "soybean.soynet":
                rows.sort(key=lambda r: -float(r.group(2)))
            out.update(label=rows[0].group(1), score=float(rows[0].group(2)))
    if p.returncode != 0:
        out["stderr_tail"] = p.stderr[-800:].replace(str(REPO), "<repo>")
    return out


def compare(app, orig):
    if not orig.get("available"):
        return "no original script (app result only)"
    if "text" in orig and orig.get("label") is None:
        return "MATCH" if app.get("text", "").strip() == orig["text"].strip() else "DIFFERENT TEXT"
    if app.get("label") != orig.get("label"):
        return "DIFFERENT LABEL"
    if app.get("score") is not None and orig.get("score") is not None and abs(app["score"] - orig["score"]) > 2e-3:
        return f"SAME LABEL, score differs by {abs(app['score'] - orig['score']):.4f}"
    if "text" in orig and orig.get("text") and app.get("text", "").strip() != orig["text"].strip():
        return "SAME LABEL, text differs"
    return "MATCH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-originals", action="store_true")
    a = ap.parse_args()
    only = [s for s in a.only.split(",") if s]
    pool = WorkerPool()
    results = []
    for m in MODELS:
        if only and m["id"] not in only:
            continue
        state, reason = availability(m)
        images = [str(p) for p in SOY] if m["crop"] == "Soybean" else [str(TOMATO)]
        if m["crop"] == "Mango":
            images = [str(next((settings.SAMPLES_DIR / "mango").glob("anthracnose__*")))]
        for image in images:
            row = {"model_id": m["id"], "model": m["name"], "image": Path(image).name, "availability": state}
            if state != "ready":
                row.update(status="NOT RUN", reason=reason)
                results.append(row)
                print(f"{m['id']:<28} NOT RUN  {reason}", flush=True)
                continue
            app = inference_engine.run(pool, m["id"], image)
            row["app"] = {k: app.get(k) for k in ("ok", "label", "score", "score_kind", "text", "error",
                                                  "infer_seconds", "load_seconds", "device")}
            if m.get("gpu"):  # free the GPU before the original script loads its own copy of the model
                pool.get(m["env"]).request({"op": "unload"})
            orig = {"available": False, "note": "skipped"} if a.skip_originals else run_original(m, image)
            row["original"] = orig
            row["status"] = ("APP ERROR: " + str(app.get("error"))) if not app["ok"] else compare(app, orig)
            results.append(row)
            print(f"{m['id']:<28} {row['status']:<40} app={app.get('label')} {app.get('score')} "
                  f"orig={orig.get('label')} {orig.get('score')}", flush=True)
    pool.stop_all()
    OUT.mkdir(parents=True, exist_ok=True)
    name = "verification_results" + ("_subset" if only else "")
    (OUT / f"{name}.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("written", OUT / f"{name}.json")


if __name__ == "__main__":
    main()
