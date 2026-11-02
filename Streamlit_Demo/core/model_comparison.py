"""Run several models on the same image and lay the answers side by side."""
from __future__ import annotations

from config.model_registry import BY_ID
from core import inference_engine
from core.prediction_formatter import score_text


def headline_metric(model: dict) -> str:
    label, acc, f1, split, n, _, _ = model["metrics"][model.get("headline", 0)]
    f1s = f"{f1:.4f}" if f1 is not None else "n/a"
    return f"{acc * 100:.2f}% acc, macro-F1 {f1s} ({label}, n={n:,})"


def compare(pool, model_ids: list[str], image_path: str, progress=None) -> list[dict]:
    rows = []
    for i, mid in enumerate(model_ids):
        if progress:
            progress(i, len(model_ids), BY_ID[mid]["name"])
        r = inference_engine.run(pool, mid, image_path)
        rows.append({
            "Model": BY_ID[mid]["name"],
            "Type": BY_ID[mid]["family"],
            "Prediction": r.get("display", "-") if r["ok"] else "not run",
            "Raw output label": r.get("label", "-") if r["ok"] else r.get("error", ""),
            "Score": score_text(r) if r["ok"] else "-",
            "Score type": r.get("score_kind", "-") if r["ok"] else "-",
            "Inference s": r.get("infer_seconds", "-") if r["ok"] else "-",
            "Load s": r.get("load_seconds", "-") if r["ok"] else "-",
            "Recorded result (own evaluation set)": headline_metric(BY_ID[mid]),
            "_result": r,
        })
    return rows
