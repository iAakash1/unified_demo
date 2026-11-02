"""PlantVillage vision-language models: BLIP-2 (Mar, Jun), LLaVA-1.6, Qwen2.5-VL April, PlantDx (May), PlantVLM,
and the Qwen2.5-VL / Florence-2 baselines without fine-tuning. Runs in the `vlm` env (Florence-2: `florence` env).

Uses evaluation/vlm_predict.py itself: its model table (adapter folders, instructions), load() and generate()
with the same generation settings as the 200-image evaluation. The condition is then read from the generated
text with the rule of evaluation/vlm_eval.py (extract_condition, copied below because vlm_eval.py reads
command-line arguments when imported).
"""
from __future__ import annotations

import gc
import re

from _repo import REPO, load_script

HEAVY = True
_P = load_script(REPO / "evaluation" / "vlm_predict.py", "eval_vlm_predict")

# evaluation/vlm_eval.py, CONDITION_LABELS and extract_condition (unchanged)
CONDITION_LABELS = [
    "Healthy", "Late Blight", "Early Blight", "Bacterial Spot",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mite Damage",
    "Target Spot", "Mosaic Virus", "Yellow Leaf Curl Virus",
]


def extract_condition(text):
    text_l = text.lower()
    m = re.search(r"condition[:\s]+([^.]+)", text_l)
    if m:
        cand = m.group(1).strip().title()
        for label in CONDITION_LABELS:
            if label.lower() in cand.lower() or cand.lower() in label.lower():
                return label
    for label in CONDITION_LABELS:
        if label.lower() in text_l:
            return label
    return "Unknown"


def load(args):
    import os
    kind, adapter, instruction = _P.MODELS[args["model"]]
    if adapter and not os.path.isdir(adapter):
        raise FileNotFoundError("adapter not downloaded - run: python setup/download_models.py")
    model, proc = _P.load(kind, adapter)
    return {"kind": kind, "model": model, "proc": proc, "instruction": instruction}


def predict(handle, image_path, args):
    from PIL import Image
    text = _P.generate(handle["kind"], handle["model"], handle["proc"], Image.open(image_path).convert("RGB"),
                       handle["instruction"])
    plant = re.search(r"plant[:\s]+([a-z ,()-]+?)[.\n]", text.lower())
    out = {"text": text, "score": None, "score_kind": "none (generated text)", "device": "CUDA"}
    if args["model"] == "plantvlm":
        out["label"] = "(free text; no class parsing)"
    else:
        out["label"] = extract_condition(text)
        out["parsed"] = {"condition": out["label"], "plant": plant.group(1).strip().title() if plant else None}
    return out


def unload(handle):
    import torch
    handle.clear()
    gc.collect()
    torch.cuda.empty_cache()
