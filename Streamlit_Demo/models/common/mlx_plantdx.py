"""Knowledge-grounded PlantDx on Apple MLX (project 09), tomato and mango. Runs only on macOS with Apple Silicon,
in the project-09 environment (mlx 0.31.2, mlx-vlm 0.6.4); point MLX_PYTHON at that environment's python.

The same calls as `plantdx infer --adapter checkpoints/<run> --image <photo>` (plantdx.training.inference:
load_model + caption_image, greedy, 128 new tokens) followed by the evaluator's disease reading
(plantdx.evaluation.classification: build_lexicon + extract_disease_id).

NOT EXECUTED on the Windows workstation where this demo was built: MLX does not exist for Windows/CUDA. The code
follows the project-09 API but has not been run.
"""
from __future__ import annotations

import os
import sys

from _repo import project, require

HEAVY = True
P09 = project("09_plantdx_mlx_knowledge_grounded")
BASE = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
INSTRUCTION = {"tomato": "Describe the visible condition of this tomato leaf.",
               "mango": "Describe the visible condition of this mango leaf."}


def load(args):
    if sys.platform != "darwin":
        raise RuntimeError("MLX models run only on macOS with Apple Silicon")
    adapter = P09 / "checkpoints" / args["run"]
    require([adapter / "adapters.safetensors"])
    os.chdir(P09)
    from plantdx.evaluation.classification import build_lexicon
    from plantdx.training.inference import load_model
    return {"loaded": load_model(BASE, adapter_path=str(adapter)), "lexicon": build_lexicon(args["crop"])}


def predict(handle, image_path, args):
    from plantdx.evaluation.classification import extract_disease_id
    from plantdx.training.inference import caption_image
    r = caption_image(handle["loaded"], image_path, instruction=INSTRUCTION[args["crop"]], max_tokens=128,
                      temperature=0.0)
    disease = extract_disease_id(r.caption, handle["lexicon"])
    return {"label": disease, "text": r.caption, "score": None, "score_kind": "none (generated text)",
            "device": "Apple Silicon (MLX)", "parsed": {"disease_id": disease}}


def unload(handle):
    handle.clear()
