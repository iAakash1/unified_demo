"""TIH 11-class tomato Qwen2.5-VL QLoRA (project 10, ICAIN 2026). Runs in the `tih` env (NVIDIA GPU required).

The same calls as `plantdx.cli company predict --config <cfg> --adapter checkpoints/<run> --image <photo>`:
load_company_train_config + load_company_dataset (class profile), load_predictor (Qwen2.5-VL-7B-Instruct in 4-bit
NF4 with the PEFT adapter) and predict_image (532 px resize, class-name prompt, greedy decoding, parse_prediction,
mean-token-probability confidence).
"""
from __future__ import annotations

import gc
import os

from _repo import project, require

HEAVY = True
P10 = project("10_tih_tomato_qwen_qlora")
CONFIGS = {"tih_tomato_moreaccurate": "configs/train/qwen25vl_tih_tomato_moreaccurate.yaml",
           "tih_tomato_qwen_qlora_strong": "configs/train/qwen25vl_tih_tomato_strong.yaml"}


def load(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("the TIH model needs an NVIDIA CUDA GPU (bitsandbytes 4-bit NF4)")
    run = args.get("run", "tih_tomato_moreaccurate")
    adapter = P10 / "checkpoints" / run
    require([adapter / "adapter_model.safetensors"])
    os.chdir(P10)  # the configs and class profile use paths relative to the project folder
    from plantdx.company.classes import load_company_dataset
    from plantdx.company.config import load_company_train_config
    from plantdx.company.predict import load_predictor
    cfg = load_company_train_config(CONFIGS[run])
    dataset = load_company_dataset(cfg.data.profile)
    loaded = load_predictor(cfg, adapter_dir=str(adapter))
    return {"cfg": cfg, "dataset": dataset, "loaded": loaded}


def predict(handle, image_path, args):
    from plantdx.company.predict import predict_image
    r = predict_image(handle["loaded"], image_path, cfg=handle["cfg"], dataset=handle["dataset"])
    return {"label": r.label, "class_id": r.class_id, "text": r.generated, "score": r.confidence,
            "score_kind": "mean probability of the generated tokens", "device": "CUDA (4-bit NF4)",
            "parsed": {"class": r.label, "generated": r.generated}}


def unload(handle):
    import torch
    handle.clear()
    gc.collect()
    torch.cuda.empty_cache()
