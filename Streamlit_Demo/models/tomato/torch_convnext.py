"""Tomato ConvNeXt-Tiny in PyTorch (project 02). Runs in the `vlm` env.

Uses projects/02_cnn_pytorch_convnext/predict.py: its load_model(), validation transforms and class order.
"""
from __future__ import annotations

from _repo import load_script, project, require, top_k

HEAVY = False
_P = load_script(project("02_cnn_pytorch_convnext") / "predict.py", "p02_predict")


def load(args):
    import torch
    require([project("02_cnn_pytorch_convnext") / "models" / "convnext_tomato_best.pth"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _P.load_model(device), device


def predict(handle, image_path, args):
    import torch
    from PIL import Image
    model, device = handle
    x = _P.TF(Image.open(image_path).convert("RGB"))[None].to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu().tolist()
    top = top_k(_P.CLASSES, probs)
    return {"label": top[0][0], "score": top[0][1], "score_kind": "softmax probability", "top": top,
            "device": device.upper()}


def unload(handle):
    pass
