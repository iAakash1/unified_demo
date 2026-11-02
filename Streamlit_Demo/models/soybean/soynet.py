"""SoyNet healthy vs diseased (project 03). Runs in the `soynet` env (TensorFlow 2.10 + PyTorch 2.5.1), on the GPU when present.

PyTorch models: the same steps as projects/03_soybean_soynet/predict_soynet.py (ModelFactory, model.pth,
get_transforms(train=False), one sigmoid logit where index 1 = Healthy).
Keras models (the 95.16% ResNet50 and MobileNetV2 of the Keras run): tf.keras load_model of the saved .keras file
and the project's own single-image loader, data.preprocessing.load_image (256x256 + the backbone's
preprocess_input), sigmoid output = P(Healthy).
"""
from __future__ import annotations

import os
import sys

from _repo import project, require

HEAVY = False
P03 = project("03_soybean_soynet")
if str(P03) not in sys.path:
    sys.path.insert(0, str(P03))
CLASSES = ["Diseased", "Healthy"]
KERAS = {"keras_resnet50": ("resnet50", "results/experiments/resnet50/soynet_resnet50_best.keras"),
         "keras_mobilenetv2": ("mobilenetv2", "results/experiments/mobilenetv2/soynet_mobilenetv2_best.keras")}


def load(args):
    name = args["model"]
    if name in KERAS:
        import tensorflow as tf
        arch, rel = KERAS[name]
        require([P03 / rel])
        return ("keras", arch, tf.keras.models.load_model(str(P03 / rel), compile=False))
    import torch
    from pytorch.models import ModelFactory
    require([P03 / "results" / "experiments" / name / "model.pth"])
    model = ModelFactory.create_model(name, pretrained=False, freeze=False)
    ckpt = torch.load(os.path.join(P03, "results", "experiments", name, "model.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"  # as predict_soynet.py
    return ("torch", name, model.to(device).eval())


def predict(handle, image_path, args):
    kind, arch, model = handle
    if kind == "keras":
        import numpy as np
        from data.preprocessing import load_image
        x = load_image(image_path, model_name=arch)
        x = x[None] if x.ndim == 3 else x
        p_healthy = float(np.asarray(model.predict(x, verbose=0)).reshape(-1)[0])
    else:
        import torch
        from PIL import Image
        from pytorch.dataset import get_transforms
        x = get_transforms(train=False)(Image.open(image_path).convert("RGB"))[None].to(next(model.parameters()).device)
        with torch.no_grad():
            p_healthy = torch.sigmoid(model(x).reshape(-1)[0]).item()
    probs = [1 - p_healthy, p_healthy]
    label = CLASSES[int(p_healthy >= 0.5)]
    return {"label": label, "score": round(max(probs), 4), "score_kind": "sigmoid probability",
            "top": [[c, round(p, 4)] for c, p in sorted(zip(CLASSES, probs), key=lambda t: -t[1])],
            "device": _device(kind, model)}


def _device(kind, model):
    if kind == "torch":
        return str(next(model.parameters()).device).upper()
    import tensorflow as tf
    return "CUDA" if tf.config.list_physical_devices("GPU") else "CPU"
