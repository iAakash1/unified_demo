"""Tomato Keras CNNs (project 01): custom CNN, MobileNetV2, EfficientNetB3, ConvNeXt-Tiny. Runs in the `keras` env.

Uses projects/01_cnn_keras_tomato/predict.py itself: its model table, class lists and the preprocessing each model
was trained with (32x32 crop/blur/sharpen for the custom CNN; keras.applications preprocess_input at 224x224).
"""
from __future__ import annotations

import os

from _repo import load_script, project, require, top_k

HEAVY = False
_P = load_script(project("01_cnn_keras_tomato") / "predict.py", "p01_predict")


def load(args):
    import keras
    path, family = _P.MODELS[args["model"]]
    full = os.path.join(_P.HERE, path)
    require([full])
    return keras.models.load_model(full, compile=False), family


def predict(handle, image_path, args):
    import cv2
    model, family = handle
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("OpenCV could not read the image")
    x = _P.preprocess_cnn(img) if family is None else _P.preprocess_transfer(img, family)
    probs = model.predict(x[None], verbose=0)[0]
    classes = _P.CNN_CLASSES if family is None else _P.TOMATO_CLASSES
    top = top_k(classes, probs)
    return {"label": top[0][0], "score": top[0][1], "score_kind": "softmax probability", "top": top,
            "device": "CPU"}
