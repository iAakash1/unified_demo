"""Recorded results, always with the evaluation set they belong to."""
from __future__ import annotations

import streamlit as st

from config.model_registry import models_for
from utils.device_detection import availability


def _f1(v):
    return "n/a" if v is None else f"{v:.4f}"


def model_metrics(model: dict) -> None:
    rows = [{"Result": label, "Accuracy": f"{acc * 100:.2f}%", "Macro-F1": _f1(f1), "Evaluation set": split,
             "n": f"{n:,}", "Note": note, "Report (in the handover repo)": rep}
            for label, acc, f1, split, n, rep, note in model["metrics"]]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def crop_overview(crop: str) -> None:
    rows = []
    for m in models_for(crop):
        for label, acc, f1, split, n, rep, note in m["metrics"]:
            rows.append({"Model": m["name"], "Result": label, "Accuracy": f"{acc * 100:.2f}%", "Macro-F1": _f1(f1),
                         "Evaluation set": split, "n": n, "Runs here": availability(m)[0]})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("Each row is measured on its own evaluation set: rows with different evaluation sets are not a "
               "ranking. The PlantVillage VLM rows are validation results; the CNN test splits, the MLX test "
               "splits and the TIH test set are different images.")
