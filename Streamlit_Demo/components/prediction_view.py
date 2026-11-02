"""Show one prediction honestly: label, the kind of score the model really gives, and its raw output."""
from __future__ import annotations

import streamlit as st

from config import settings
from core.prediction_formatter import readable


def render(result: dict, model: dict) -> None:
    if not result["ok"]:
        st.error(f"**{model['name']}** could not run: {result.get('error')}")
        return
    label = result.get("display") or readable(result.get("label"))
    score = result.get("score")
    st.markdown(f"<div class='pred-card'><div class='pred-model'>{model['name']}</div>"
                f"<div class='pred-label'>{label}</div>"
                f"<div class='pred-meta'>raw output label: <code>{result.get('label')}</code></div></div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", "-" if score is None else f"{score:.3f}", help=result.get("score_kind"))
    c2.metric("Inference time", f"{result.get('infer_seconds', 0):.2f} s")
    c3.metric("Model load", "cached" if result.get("cached") else f"{result.get('load_seconds', 0):.1f} s")
    st.caption(f"Score type: {result.get('score_kind')} · device: {result.get('device', '-')}")

    if model["id"] == "tomato_tih_qlora" and score is not None and score < settings.TIH_LOW_CONFIDENCE:
        st.warning(f"Low confidence (below {settings.TIH_LOW_CONFIDENCE}, the project-10 app's default threshold).")
    if model["id"] == "tomato_tih_qlora":
        st.caption("This model was trained on private TIH field photographs (11 classes, leaves and fruit); "
                   "PlantVillage-style photos are outside its training domain.")
    if result.get("top"):
        st.markdown("**Top classes**")
        st.dataframe([{"class": readable(c), "raw label": c, "probability": p} for c, p in result["top"]],
                     hide_index=True, use_container_width=True)
    if result.get("text"):
        st.markdown("**Generated text**")
        st.code(result["text"], language=None, wrap_lines=True)
    if result.get("parsed"):
        st.caption("Parsed from the text: " + ", ".join(f"{k} = {v}" for k, v in result["parsed"].items()
                                                         if k != "generated"))
