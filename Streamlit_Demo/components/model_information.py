"""Model card: what the model is, what it was trained on, where its code and weights are."""
from __future__ import annotations

import streamlit as st

from config import settings
from utils.device_detection import availability


def _size(rel: str) -> str:
    p = settings.REPO / rel
    return f"{p.stat().st_size / 1e6:,.1f} MB" if p.exists() else "not downloaded"


def render(model: dict) -> None:
    state, reason = availability(model)
    c1, c2 = st.columns(2)
    c1.markdown(f"**Architecture:** {model['architecture']}  \n"
                f"**Type:** {model['family']}  \n**Period:** {model['period']}  \n"
                f"**Output:** {model['output']}")
    c2.markdown(f"**Dataset:** {model['dataset']}  \n**Runtime:** `{model['env']}` environment  \n"
                f"**Status here:** {reason}  \n**Source:** `{model['source']}`")
    if model["weights"]:
        st.markdown("**Weights** (downloaded by `python setup/download_models.py` from github.com/iAakash1/TIH-IoT):")
        st.dataframe([{"file": w, "size": _size(w)} for w in model["weights"]], hide_index=True,
                     use_container_width=True)
    else:
        st.markdown("**Weights:** base model only (downloaded from Hugging Face on first use).")
    st.info(f"**Limitations:** {model['limitations']}")
