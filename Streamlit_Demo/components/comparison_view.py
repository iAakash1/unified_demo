"""Side-by-side answers of several models on the same photo."""
from __future__ import annotations

import streamlit as st

from config.model_registry import BY_ID


def render(rows: list[dict]) -> None:
    table = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Scores are not comparable across models: a CNN softmax/sigmoid probability, the TIH model's mean "
               "token probability and a generated VLM description measure different things and none is "
               "calibrated against the others. The recorded results come from each model's own evaluation set.")
    for r in rows:
        res = r["_result"]
        if res.get("ok") and res.get("text"):
            with st.expander(f"Generated text - {BY_ID[res['model_id']]['name']}"):
                st.code(res["text"], language=None, wrap_lines=True)
