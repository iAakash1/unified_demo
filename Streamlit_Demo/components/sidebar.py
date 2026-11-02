"""Sidebar: crop, model, mode, and what this machine can run."""
from __future__ import annotations

import streamlit as st

from config import settings
from config.model_registry import BY_ID, CROPS, models_for
from utils.device_detection import availability, env_installed, host

ICON = {"ready": "🟢", "restricted": "🟠", "missing": "⚪"}


def render(pool, defaults: dict) -> dict:
    sb = st.sidebar
    sb.markdown("### 🌿 TIH-IoT crop disease demo")
    crop = sb.selectbox("Crop", CROPS, index=CROPS.index(defaults.get("crop", "Tomato")))
    mode = sb.radio("Inference mode", ["Single model", "Compare models"],
                    index=1 if defaults.get("mode") == "compare" else 0, horizontal=True)

    entries = models_for(crop)
    labels = {m["id"]: f"{ICON[availability(m)[0]]} {m['name']}" for m in entries}
    ids = [m["id"] for m in entries]
    default_id = defaults.get("model") if defaults.get("model") in ids else ids[0]
    selected = []
    if mode == "Single model":
        model_id = sb.selectbox("Model", ids, index=ids.index(default_id), format_func=labels.get)
        state, reason = availability(BY_ID[model_id])
        (sb.success if state == "ready" else sb.warning)(reason if state != "ready" else "Ready on this machine")
    else:
        model_id = None
        ready = [i for i in ids if availability(BY_ID[i])[0] == "ready"]
        pre = [i for i in defaults.get("models", []) if i in ready] or ready[:3]
        selected = sb.multiselect("Models to compare (runnable here)", ready, default=pre, format_func=labels.get)
        sb.caption("Models that cannot run on this machine are left out of the list; see the status below.")

    sb.caption("🟢 runs here · 🟠 needs other hardware · ⚪ environment or weights missing")

    with sb.expander("Machine and environment status", expanded=False):
        h = host()
        st.markdown(f"**OS:** {h['os']} ({h['machine']})  \n"
                    f"**NVIDIA GPU:** {h['nvidia_gpu'] or 'none'}  \n"
                    f"**Apple Silicon (MLX):** {'yes' if h['apple_silicon'] else 'no'}  \n"
                    f"**Handover repository:** {'found' if h['repo'] else 'NOT FOUND - set HANDOVER_ROOT'}")
        rows = []
        for env in ("keras", "soynet", "vlm", "florence", "tih"):
            rows.append({"environment": env, "installed": "yes" if env_installed(env) else "no",
                         "worker": pool.status().get(env, "not started")})
        st.dataframe(rows, hide_index=True, use_container_width=True)
        if st.button("Stop all model workers (free memory)"):
            pool.stop_all()
            st.toast("All workers stopped")

    sb.markdown("---")
    sb.caption(settings.DISCLAIMER)
    return {"crop": crop, "mode": mode, "model_id": model_id, "compare_ids": selected}
