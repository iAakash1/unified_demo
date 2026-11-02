"""TIH-IoT crop disease detection and vision-language model demonstration.

One page for every model of github.com/iAakash1/handover: pick a crop and a model, upload a leaf photo (or use a
bundled sample), and the model runs in its own environment through a worker process (see core/model_loader.py).

    <python with streamlit> -m streamlit run app.py

Deep links (used for the demonstration script and screenshots):
    ?crop=Tomato&model=tomato_convnext_torch&sample=late_blight&run=1
    ?crop=Tomato&mode=compare&models=tomato_convnext_torch,tomato_convnext_keras&sample=late_blight&run=1
"""
from __future__ import annotations

import atexit
import json
import time
from pathlib import Path

import streamlit as st

from components import comparison_view, metrics_dashboard, model_information, prediction_view, sidebar
from config import settings
from config.model_registry import BY_ID
from core import inference_engine, model_comparison
from core.image_preprocessing import store_upload
from core.model_loader import WorkerPool
from utils.device_detection import availability
from utils.validation import validate_image

st.set_page_config(page_title="TIH-IoT crop disease demo", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(90deg, #14532d 0%, #166534 55%, #15803d 100%); color: #f0fdf4;
       padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 0.8rem;}
.hero h1 {color: #f0fdf4; font-size: 1.55rem; margin: 0 0 .25rem 0;}
.hero p {margin: 0; opacity: .9; font-size: .95rem;}
.pred-card {border: 1px solid #bbf7d0; background: #f0fdf4; border-radius: 12px; padding: .9rem 1.1rem;
            margin-bottom: .6rem;}
.pred-model {font-size: .82rem; color: #166534; text-transform: uppercase; letter-spacing: .04em;}
.pred-label {font-size: 1.7rem; font-weight: 700; color: #14532d;}
.pred-meta {font-size: .82rem; color: #4b5563;}
@media (prefers-color-scheme: dark) {
  .pred-card {background: #052e16; border-color: #166534;}
  .pred-label {color: #bbf7d0;} .pred-model {color: #86efac;} .pred-meta {color: #d1d5db;}
}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_pool() -> WorkerPool:
    pool = WorkerPool()
    atexit.register(pool.stop_all)
    return pool


pool = get_pool()
RECORDED = settings.DEMO_ROOT / "Demo_Assets" / "recorded_predictions" / "mlx_recorded_answers_for_samples.json"


def recorded_mlx_answers(sample_name: str) -> None:
    """Genuine answers of the MLX models for this test image, recorded on the Mac during evaluation."""
    if not RECORDED.exists():
        return
    entry = json.loads(RECORDED.read_text(encoding="utf-8")).get(sample_name)
    if not entry:
        return
    st.markdown("**Recorded answers for this test image** - genuine outputs from the Mac evaluation runs "
                "(not run live on this machine)")
    for run, r in entry["runs"].items():
        ok = "✅" if r["predicted_disease"] == r["true_disease"] else "❌"
        with st.container(border=True):
            st.markdown(f"**{run}** · parsed disease `{r['predicted_disease']}` {ok} (true: `{r['true_disease']}`)")
            st.markdown(f"*Instruction:* {r['instruction']}  \n*Answer:* {r['answer']}")
            st.caption(f"Source: {r['source']} in the handover repository")


qp = st.query_params
defaults = {"crop": qp.get("crop", "Tomato"), "model": qp.get("model"), "mode": qp.get("mode"),
            "models": [m for m in qp.get("models", "").split(",") if m]}
choice = sidebar.render(pool, defaults)
crop = choice["crop"]

st.markdown("<div class='hero'><h1>TIH-IoT Crop Disease Detection and Vision-Language Model Demonstration</h1>"
            "<p>Chanakya Fellowship, TIH-IoT IIT Bombay · Dec 2025 - Sep 2026 · every model runs from "
            "github.com/iAakash1/handover with its own environment, preprocessing and weights</p></div>",
            unsafe_allow_html=True)
st.caption("⚠️ " + settings.DISCLAIMER)

if "history" not in st.session_state:
    st.session_state.history = []

tab_run, tab_info, tab_overview, tab_history = st.tabs(
    ["🔬 Run prediction", "📄 Model information", f"📊 {crop} results overview", "🕘 Session history"])

# ------------------------------------------------------------------ image source
with tab_run:
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        samples = sorted((settings.SAMPLES_DIR / crop.lower()).glob("*")) if (settings.SAMPLES_DIR / crop.lower()).is_dir() else []
        samples = [s for s in samples if s.suffix.lower() in (".jpg", ".jpeg", ".png")]
        source = st.radio("Image", ["Upload a photo", "Use a bundled sample"], horizontal=True,
                          index=1 if qp.get("sample") and samples else 0)
        image_path, image_label = None, None
        if source == "Upload a photo":
            up = st.file_uploader("Leaf photo", type=settings.ALLOWED_TYPES)
            if up is not None:
                data = up.getvalue()
                ok, msg = validate_image(up.name, data)
                if ok:
                    image_path, image_label = store_upload(up.name, data), up.name
                    st.caption(f"Accepted: {msg}")
                else:
                    st.error(f"Rejected: {msg}")
        elif samples:
            names = [s.name for s in samples]
            want = next((i for i, n in enumerate(names) if n.startswith(qp.get("sample", "") + "__")), 0)
            pick = st.selectbox("Sample (MLX test split, from the handover repository)", names, index=want,
                                format_func=lambda n: n.split("__")[0].replace("_", " "))
            image_path, image_label = settings.SAMPLES_DIR / crop.lower() / pick, pick.split("__")[0]
        else:
            st.info("No bundled sample for this crop (SoyNet images are not redistributed). Upload a photo, "
                    "e.g. from the SoyNet folder on the TIH workstation.")
        if image_path:
            st.image(str(image_path), caption=image_label, use_container_width=True)

    with right:
        auto = qp.get("run") == "1" and not st.session_state.get("_autorun_done")
        if choice["mode"] == "Single model":
            model = BY_ID[choice["model_id"]]
            st.subheader(model["name"])
            st.caption(f"{model['family']} · {model_comparison.headline_metric(model)}")
            state, why = availability(model)
            go = st.button("▶ Run prediction", type="primary", disabled=image_path is None or state != "ready")
            if state != "ready":
                st.warning(f"This model cannot run here: {why}.")
            if (go or auto) and image_path:
                st.session_state._autorun_done = True
                note = " (a 7B vision-language model needs 1-3 minutes to load the first time)" if model.get("gpu") else ""
                with st.spinner(f"Running {model['name']}{note}..."):
                    result = inference_engine.run(pool, model["id"], str(image_path))
                st.session_state.last = (model["id"], result)
                st.session_state.history.insert(0, {"time": time.strftime("%H:%M:%S"), "crop": crop,
                                                    "model": model["name"], "image": image_label,
                                                    "prediction": result.get("display") if result["ok"] else "error",
                                                    "score": result.get("score"),
                                                    "score type": result.get("score_kind", ""),
                                                    "seconds": result.get("infer_seconds")})
            last = st.session_state.get("last")
            if last and last[0] == model["id"]:
                prediction_view.render(last[1], model)
            if model["env"] == "mlx" and image_path and Path(image_path).parent.parent == settings.SAMPLES_DIR:
                recorded_mlx_answers(Path(image_path).name)
        else:
            ids = choice["compare_ids"]
            st.subheader(f"Compare {len(ids)} models")
            go = st.button("▶ Run all selected models", type="primary", disabled=not ids or image_path is None)
            if (go or auto) and ids and image_path:
                st.session_state._autorun_done = True
                bar = st.progress(0.0, text="starting")
                rows = model_comparison.compare(
                    pool, ids, str(image_path),
                    progress=lambda i, n, name: bar.progress(i / n, text=f"{i + 1}/{n}: {name}"))
                bar.progress(1.0, text="done")
                st.session_state.compare = rows
                for r in rows:
                    res = r["_result"]
                    st.session_state.history.insert(0, {"time": time.strftime("%H:%M:%S"), "crop": crop,
                                                        "model": r["Model"], "image": image_label,
                                                        "prediction": r["Prediction"], "score": res.get("score"),
                                                        "score type": res.get("score_kind", ""),
                                                        "seconds": res.get("infer_seconds")})
    if choice["mode"] != "Single model" and st.session_state.get("compare"):
        st.markdown("#### Comparison on this image")
        comparison_view.render(st.session_state.compare)

with tab_info:
    if choice["mode"] == "Single model":
        m = BY_ID[choice["model_id"]]
        st.subheader(m["name"])
        model_information.render(m)
        st.markdown("**Recorded results**")
        metrics_dashboard.model_metrics(m)
    else:
        for mid in choice["compare_ids"]:
            with st.expander(BY_ID[mid]["name"], expanded=False):
                model_information.render(BY_ID[mid])
                metrics_dashboard.model_metrics(BY_ID[mid])

with tab_overview:
    metrics_dashboard.crop_overview(crop)

with tab_history:
    if st.session_state.history:
        st.dataframe(st.session_state.history, hide_index=True, use_container_width=True)
    else:
        st.caption("No predictions yet in this session.")
