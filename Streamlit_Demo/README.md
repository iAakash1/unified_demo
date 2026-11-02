# TIH-IoT Crop Disease Detection and Vision-Language Model Demonstration

One Streamlit application for every crop-disease model of the Chanakya Fellowship
(github.com/iAakash1/handover). Pick a crop and a model, upload a leaf photo (or use a bundled test image), and run
the model. Comparison mode runs several models on the same photo.

The models are **not re-implemented**. Each runs in its own environment from the handover clone, through the
project's own loading, preprocessing and generation code. On the TIH workstation, every runnable model that has an original
single-image script gives the same answer through the app as through that script (see *Verification*). The two
Keras SoyNet models have no such script.

## Quick start (TIH workstation)

The handover repository is at `D:\handover`, with its environments (`.venvs\`) and weights already in place.

```powershell
cd C:\Users\Admin\Desktop\Demo_TIH_Handover\Streamlit_Demo
D:\handover\.venvs\vlm\Scripts\python -m streamlit run app.py
```

Open http://localhost:8765 (port and theme are set in `.streamlit/config.toml`). Stop with Ctrl+C; the model
workers stop with the app.

## Another machine

```powershell
git clone https://github.com/iAakash1/handover.git
cd handover
python setup\setup_envs.py          # builds .venvs\keras, soynet, vlm, florence, tih (Windows, Python 3.10)
python setup\download_models.py     # final weights from github.com/iAakash1/TIH-IoT, SHA-256 checked
$env:HANDOVER_ROOT = "<path to the clone>"
<path to the clone>\.venvs\vlm\Scripts\python -m streamlit run <this folder>\app.py
```

- **Host process:** the app itself needs only `streamlit` and `pillow` (`requirements.txt`). The clone's `vlm`
  environment has both.
- **Model packages:** the models run in the clone's environments, so their packages (TensorFlow, PyTorch,
  transformers, PEFT, bitsandbytes) come from those, not from `requirements.txt`.
- **Base models:** downloaded from Hugging Face on first use: Qwen2.5-VL-7B ~16 GB, LLaVA-1.6 ~15 GB,
  BLIP-2 ~14 GB, Florence-2 ~1 GB.
- **MLX models (Mac with Apple Silicon only):**
  1. Build project 09's environment
     (`projects/09_plantdx_mlx_knowledge_grounded/HANDOVER_GUIDE.md`).
  2. `export MLX_PYTHON=<that env>/bin/python`, then start the app on the Mac.

  The MLX adapter (`models/common/mlx_plantdx.py`) follows the project-09 API, but **it has not been run**: it
  was built on Windows, where MLX does not exist.

## Hardware

| Models | Needs |
|---|---|
| Tomato Keras CNNs | CPU |
| ConvNeXt-Tiny (PyTorch), SoyNet | NVIDIA GPU if present (CPU works for the PyTorch models) |
| BLIP-2 | NVIDIA GPU, ~8 GB |
| LLaVA-1.6, Qwen2.5-VL April, PlantDx, PlantVLM, Qwen baseline | NVIDIA GPU, ~16–17 GB |
| TIH 11-class QLoRA | NVIDIA CUDA GPU, ~10–11 GB (bitsandbytes 4-bit NF4) |
| MLX tomato and mango | macOS on Apple Silicon (~12 GB unified memory) |

The app keeps **at most one large GPU model loaded** across its workers. Before loading another one, it asks the
other workers to free theirs. "Stop all model workers" in the sidebar frees everything.

## How it works

```
app.py                        page layout, deep links, session history
config/settings.py            repository location, environment interpreters, limits, disclaimer
config/model_registry.py      every model: crop, runtime, adapter, weight files, recorded metrics (with report path)
core/model_loader.py          one persistent worker process per environment; GPU budget; timeouts
core/inference_engine.py      availability check -> worker -> standard result
core/image_preprocessing.py   stores the validated upload unchanged (model preprocessing happens in the adapters)
core/prediction_formatter.py  readable class names
core/model_comparison.py      several models on one photo
components/                   sidebar, prediction view, comparison view, model card, metrics tables
models/tomato/keras_cnn.py         -> projects/01_cnn_keras_tomato/predict.py (preprocessing, class lists)
models/tomato/torch_convnext.py    -> projects/02_cnn_pytorch_convnext/predict.py
models/tomato/vlm_plantvillage.py  -> evaluation/vlm_predict.py (load, generate) + vlm_eval.py condition rule
models/tomato/tih_qlora.py         -> plantdx.company.predict (as `plantdx.cli company predict`)
models/common/mlx_plantdx.py       -> plantdx.training.inference (as `plantdx infer`), Mac only
models/soybean/soynet.py           -> predict_soynet.py steps (PyTorch); data.preprocessing.load_image (Keras)
workers/worker_main.py        the worker: JSON requests on stdin, replies on stdout, library output to logs/
utils/                        device and weight detection, upload validation, path-redacting logging
tests/verify_against_originals.py   app vs original scripts, writes Demo_Assets/verification/
tests/capture_screenshots.py        real screenshots of the running app (headless Edge)
tests/build_recorded_predictions.py recorded MLX answers for the bundled test images
```

**Deep links** (useful in a live demo):
- `?crop=Tomato&model=tomato_qwen_april&sample=target_spot&run=1` opens a model on a sample and runs it.
- `?crop=Tomato&mode=compare&models=tomato_convnext_torch,tomato_convnext_keras&sample=late_blight&run=1` runs
  a comparison.

## Model registry (status on the TIH workstation)

| Crop | Model | Runtime | Status here |
|---|---|---|---|
| Tomato | Custom CNN, MobileNetV2, EfficientNetB3, ConvNeXt-Tiny (Keras) | keras, CPU | runs; matches `predict.py` |
| Tomato | ConvNeXt-Tiny (PyTorch) | vlm | runs; matches `predict.py` |
| Tomato | BLIP-2 March, BLIP-2 June, LLaVA-1.6, Qwen2.5-VL April, PlantDx, PlantVLM | vlm, GPU | runs; identical text to `vlm_predict.py` |
| Tomato | Qwen2.5-VL-7B and Florence-2 without fine-tuning (baselines) | vlm / florence, GPU | runs; identical text to `vlm_predict.py` |
| Tomato | TIH 11-class Qwen2.5-VL QLoRA | tih, GPU | runs; matches `plantdx.cli company predict` |
| Tomato | Knowledge-grounded MLX QLoRA | MLX | **restricted:** needs macOS on Apple Silicon. The app shows the model's recorded answers for the bundled test images |
| Mango | Knowledge-grounded MLX QLoRA | MLX | **restricted:** as above |
| Soybean | ResNet50, MobileNetV2, EfficientNetB0, EfficientNetB3 (PyTorch) | soynet | runs; matches `predict_soynet.py` |
| Soybean | ResNet50, MobileNetV2 (Keras) | soynet, GPU | runs. No single-image script exists in the repository to compare against; uses the project's own `load_image` |

**Inputs and outputs:**
- **Bundled samples:** none for soybean, because SoyNet images are not redistributed. Upload a soybean photo
  instead.
- **SoyNet Keras on the GPU:** these models were saved with the `mixed_float16` policy and crashed in
  TensorFlow's CPU kernels on this machine, so the SoyNet worker uses the GPU (with memory growth).
- **Scores:**

  | Model type | Score shown |
  |---|---|
  | CNNs | softmax probability (SoyNet: sigmoid probability) |
  | TIH model | mean probability of its generated tokens |
  | Other VLMs | no score: they generate text, and the condition is parsed from the text |

  These scores are **not comparable across models**.

## Verification

`tests/verify_against_originals.py` runs each runnable model on the same image twice: through the app's worker
path, and through the original command from the repository. It then compares the outputs.
Results: `../Demo_Assets/verification/verification_results.json` and `VERIFICATION.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "handover repository not found" | set `HANDOVER_ROOT` to the clone |
| "environment 'x' not installed" | `python setup\setup_envs.py` in the clone |
| "weights not downloaded" | `python setup\download_models.py` (or with the project filter shown in the model card) |
| first VLM prediction takes minutes | the base model loads from disk (or downloads from Hugging Face the first time) |
| CUDA out of memory | "Stop all model workers" in the sidebar, then run the model again |
| a worker stopped | see `logs/worker_<env>.log`; the next request starts it again |
| MLX model "needs macOS on Apple Silicon" | expected on Windows/Linux; run the app on the Mac with `MLX_PYTHON` set |

Logs are in `logs/`:
- `app.log` replaces the repository path, the user's home folder and anything that looks like a token with
  placeholders.
- `worker_<env>.log` holds the libraries' raw console output; it stays on this machine.

*Predictions are experimental decision support from research models, not a definitive agricultural diagnosis.*
