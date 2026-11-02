# Demo assets: TIH-IoT crop disease models

Supplementary material for the live demonstration. The primary documentation is
`../TIH_IoT_Cropwise_Modelwise_Handover_Report.docx` (and the matching PDF).

| Folder / file | What it is |
|---|---|
| `../Streamlit_Demo/` | the unified demo application (its README has installation details) |
| `DEMONSTRATION_GUIDE.md` | recommended crop-by-crop demonstration sequence with timings |
| `MODEL_INVENTORY.csv` | every model, crop, runtime, status on the TIH workstation, recorded results, report and weight paths |
| `sample_images/tomato`, `sample_images/mango` | one held-out **test-split** image per class from the MLX runs (PlantVillage, MangoLeafBD; both public; source path per image in `samples.json`) |
| `screenshots/` | real screenshots of the running application on the TIH workstation |
| `recorded_predictions/` | genuine prediction files written by the evaluation runs (see its README) |
| `verification/` | app vs original scripts on the same images (`VERIFICATION.md`, `verification_results.json`) |

No soybean sample is bundled: SoyNet images are not redistributed. On the TIH workstation they are in
`C:\Users\Admin\Desktop\latest\SoyNet\Preprocessing_SoyNet_Data\Camera Clicks_256_256\`. No TIH image is bundled
either: that dataset is private.

## Launch (TIH workstation)

```powershell
cd C:\Users\Admin\Desktop\Demo_TIH_Handover\Streamlit_Demo
D:\handover\.venvs\vlm\Scripts\python -m streamlit run app.py
```

Then open http://localhost:8765. The repository's own per-project apps are started with `python run_app.py <n>` in
`D:\handover` (1 = TIH app, 2 = crop disease app with CLIP check and GradCAM, 4 = PlantDx).

## Hardware requirements

| Models | Needs |
|---|---|
| Keras CNNs (tomato) | CPU |
| ConvNeXt (PyTorch), SoyNet | NVIDIA GPU if present (CPU possible for the PyTorch models) |
| BLIP-2 / LLaVA / Qwen2.5-VL / PlantDx / PlantVLM | NVIDIA GPU, 8–17 GB; base models are cached on the TIH workstation |
| TIH 11-class QLoRA | NVIDIA CUDA GPU, ~10–11 GB |
| MLX tomato and mango | macOS on Apple Silicon only |

## Before the demo

1. Start the app 10 minutes early.
2. Run one CNN and the Qwen April model once, so the first 1–3 minute model load does not happen in front of the
   audience.
3. Keep the TIH app (`python run_app.py 1`) ready in a second browser tab.

## Troubleshooting

| Symptom | Fix |
|---|---|
| a VLM takes minutes | first load of a 7B model; later predictions take 2–12 s |
| "CUDA out of memory" | sidebar → "Stop all model workers", then run again |
| MLX model "needs macOS on Apple Silicon" | expected on Windows: show the recorded answers instead, or run the app on the Mac (`MLX_PYTHON`) |
| "weights not downloaded" | `python setup\download_models.py` in `D:\handover` |
| page does not respond | stop the app (Ctrl+C) and start it again; workers restart automatically |
| a model errors | `Streamlit_Demo/logs/app.log` and `logs/worker_<env>.log` |

*Predictions are experimental decision support from research models, not a definitive agricultural diagnosis.*
