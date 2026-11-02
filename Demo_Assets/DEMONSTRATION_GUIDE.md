# Demonstration guide (about 20 minutes)

App: http://localhost:8765. Start it as in `README.md`, and warm up the models first (see "Before the demo" there).
Each step has a deep link that opens the page with the right crop, model and image. Add `&run=1` to start the
prediction immediately. The load times below were measured on the TIH workstation; once a model is loaded, its
later predictions take only the inference time.

| # | Step | Deep link (append to http://localhost:8765/) | Time | What to point out |
|---|---|---|---|---|
| 1 | Overview | `?crop=Tomato` | 1 min | crop → model selection; 🟢/🟠/⚪ status per model; sidebar "Machine and environment status"; the disclaimer |
| 2 | Best tomato CNN | `?crop=Tomato&model=tomato_convnext_torch&sample=late_blight&run=1` | 1 min (load ~2 s, inference <0.1 s) | top-3 softmax probabilities; recorded result 99.92% on the 2,411-image test split |
| 3 | Five CNNs on one image | `?crop=Tomato&mode=compare&models=tomato_custom_cnn,tomato_mobilenetv2_keras,tomato_efficientnetb3_keras,tomato_convnext_keras,tomato_convnext_torch&sample=early_blight&run=1` | 2 min | same prediction from all five; each recorded result belongs to its own evaluation set ("Tomato results overview" tab) |
| 4 | Structured diagnosis (VLM) | `?crop=Tomato&model=tomato_qwen_april&sample=target_spot&run=1` | 2 min (load ~20 s from cache, inference ~11 s) | plant, condition, severity, pathogen and symptoms in text; 98.50% is a *validation* result on PlantVillage |
| 5 | Why fine-tuning matters | `?crop=Tomato&mode=compare&models=tomato_qwen_april,tomato_qwen_base,tomato_llava16&sample=late_blight&run=1` | 3 min | the base Qwen and the LLaVA adapter do not name the disease; only one large model is held in GPU memory at a time, so this takes ~2 min |
| 6 | TIH 11-class model (ICAIN 2026) | `?crop=Tomato&model=tomato_tih_qlora&sample=late_blight&run=1` | 2 min (load ~45 s, inference ~4 s) | 4-bit NF4 on CUDA; the score is the mean token probability, with 0.55 as the low-confidence threshold; 92.76% internal, 69.32% external. The bundled image is a PlantVillage leaf, outside the TIH domain. For TIH photos, use the project app (`python run_app.py 1`); the data is private |
| 7 | MLX tomato (Mac only) | `?crop=Tomato&model=tomato_mlx_kg&sample=healthy` | 2 min | marked "needs macOS on Apple Silicon"; the recorded July and November answers for this exact test image; the "Model information" tab separates July (93.7%, weights lost) from the November retrain (90.9%, served) |
| 8 | MLX mango | `?crop=Mango&model=mango_mlx_kg&sample=sooty_mould` | 1 min | July got this image right and November did not: the runs differ, which is why they are reported separately (July 82.0%, November 86.0%) |
| 9 | Soybean | `?crop=Soybean&mode=compare&models=soy_resnet50_keras,soy_efficientnetb3_torch,soy_resnet50_torch` | 2 min | upload a SoyNet photo from `C:\Users\Admin\Desktop\latest\SoyNet\Preprocessing_SoyNet_Data\Camera Clicks_256_256\`; healthy vs diseased only; sigmoid probability |
| 10 | Wrap-up | "Session history" tab | 1 min | everything run in this session; limitations: lab images, domain shift, hardware |

**If something is slow or fails:** use the screenshots in `screenshots/` and the files in `recorded_predictions/`.
Never present a screenshot or recorded answer as a live run.
