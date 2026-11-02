"""Every model of the handover repository the demo knows about, crop by crop.

Each entry says which runtime runs it, which adapter (models/<crop>/...) loads it, which weight files it needs
(paths relative to the handover clone) and which recorded results belong to it. Metrics are copied from the
evaluation report named next to them; nothing is computed here. Different evaluation sets are never merged.
"""
from __future__ import annotations

# metric tuple: (label, accuracy, macro-F1 or None, split description, sample count, report path, note)
P01 = "projects/01_cnn_keras_tomato"
P03 = "projects/03_soybean_soynet"
P04 = "projects/04_llava_blip2"
P09 = "projects/09_plantdx_mlx_knowledge_grounded"
P10 = "projects/10_tih_tomato_qwen_qlora"

PV200 = ("PlantVillage validation records 1-200 (155 tomato, 45 potato/pepper; 15-class data, scored on "
         "10 condition labels)")

MODELS: list[dict] = [
    # ------------------------------------------------------------------ TOMATO: CNNs
    {
        "id": "tomato_custom_cnn", "crop": "Tomato", "name": "Custom CNN (Keras)", "family": "CNN classifier",
        "period": "Dec 2025", "env": "keras", "adapter": "tomato.keras_cnn", "args": {"model": "cnn"},
        "weights": [f"{P01}/model/best_model_cnn.keras"],
        "architecture": "3 x (Conv-BatchNorm-MaxPool, 32/64/128 filters) + Dense 128 + Dropout 0.5, 32x32 input",
        "dataset": "PlantVillage, all 15 folders (16 outputs: the Mac '.DS_Store' entry is output 0)",
        "output": "class probabilities (softmax)",
        "metrics": [("Tomato test split", 0.9851, 0.9830, "70/15/15 tomato test split; overlaps the CNN's own "
                     "training data, indicative only", 2411,
                     "evaluation/keras/custom_cnn_best_tomato_test/evaluation_report.json", ""),
                    ("All 15 PlantVillage classes", 0.8677, 0.8508, "all PlantVillage images", 20638,
                     "evaluation/keras/custom_cnn_best_plantvillage15/evaluation_report.json", "")],
        "source": f"{P01}/model/train_cnn.py; {P01}/predict.py",
        "limitations": "32x32 input; evaluated on images that overlap its training data; can output potato or "
                       "pepper classes.",
    },
    {
        "id": "tomato_mobilenetv2_keras", "crop": "Tomato", "name": "MobileNetV2 (Keras)", "family": "CNN classifier",
        "period": "Jan 2026", "env": "keras", "adapter": "tomato.keras_cnn", "args": {"model": "mobilenet"},
        "weights": [f"{P01}/model/best_model_mobilenet.keras"],
        "architecture": "ImageNet MobileNetV2 + classifier head, 224x224, two-stage fine-tuning",
        "dataset": "PlantVillage tomato, 10 classes", "output": "class probabilities (softmax)",
        "metrics": [("Validation", 0.9090, 0.8939, "Keras validation_split 0.2", 3198,
                     "evaluation/keras/mobilenetv2_keras/evaluation_report.json", "")],
        "source": f"{P01}/model/train_mobilenet.py; {P01}/predict.py",
        "limitations": "Lab images only; validation result (no separate test split).",
    },
    {
        "id": "tomato_efficientnetb3_keras", "crop": "Tomato", "name": "EfficientNetB3 (Keras)",
        "family": "CNN classifier", "period": "Jan 2026", "env": "keras", "adapter": "tomato.keras_cnn",
        "args": {"model": "efficientnet"}, "weights": [f"{P01}/model/best_model_efficientNetB3.keras"],
        "architecture": "ImageNet EfficientNetB3 + classifier head, 224x224, two-stage fine-tuning",
        "dataset": "PlantVillage tomato, 10 classes", "output": "class probabilities (softmax)",
        "metrics": [("Validation", 0.9359, 0.9275, "Keras validation_split 0.2", 3198,
                     "evaluation/keras/efficientnetb3_keras/evaluation_report.json", "")],
        "source": f"{P01}/model/train_transfer.py; {P01}/predict.py",
        "limitations": "Lab images only; validation result.",
    },
    {
        "id": "tomato_convnext_keras", "crop": "Tomato", "name": "ConvNeXt-Tiny (Keras)", "family": "CNN classifier",
        "period": "Jan 2026", "env": "keras", "adapter": "tomato.keras_cnn", "args": {"model": "convnext"},
        "weights": [f"{P01}/model/best_model_convnext.keras"],
        "architecture": "ImageNet ConvNeXt-Tiny + Dense 256 (GELU) + Dropout 0.4, 224x224, two-stage fine-tuning",
        "dataset": "PlantVillage tomato, 10 classes", "output": "class probabilities (softmax)",
        "metrics": [("Validation", 0.9615, 0.9507, "Keras validation_split 0.2", 3198,
                     "evaluation/keras/convnext_tiny_keras/evaluation_report.json", "")],
        "source": f"{P01}/model/train_convnext.py; {P01}/predict.py",
        "limitations": "Lab images only; validation result; ~100 ms per image on CPU.",
    },
    {
        "id": "tomato_convnext_torch", "crop": "Tomato", "name": "ConvNeXt-Tiny (PyTorch)", "family": "CNN classifier",
        "period": "Jan 2026", "env": "vlm", "adapter": "tomato.torch_convnext", "args": {},
        "weights": ["projects/02_cnn_pytorch_convnext/models/convnext_tomato_best.pth"],
        "architecture": "torchvision ConvNeXt-Tiny, full fine-tuning, 224x224", "gpu": False,
        "dataset": "PlantVillage tomato, 10 classes", "output": "class probabilities (softmax)",
        "metrics": [("Test", 0.9992, 0.9991, "70/15/15 per-class split, test part", 2411,
                     "evaluation/convnext_torch/evaluation_report.json", "validation 99.79% (2,397)")],
        "source": "projects/02_cnn_pytorch_convnext/predict.py; notebooks/02_training_convnext.ipynb",
        "limitations": "PlantVillage lab images; near-perfect scores reflect an easy benchmark.",
    },
    # ------------------------------------------------------------------ TOMATO: VLMs on PlantVillage
    {
        "id": "tomato_blip2_march", "crop": "Tomato", "name": "BLIP-2 OPT-2.7B + LoRA (March)",
        "family": "Vision-language (structured caption)", "period": "Mar 2026", "env": "vlm",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "blip2_march"},
        "weights": [f"{P04}/outputs/blip2_plantvillage/best/adapter_model.safetensors"],
        "architecture": "Salesforce/blip2-opt-2.7b (bf16) + LoRA r32 / alpha 64", "gpu": True,
        "dataset": "PlantVillage 15 classes, template captions", "output": "generated diagnosis text; condition "
        "parsed from the text",
        "metrics": [("Validation (condition)", 0.8900, 0.8928, PV200, 200,
                     "evaluation/blip2_march/evaluation_report.json", "")],
        "source": f"{P04}/blip2_march/; evaluation/vlm_predict.py",
        "limitations": "Short captions; 21 of 200 answers named no condition; needs an NVIDIA GPU (~8 GB).",
    },
    {
        "id": "tomato_blip2_june", "crop": "Tomato", "name": "BLIP-2 OPT-2.7B + LoRA (June re-run)",
        "family": "Vision-language (structured caption)", "period": "Jun 2026", "env": "vlm",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "blip2_june"},
        "weights": [f"{P04}/blip2_rerun_june/outputs/blip2_plantvillage/best/adapter_model.safetensors"],
        "architecture": "Salesforce/blip2-opt-2.7b (bf16) + LoRA r32 / alpha 64, 1 epoch", "gpu": True,
        "dataset": "PlantVillage 15 classes, template captions", "output": "generated text; parsed condition",
        "metrics": [("Validation (condition)", 0.8900, 0.8948, PV200, 200,
                     "evaluation/blip2_june/evaluation_report.json", "its own June script reported 0% (scoring bug)")],
        "source": f"{P04}/blip2_rerun_june/; evaluation/vlm_predict.py",
        "limitations": "As March; one epoch of training.",
    },
    {
        "id": "tomato_llava16", "crop": "Tomato", "name": "LLaVA-1.6-Mistral-7B + LoRA",
        "family": "Vision-language (did not learn the task)", "period": "Mar 2026", "env": "vlm",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "llava"},
        "weights": [f"{P04}/outputs/llava_plantvillage/best/adapter_model.safetensors"],
        "architecture": "llava-hf/llava-v1.6-mistral-7b-hf (bf16) + LoRA r64 / alpha 128", "gpu": True,
        "dataset": "PlantVillage 15 classes, template captions", "output": "generated text; parsed condition",
        "metrics": [("Validation (condition)", 0.0200, 0.0151, PV200, 200,
                     "evaluation/llava_v16/evaluation_report.json", "178 of 200 answers named no condition")],
        "source": "failed_experiments/02_llava16_mistral_lora/; evaluation/vlm_predict.py",
        "limitations": "Answers generically; kept for comparison only. ~16 GB GPU memory.",
    },
    {
        "id": "tomato_qwen_april", "crop": "Tomato", "name": "Qwen2.5-VL-7B + LoRA, PlantVillage (April)",
        "family": "Vision-language (structured diagnosis)", "period": "Apr 2026", "env": "vlm",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "qwen_april"},
        "weights": ["projects/06_qwen25vl_plantvillage/qwen/outputs/qwen25vl_plantvillage/best/adapter_model.safetensors"],
        "architecture": "Qwen/Qwen2.5-VL-7B-Instruct (bf16) + LoRA r32 / alpha 64", "gpu": True,
        "dataset": "PlantVillage 15 classes (18,574 / 2,064 caption records)",
        "output": "generated diagnosis (plant, condition, severity, pathogen, symptoms); parsed condition",
        "metrics": [("Validation (condition)", 0.9850, 0.9827, PV200, 200,
                     "projects/06_qwen25vl_plantvillage/qwen/evaluation_report.json",
                     "adapter chosen on validation loss; no separate test split")],
        "source": "projects/06_qwen25vl_plantvillage/qwen/; evaluation/vlm_predict.py",
        "limitations": "Validation result on lab images; ~17 GB GPU memory; ~1 min first load.",
    },
    {
        "id": "tomato_plantdx", "crop": "Tomato", "name": "PlantDx Qwen2.5-VL-7B + LoRA (May)",
        "family": "Vision-language (structured diagnosis)", "period": "May 2026", "env": "vlm",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "plantdx"},
        "weights": ["projects/07_plantdx_plantvillage/outputs/checkpoints/best/adapter_model.safetensors"],
        "architecture": "Qwen/Qwen2.5-VL-7B-Instruct (bf16) + LoRA r32 / alpha 64", "gpu": True,
        "dataset": "PlantVillage 15 classes", "output": "generated diagnosis; parsed condition",
        "metrics": [("Validation (condition)", 0.9800, 0.9777, "PlantDx validation records 1-200", 200,
                     "projects/07_plantdx_plantvillage/outputs/evaluations/report.json", "")],
        "source": "projects/07_plantdx_plantvillage/; evaluation/vlm_predict.py",
        "limitations": "Validation result on lab images; ~17 GB GPU memory.",
    },
    {
        "id": "tomato_plantvlm", "crop": "Tomato", "name": "PlantVLM expert-caption Qwen2.5-VL + LoRA",
        "family": "Vision-language (exploratory)", "period": "Aug 2026", "env": "vlm",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "plantvlm"},
        "weights": ["projects/08_plantvlm_expert_captions/runs/run_002_fold0/checkpoints/best/adapter_model.safetensors"],
        "architecture": "Qwen2.5-VL-7B-Instruct + LoRA r16 / alpha 32 on the language model", "gpu": True,
        "dataset": "30 PlantVillage tomato images, 150 expert-revised captions, 6 classes",
        "output": "free-text description (no class parsing: the 6-class retrieval protocol is not reproduced here)",
        "metrics": [("Validation fold (class retrieval)", 2 / 6, None, "fold 0 validation, 1 image per class", 6,
                     "projects/08_plantvlm_expert_captions/runs/run_002_fold0/evaluation/val_report.json",
                     "chance = 1/6")],
        "source": "projects/08_plantvlm_expert_captions/; evaluation/vlm_predict.py",
        "limitations": "Trained on 30 images: exploratory, not a usable classifier.",
    },
    {
        "id": "tomato_qwen_base", "crop": "Tomato", "name": "Qwen2.5-VL-7B without fine-tuning (baseline)",
        "family": "Vision-language baseline", "period": "Oct 2026 re-evaluation", "env": "vlm",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "qwen_base"}, "weights": [],
        "architecture": "Qwen/Qwen2.5-VL-7B-Instruct, no adapter", "gpu": True,
        "dataset": "-", "output": "generated text; parsed condition",
        "metrics": [("Validation (condition)", 0.0850, 0.0800, PV200, 200,
                     "evaluation/qwen_zeroshot/evaluation_report.json", "")],
        "source": "evaluation/vlm_predict.py", "limitations": "Shows why fine-tuning was needed.",
    },
    {
        "id": "tomato_florence_base", "crop": "Tomato", "name": "Florence-2-base-ft without fine-tuning (baseline)",
        "family": "Vision-language baseline (failed experiment)", "period": "Apr / Jun 2026", "env": "florence",
        "adapter": "tomato.vlm_plantvillage", "args": {"model": "florence"}, "weights": [],
        "architecture": "microsoft/Florence-2-base-ft, no adapter (fine-tuning never completed)", "gpu": True,
        "dataset": "-", "output": "scene caption; parsed condition",
        "metrics": [("Validation (condition)", 0.0, 0.0, PV200, 200,
                     "evaluation/florence_zeroshot/evaluation_report.json", "")],
        "source": "failed_experiments/01_florence2_finetuning/; evaluation/vlm_predict.py",
        "limitations": "Describes the scene, never names a disease.",
    },
    # ------------------------------------------------------------------ TOMATO: MLX and TIH
    {
        "id": "tomato_mlx_kg", "crop": "Tomato", "name": "Knowledge-grounded Qwen2.5-VL MLX QLoRA (tomato)",
        "family": "Vision-language (knowledge-grounded caption)", "period": "Jul 2026; retrained Nov 2026",
        "env": "mlx", "adapter": "common.mlx_plantdx", "args": {"crop": "tomato", "run": "qwen25vl_tomato_qlora"},
        "headline": 1,
        "weights": [f"{P09}/checkpoints/qwen25vl_tomato_qlora/adapters.safetensors"],
        "architecture": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit + QLoRA r16 / alpha 32 (196 modules)",
        "dataset": "PlantVillage tomato, 10 classes, 18,006 images (16,201 / 895 / 910)",
        "output": "generated diagnostic caption; disease parsed with the project's lexicon",
        "metrics": [("Test, July 2026 adapter (not retained)", 0.9374, 0.9194, "MLX test split", 910,
                     f"{P09}/reports/qwen25vl_tomato_qlora/evaluation/metrics.json", "weights lost"),
                    ("Test, Nov 2026 retrain, iteration 13,200 (served)", 0.9088, 0.8767, "MLX test split", 910,
                     f"{P09}/retrained_2026-11/qwen25vl_tomato_qlora/evaluation/metrics.json",
                     "snapshot chosen on validation: 92.6%, F1 0.904"),
                    ("Test, Nov 2026 final iteration 16,201", 0.8692, 0.8471, "MLX test split", 910,
                     f"{P09}/retrained_2026-11/qwen25vl_tomato_qlora/final_iteration_16201/evaluation/metrics.json",
                     ""),
                    ("Test, base model (no adapter)", 0.0725, 0.034, "MLX test split", 910,
                     f"{P09}/reports/qwen25vl_tomato_qlora/evaluation/metrics.json", "")],
        "source": f"{P09}/src/plantdx/ (training/inference.py, evaluation/)",
        "limitations": "Runs only on macOS with Apple Silicon (MLX). Retraining is not bit-repeatable "
                       "(mlx-vlm 0.6.4 takes no seed).",
        "requires": "mac",
    },
    {
        "id": "tomato_tih_qlora", "crop": "Tomato", "name": "TIH 11-class Qwen2.5-VL QLoRA (ICAIN 2026)",
        "family": "Vision-language classifier", "period": "Sep 2026", "env": "tih", "adapter": "tomato.tih_qlora",
        "args": {"run": "tih_tomato_moreaccurate"},
        "weights": [f"{P10}/checkpoints/tih_tomato_moreaccurate/adapter_model.safetensors"],
        "architecture": "Qwen/Qwen2.5-VL-7B-Instruct in 4-bit NF4 (bitsandbytes) + PEFT LoRA r64 / alpha 128, "
                        "532 px", "gpu": True,
        "dataset": "Private TIH Tomato_12thAug26, 11 classes (5,247 / 649 / 649)",
        "output": "class name generated greedily, parsed to 1 of 11 classes; confidence = mean probability of "
                  "the generated tokens",
        "metrics": [("Internal test", 0.9276, 0.9378, "frozen TIH test split", 649,
                     f"{P10}/reports/tih_tomato_moreaccurate/test_finetuned/metrics.json", ""),
                    ("External benchmark", 0.6932, 0.7195, "176 images from another source (10 classes)", 176,
                     f"{P10}/reports/tih_tomato_moreaccurate_external/metrics.json", "Late Blight F1 14.29%"),
                    ("Internal test, earlier 'strong' run (r32, 392 px)", 0.8798, 0.8991, "frozen TIH test split",
                     649, f"{P10}/reports/tih_tomato_qwen_qlora_strong/test_finetuned/metrics.json",
                     "not the served adapter")],
        "source": f"{P10}/src/plantdx/company/ (predict.py, modeling.py, evaluate.py); {P10}/app/",
        "limitations": "Needs an NVIDIA CUDA GPU (bitsandbytes 4-bit). Trained on private TIH images; "
                       "PlantVillage-style photos are outside its training domain. 69% on external photos.",
    },
    # ------------------------------------------------------------------ MANGO
    {
        "id": "mango_mlx_kg", "crop": "Mango", "name": "Knowledge-grounded Qwen2.5-VL MLX QLoRA (mango)",
        "family": "Vision-language (knowledge-grounded caption)", "period": "Jul 2026; retrained Nov 2026",
        "env": "mlx", "adapter": "common.mlx_plantdx", "args": {"crop": "mango", "run": "qwen25vl_mango_qlora"},
        "headline": 1,
        "weights": [f"{P09}/checkpoints/qwen25vl_mango_qlora/adapters.safetensors"],
        "architecture": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit + QLoRA r16 / alpha 32",
        "dataset": "MangoLeafBD, 8 classes, 4,000 images (3,600 / 200 / 200)",
        "output": "generated diagnostic caption; disease parsed with the project's lexicon",
        "metrics": [("Test, July 2026 adapter (not retained)", 0.820, 0.8112, "MLX test split", 200,
                     f"{P09}/reports/qwen25vl_mango_qlora/evaluation/metrics.json", "weights lost"),
                    ("Test, Nov 2026 retrain, final iteration 3,600 (served)", 0.860, 0.8906, "MLX test split", 200,
                     f"{P09}/retrained_2026-11/qwen25vl_mango_qlora/evaluation/metrics.json", ""),
                    ("Test, base model (no adapter)", 0.175, 0.128, "MLX test split", 200,
                     f"{P09}/reports/qwen25vl_mango_qlora/evaluation/metrics.json", "")],
        "source": f"{P09}/src/plantdx/",
        "limitations": "Runs only on macOS with Apple Silicon. Die Back is the weakest class.",
        "requires": "mac",
    },
    # ------------------------------------------------------------------ SOYBEAN
]

_SOY = [
    ("soy_resnet50_keras", "SoyNet ResNet50 (Keras)", "keras_resnet50", 0.9516, (0.7692, 0.9639, 0.8556, 0.9903), 558,
     f"{P03}/results/experiments/resnet50/soynet_resnet50_best.keras", f"{P03}/results/experiments/model_comparison.json",
     "Keras run, own split"),
    ("soy_mobilenetv2_keras", "SoyNet MobileNetV2 (Keras)", "keras_mobilenetv2", 0.8907, (0.7292, 0.4217, 0.5344, 0.8748),
     558, f"{P03}/results/experiments/mobilenetv2/soynet_mobilenetv2_best.keras",
     f"{P03}/results/experiments/model_comparison.json", "Keras run, own split"),
    ("soy_resnet50_torch", "SoyNet ResNet50 (PyTorch)", "resnet50", 0.9161, (0.6412, 1.0, 0.7814, 0.9866), 560,
     f"{P03}/results/experiments/resnet50/model.pth", f"{P03}/results/experiments/resnet50/metrics.json",
     "70/15/15 seed 42 test split"),
    ("soy_mobilenetv2_torch", "SoyNet MobileNetV2 (PyTorch)", "mobilenetv2", 0.9250, (0.6667, 1.0, 0.8000, 0.9889), 560,
     f"{P03}/results/experiments/mobilenetv2/model.pth", f"{P03}/results/experiments/mobilenetv2/metrics.json",
     "70/15/15 seed 42 test split"),
    ("soy_efficientnetb0_torch", "SoyNet EfficientNetB0 (PyTorch)", "efficientnet_b0", 0.9339,
     (0.6942, 1.0, 0.8195, 0.9897), 560, f"{P03}/results/experiments/efficientnet_b0/model.pth",
     f"{P03}/results/experiments/efficientnet_b0/metrics.json", "70/15/15 seed 42 test split"),
    ("soy_efficientnetb3_torch", "SoyNet EfficientNetB3 (PyTorch)", "efficientnet_b3", 0.9393,
     (0.7119, 1.0, 0.8317, 0.9875), 560, f"{P03}/results/experiments/efficientnet_b3/model.pth",
     f"{P03}/results/experiments/efficientnet_b3/metrics.json", "70/15/15 seed 42 test split"),
]
for mid, name, arch, acc, (p, r, f1, auc), n, w, rep, split in _SOY:
    MODELS.append({
        "id": mid, "crop": "Soybean", "name": name, "family": "CNN classifier (binary)", "period": "27 Jan 2026",
        "env": "soynet", "adapter": "soybean.soynet", "args": {"model": arch}, "weights": [w],
        "architecture": f"ImageNet {name.split()[1]} backbone + sigmoid head, 256x256",
        "dataset": "SoyNet camera photos, healthy vs diseased (3,164 diseased, 556 healthy)",
        "output": "P(Healthy) from the sigmoid output; P(Diseased) = 1 - P(Healthy)",
        "metrics": [("Test", acc, None, split, n, rep,
                     f"Healthy-class precision {p:.4f}, recall {r:.4f}, F1 {f1:.4f}; ROC-AUC {auc:.4f}")],
        "source": f"{P03}/" + ("run_experiments.py (Keras)" if "keras" in mid else "run_pytorch.py, predict_soynet.py"),
        "limitations": "Binary task only (no disease name); 5.7:1 class imbalance; metrics are for the Healthy "
                       "class. SoyNet images are not redistributed, so no soybean sample image is bundled.",
    })

BY_ID = {m["id"]: m for m in MODELS}
CROPS = ["Tomato", "Mango", "Soybean"]


def models_for(crop: str) -> list[dict]:
    return [m for m in MODELS if m["crop"] == crop]
