# Verification: app output vs the repository's original scripts

Run on the TIH workstation (Windows, NVIDIA RTX 4500 Ada) with `Streamlit_Demo/tests/verify_against_originals.py`.

**What each row compares.** The same image went through two paths:

1. the app's worker path, which is what the Streamlit page uses;
2. the original command from the handover repository, run in its own environment.

**Images used.**

- Tomato: `late_blight` test image from `Demo_Assets/sample_images/tomato`.
- Soybean: one healthy and one diseased SoyNet test image, read from the local SoyNet folder (not copied).

**What MATCH means.**

- Classifiers: the same top label, with the same probability to within 0.002. The original scripts print probabilities rounded to 3 decimals.
- Vision-language models: identical generated text.

| Model | Image | Result | App output | Score | Original script | Load s | Inference s |
|---|---|---|---|---:|---|---:|---:|
| Custom CNN (Keras) | late_blight__01425d17-4c | MATCH | Tomato_Late_blight | 0.9999 | `python.exe predict.py --image <image> --model cnn` | 2.27 | 0.097 |
| MobileNetV2 (Keras) | late_blight__01425d17-4c | MATCH | Tomato_Late_blight | 1.0000 | `python.exe predict.py --image <image> --model mobilenet` | 0.47 | 0.457 |
| EfficientNetB3 (Keras) | late_blight__01425d17-4c | MATCH | Tomato_Late_blight | 0.9993 | `python.exe predict.py --image <image> --model efficientnet` | 1.14 | 1.259 |
| ConvNeXt-Tiny (Keras) | late_blight__01425d17-4c | MATCH | Tomato_Late_blight | 1.0000 | `python.exe predict.py --image <image> --model convnext` | 0.94 | 0.647 |
| ConvNeXt-Tiny (PyTorch) | late_blight__01425d17-4c | MATCH | Tomato_Late_blight | 1.0000 | `python.exe predict.py --image <image>` | 0.4 | 0.135 |
| BLIP-2 OPT-2.7B + LoRA (March) | late_blight__01425d17-4c | MATCH | Late Blight - "Plant: Tomato. Condition: Late Blight. Severity: Moderate to Severe. Progressive dark lesi…" | - | `python.exe vlm_predict.py --model blip2_march --image <image>` | 12.55 | 2.359 |
| BLIP-2 OPT-2.7B + LoRA (June re-run) | late_blight__01425d17-4c | MATCH | Late Blight - "Plant: Tomato. Condition: Late Blight. Severity: Moderate to Severe. Progressive dark lesi…" | - | `python.exe vlm_predict.py --model blip2_june --image <image>` | 8.68 | 2.114 |
| LLaVA-1.6-Mistral-7B + LoRA | late_blight__01425d17-4c | MATCH | Unknown - "The image you've provided shows a close-up of a leaf with several visible features that co…" | - | `python.exe vlm_predict.py --model llava --image <image>` | 14.19 | 20.837 |
| Qwen2.5-VL-7B + LoRA, PlantVillage (April) | late_blight__01425d17-4c | MATCH | Late Blight - "Plant: Tomato. Condition: Late Blight. Severity: Moderate to Severe. Pathogen: Phytophthor…" | - | `python.exe vlm_predict.py --model qwen_april --image <image>` | 12.56 | 10.639 |
| PlantDx Qwen2.5-VL-7B + LoRA (May) | late_blight__01425d17-4c | MATCH | Late Blight - "Plant: Tomato. Condition: Late Blight. Severity: Moderate to Severe. The leaf exhibits rap…" | - | `python.exe vlm_predict.py --model plantdx --image <image>` | 12.19 | 3.374 |
| PlantVLM expert-caption Qwen2.5-VL + LoRA | late_blight__01425d17-4c | MATCH | (free text; no class parsing) - "The central vein is clearly visible and runs from the base to the tip of the leaf. The col…" | - | `python.exe vlm_predict.py --model plantvlm --image <image>` | 10.94 | 9.479 |
| Qwen2.5-VL-7B without fine-tuning (baseline) | late_blight__01425d17-4c | MATCH | Unknown - "The image shows a leaf that appears to be from a broadleaf plant, possibly an oak (Quercus…" | - | `python.exe vlm_predict.py --model qwen_base --image <image>` | 13.7 | 7.985 |
| Florence-2-base-ft without fine-tuning (baseline) | late_blight__01425d17-4c | MATCH | Unknown - "There is a leaf on the ground. The leaf is green. There are small holes in the leaf. There…" | - | `python.exe vlm_predict.py --model florence --image <image>` | 10.97 | 3.119 |
| Knowledge-grounded Qwen2.5-VL MLX QLoRA (tomato) | late_blight__01425d17-4c | NOT RUN: needs macOS on Apple Silicon (MLX); not available on this machine | - | - | `-` | - | - |
| TIH 11-class Qwen2.5-VL QLoRA (ICAIN 2026) | late_blight__01425d17-4c | MATCH | Late Blight | 1.0000 | `python.exe -m plantdx.cli company predict --config configs/train/qwen25vl_tih_tomato_moreaccurate.yaml --adapter checkpoints/tih_tomato_moreaccurate --image <image> --json` | 36.21 | 3.958 |
| Knowledge-grounded Qwen2.5-VL MLX QLoRA (mango) | anthracnose__20211008_12 | NOT RUN: needs macOS on Apple Silicon (MLX); not available on this machine | - | - | `-` | - | - |
| SoyNet ResNet50 (Keras) | soy (23).JPG | no original script (app result only) | Healthy | 0.9922 | `no single-image script in the repository for this model` | 14.83 | 3.755 |
| SoyNet ResNet50 (Keras) | Soy  (218).JPG | no original script (app result only) | Diseased | 1.0000 | `no single-image script in the repository for this model` | 0.0 | 0.038 |
| SoyNet MobileNetV2 (Keras) | soy (23).JPG | no original script (app result only) | Diseased | 0.9426 | `no single-image script in the repository for this model` | 0.73 | 0.44 |
| SoyNet MobileNetV2 (Keras) | Soy  (218).JPG | no original script (app result only) | Diseased | 1.0000 | `no single-image script in the repository for this model` | 0.0 | 0.035 |
| SoyNet ResNet50 (PyTorch) | soy (23).JPG | MATCH | Healthy | 0.9950 | `python.exe predict_soynet.py --image <image> --model resnet50` | 13.88 | 1.638 |
| SoyNet ResNet50 (PyTorch) | Soy  (218).JPG | MATCH | Diseased | 0.8812 | `python.exe predict_soynet.py --image <image> --model resnet50` | 0.0 | 0.011 |
| SoyNet MobileNetV2 (PyTorch) | soy (23).JPG | MATCH | Healthy | 0.9921 | `python.exe predict_soynet.py --image <image> --model mobilenetv2` | 0.19 | 0.169 |
| SoyNet MobileNetV2 (PyTorch) | Soy  (218).JPG | MATCH | Diseased | 0.9987 | `python.exe predict_soynet.py --image <image> --model mobilenetv2` | 0.0 | 0.006 |
| SoyNet EfficientNetB0 (PyTorch) | soy (23).JPG | MATCH | Healthy | 0.9843 | `python.exe predict_soynet.py --image <image> --model efficientnet_b0` | 0.3 | 0.164 |
| SoyNet EfficientNetB0 (PyTorch) | Soy  (218).JPG | MATCH | Diseased | 0.9848 | `python.exe predict_soynet.py --image <image> --model efficientnet_b0` | 0.0 | 0.008 |
| SoyNet EfficientNetB3 (PyTorch) | soy (23).JPG | MATCH | Healthy | 0.9966 | `python.exe predict_soynet.py --image <image> --model efficientnet_b3` | 0.58 | 0.032 |
| SoyNet EfficientNetB3 (PyTorch) | Soy  (218).JPG | MATCH | Diseased | 0.9886 | `python.exe predict_soynet.py --image <image> --model efficientnet_b3` | 0.0 | 0.011 |

Load and inference times are the app's worker timings. A load time of 0 means the model was already loaded. Models marked NOT RUN need hardware that this workstation does not have.
