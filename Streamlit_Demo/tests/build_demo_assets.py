"""Write Demo_Assets/MODEL_INVENTORY.csv from the registry and copy genuine recorded prediction files."""
import csv
import shutil
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from config import settings  # noqa: E402
from config.model_registry import MODELS  # noqa: E402
from utils.device_detection import availability  # noqa: E402

ASSETS = settings.DEMO_ROOT / "Demo_Assets"
with open(ASSETS / "MODEL_INVENTORY.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["crop", "model", "registry id", "type", "period", "runtime", "status on TIH workstation",
                "result (evaluation set, n)", "accuracy", "macro-F1", "evaluation report", "weights (handover path)",
                "source code"])
    for m in MODELS:
        state, reason = availability(m)
        for i, (label, acc, f1, split, n, rep, note) in enumerate(m["metrics"]):
            w.writerow([m["crop"], m["name"], m["id"], m["family"], m["period"], m["env"],
                        reason if i == 0 else "", f"{label}: {split} ({n})", f"{acc * 100:.2f}%",
                        "" if f1 is None else f"{f1:.4f}", rep, "; ".join(m["weights"]) if i == 0 else "",
                        m["source"] if i == 0 else ""])

# genuine prediction files written by the evaluation runs (public images only; no TIH data)
R = settings.REPO
P09 = R / "projects" / "09_plantdx_mlx_knowledge_grounded"
copies = {
    "plantvillage_vlms/blip2_march_predictions.json": R / "evaluation/blip2_march/predictions.json",
    "plantvillage_vlms/blip2_june_predictions.json": R / "evaluation/blip2_june/predictions.json",
    "plantvillage_vlms/llava_v16_predictions.json": R / "evaluation/llava_v16/predictions.json",
    "plantvillage_vlms/qwen_zeroshot_predictions.json": R / "evaluation/qwen_zeroshot/predictions.json",
    "plantvillage_vlms/florence_zeroshot_predictions.json": R / "evaluation/florence_zeroshot/predictions.json",
    "plantvillage_vlms/qwen_april_inference_results.json": R / "projects/06_qwen25vl_plantvillage/qwen/inference_results.json",
    "mlx/tomato_july_predictions.csv": P09 / "reports/qwen25vl_tomato_qlora/evaluation/predictions.csv",
    "mlx/tomato_nov_iter13200_predictions.csv": P09 / "retrained_2026-11/qwen25vl_tomato_qlora/evaluation/predictions.csv",
    "mlx/mango_july_predictions.csv": P09 / "reports/qwen25vl_mango_qlora/evaluation/predictions.csv",
    "mlx/mango_nov_predictions.csv": P09 / "retrained_2026-11/qwen25vl_mango_qlora/evaluation/predictions.csv",
}
out = ASSETS / "recorded_predictions"
lines = ["# Recorded predictions (genuine outputs written by the evaluation runs)", "",
         "These are copies of files in github.com/iAakash1/handover; no prediction here was produced for the demo.",
         "TIH predictions are not copied because they contain private image paths.", "",
         "| File | Copied from (handover repository) |", "|---|---|"]
for dst, src in copies.items():
    if src.exists():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / dst)
        lines.append(f"| `{dst}` | `{src.relative_to(R).as_posix()}` |")
lines += ["| `mlx_recorded_answers_for_samples.json` | rows of the four MLX predictions.csv files for the bundled "
          "sample images (built by `Streamlit_Demo/tests/build_recorded_predictions.py`) |"]
(out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("inventory rows written; copied", len(copies), "prediction files")
