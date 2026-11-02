"""Collect the MLX models' RECORDED answers for the bundled sample images (genuine outputs, not live inference).

The bundled tomato and mango samples come from the MLX test split, so the Mac evaluation runs already contain
each model's answer for exactly these images: the July 2026 run (reports/...) and the November 2026 retraining
(retrained_2026-11/...). This script copies those rows into Demo_Assets/recorded_predictions/; the app shows them
for the MLX models, labelled as recorded on the Mac.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from config import settings  # noqa: E402

P09 = settings.REPO / "projects" / "09_plantdx_mlx_knowledge_grounded"
RUNS = {
    "July 2026 adapter (not retained)": "reports/qwen25vl_{crop}_qlora/evaluation/predictions.csv",
    "Nov 2026 retrain (served adapter)": "retrained_2026-11/qwen25vl_{crop}_qlora/evaluation/predictions.csv",
}
OUT = settings.DEMO_ROOT / "Demo_Assets" / "recorded_predictions"


def main():
    samples = json.loads((settings.SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))
    out = {}
    for s in samples:
        crop = s["crop"]
        original_name = Path(s["source"]).name
        entry = {}
        for run, rel in RUNS.items():
            path = P09 / rel.format(crop=crop)
            with open(path, encoding="utf-8", newline="") as f:
                row = next((r for r in csv.DictReader(f) if r["image_id"].endswith("/" + original_name)), None)
            if row:
                entry[run] = {"instruction": row["instruction"], "answer": row["finetuned_prediction"],
                              "predicted_disease": row["finetuned_predicted_disease"],
                              "true_disease": row["disease_id"],
                              "source": str(path.relative_to(settings.REPO)).replace("\\", "/")}
        out[Path(s["file"]).name] = {"crop": crop, "class": s["class"], "runs": entry}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mlx_recorded_answers_for_samples.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    hits = sum(len(v["runs"]) for v in out.values())
    print(len(out), "samples,", hits, "recorded answers")


if __name__ == "__main__":
    main()
