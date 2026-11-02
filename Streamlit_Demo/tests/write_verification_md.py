"""Turn Demo_Assets/verification/verification_results.json into a readable VERIFICATION.md."""
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "Demo_Assets" / "verification"
rows = json.loads((D / "verification_results.json").read_text(encoding="utf-8"))
out = ["# Verification: app output vs the repository's original scripts", "",
       "Run on the TIH workstation (Windows, NVIDIA RTX 4500 Ada) with `Streamlit_Demo/tests/verify_against_originals.py`.",
       "",
       "**What each row compares.** The same image went through two paths:",
       "",
       "1. the app's worker path, which is what the Streamlit page uses;",
       "2. the original command from the handover repository, run in its own environment.",
       "",
       "**Images used.**",
       "",
       "- Tomato: `late_blight` test image from `Demo_Assets/sample_images/tomato`.",
       "- Soybean: one healthy and one diseased SoyNet test image, read from the local SoyNet folder (not copied).",
       "",
       "**What MATCH means.**",
       "",
       "- Classifiers: the same top label, with the same probability to within 0.002. The original scripts print "
       "probabilities rounded to 3 decimals.",
       "- Vision-language models: identical generated text.",
       "",
       "| Model | Image | Result | App output | Score | Original script | Load s | Inference s |",
       "|---|---|---|---|---:|---|---:|---:|"]
for r in rows:
    a, o = r.get("app", {}), r.get("original", {})
    label = (a.get("label") or "-") if r.get("app") else "-"
    text = (a.get("text") or "").replace("\n", " ").replace("|", "/")
    shown = f"{label}" + (f" - \"{text[:90]}{'…' if len(text) > 90 else ''}\"" if text and text != label else "")
    cmd = o.get("command", o.get("note", "-")) if o else "-"
    cmd = re.sub(r"--image .*?(?= --|$)", "--image <image>", cmd)
    score = "-" if a.get("score") is None else f"{a['score']:.4f}"
    out.append(f"| {r['model']} | {r['image'][:24]} | {r['status'] if r['status'] != 'NOT RUN' else 'NOT RUN: ' + r.get('reason', '')} | "
               f"{shown} | {score} | `{cmd}` | {a.get('load_seconds', '-')} | {a.get('infer_seconds', '-')} |")
out += ["", "Load and inference times are the app's worker timings. A load time of 0 means the model was already "
        "loaded. Models marked NOT RUN need hardware that this workstation does not have."]
(D / "VERIFICATION.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("written", D / "VERIFICATION.md", len(rows), "rows")
