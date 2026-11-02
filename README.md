# TIH-IoT technical handover: demo package

| Item | Path |
|---|---|
| **Consolidated report, version 3 (primary)** | `TIH_IoT_Cropwise_Modelwise_Handover_Report_v3.docx`: adds Chapter 11, a one-to-two-page usage and developer guide per model |
| Version 3 as PDF | `TIH_IoT_Cropwise_Modelwise_Handover_Report_v3.pdf` (clickable GitHub links, clickable contents, bookmarks) |
| Version 2 (no developer guides) | `TIH_IoT_Cropwise_Modelwise_Handover_Report_v2.docx` / `.pdf` |
| Version 1 (same content, links as plain text) | `TIH_IoT_Cropwise_Modelwise_Handover_Report.docx` / `.pdf` |
| Unified demo application | `Streamlit_Demo/` (start: see `Streamlit_Demo/README.md`) |
| Demo assets | `Demo_Assets/`: demonstration guide, model inventory, sample images, screenshots, recorded predictions, verification |

Code, weights and per-model guides live in https://github.com/iAakash1/handover. On the TIH workstation it is
cloned at `D:\handover`. Weights are fetched from https://github.com/iAakash1/TIH-IoT (Git LFS) by
`setup\download_models.py`.

Start the demo:

```powershell
cd C:\Users\Admin\Desktop\Demo_TIH_Handover\Streamlit_Demo
D:\handover\.venvs\vlm\Scripts\python -m streamlit run app.py
```
