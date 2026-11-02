"""Take real screenshots of the running app with headless Microsoft Edge (Chrome DevTools Protocol).

The app must already be running (default http://localhost:8765). Each shot opens a deep link, waits until
Streamlit has finished running (status widget gone, the expected element present), sizes the viewport to the page
and saves a PNG into Demo_Assets/screenshots/. Needs tornado (installed with streamlit).

    <python with streamlit> tests/capture_screenshots.py [--base http://localhost:8765] [--only name1,name2]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

APP = Path(__file__).resolve().parents[1]
OUT = APP.parent / "Demo_Assets" / "screenshots"
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", shutil.which("msedge") or "",
        shutil.which("chrome") or "", shutil.which("google-chrome") or ""]

# name, query string, CSS selector that must exist when done, max seconds
SHOTS = [
    ("01_home_tomato", "?crop=Tomato&model=tomato_convnext_torch", "[data-testid='stFileUploader']", 60),
    ("02_tomato_convnext_torch_prediction", "?crop=Tomato&model=tomato_convnext_torch&sample=late_blight&run=1",
     ".pred-card", 180),
    ("03_tomato_cnn_comparison", "?crop=Tomato&mode=compare&models=tomato_custom_cnn,tomato_mobilenetv2_keras,"
     "tomato_efficientnetb3_keras,tomato_convnext_keras,tomato_convnext_torch&sample=early_blight&run=1",
     "[data-testid='stDataFrame']", 300),
    ("04_tomato_qwen_april_prediction", "?crop=Tomato&model=tomato_qwen_april&sample=target_spot&run=1",
     ".pred-card", 900),
    ("05_tomato_tih_qlora_prediction", "?crop=Tomato&model=tomato_tih_qlora&sample=late_blight&run=1",
     ".pred-card", 900),
    ("06_tomato_mlx_restricted_with_recorded_answers", "?crop=Tomato&model=tomato_mlx_kg&sample=healthy",
     "[data-testid='stDataFrame']", 60),
    ("07_mango_mlx_restricted_with_recorded_answers", "?crop=Mango&model=mango_mlx_kg&sample=anthracnose",
     "[data-testid='stDataFrame']", 60),
    ("08_soybean_models", "?crop=Soybean&model=soy_resnet50_keras", "[data-testid='stFileUploader']", 60),
]


class CDP:
    def __init__(self, ws):
        self.ws, self.i = ws, 0

    async def call(self, method, **params):
        self.i += 1
        await self.ws.write_message(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == self.i:
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        return r.get("result", {}).get("value")


async def shoot(cdp, base, name, query, selector, limit):
    await cdp.call("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1, mobile=False)
    await cdp.call("Page.navigate", url=base + "/" + query)
    t0 = time.time()
    await asyncio.sleep(4)
    while time.time() - t0 < limit:
        done = await cdp.js(
            f"!!document.querySelector({json.dumps(selector)}) && "
            "!document.querySelector('[data-testid=\"stStatusWidget\"]') && "
            "!document.querySelector('[data-testid=\"stSpinner\"]')")
        if done:
            break
        await asyncio.sleep(2)
    else:
        print(f"{name}: timed out after {limit}s (captured anyway)")
    await asyncio.sleep(2)
    err = await cdp.js("!!document.querySelector('[data-testid=stException]') || "
                       "document.body.innerText.includes('Script compilation error') || "
                       "!document.querySelector('.hero')")
    if err:
        print(f"{name}: the page shows an error - NOT saved")
        return
    h = await cdp.js("Math.max(...[...document.querySelectorAll('[data-testid=\"stMainBlockContainer\"], "
                     "[data-testid=\"stSidebarContent\"]')].map(e => e.scrollHeight))") or 1000
    await cdp.call("Emulation.setDeviceMetricsOverride", width=1440, height=min(int(h) + 120, 4200),
                   deviceScaleFactor=1, mobile=False)
    for _ in range(30):  # the page must still be complete (no skeleton placeholders) after the resize
        await asyncio.sleep(2)
        stable = await cdp.js(f"!!document.querySelector({json.dumps(selector)}) && "
                              "!document.querySelector('[data-testid=stSkeleton]') && "
                              "!document.querySelector('[data-testid=stStatusWidget]')")
        if stable:
            break
    else:
        print(f"{name}: page never settled - NOT saved")
        return
    png = await cdp.call("Page.captureScreenshot", format="png")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.png").write_bytes(base64.b64decode(png["data"]))
    print(f"{name}: saved ({time.time() - t0:.0f}s)", flush=True)


async def main_async(base, only):
    exe = next((e for e in EDGE if e and os.path.exists(e)), None)
    if not exe:
        sys.exit("Microsoft Edge or Chrome not found")
    profile = tempfile.mkdtemp(prefix="tihdemo_edge_")
    port = 9333
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                             "--no-first-run", "--disable-extensions", "--hide-scrollbars", "about:blank"])
    try:
        for _ in range(60):
            try:
                pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                ws_url = next(p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page")
                break
            except Exception:
                await asyncio.sleep(0.5)
        ws = await websocket_connect(ws_url, max_message_size=200 * 1024 * 1024)
        cdp = CDP(ws)
        await cdp.call("Page.enable")
        for name, query, selector, limit in SHOTS:
            if only and name not in only:
                continue
            await shoot(cdp, base, name, query, selector, limit)
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8765")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    asyncio.run(main_async(a.base, [s for s in a.only.split(",") if s]))
