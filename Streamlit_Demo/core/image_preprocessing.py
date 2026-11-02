"""Store an accepted upload unchanged on disk for the workers.

Model-specific preprocessing is NOT done here: each adapter calls its project's own code (OpenCV 32x32 pipeline,
keras.applications preprocess_input, torchvision transforms, the Qwen/BLIP-2 processors, the TIH 532 px resize),
so the bytes the user uploaded are passed through as they are.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from config import settings


def store_upload(name: str, data: bytes) -> Path:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = name.rsplit(".", 1)[-1].lower()
    path = settings.UPLOAD_DIR / f"{hashlib.sha256(data).hexdigest()[:16]}.{ext}"
    if not path.exists():
        path.write_bytes(data)
    return path
