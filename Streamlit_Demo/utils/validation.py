"""Check an uploaded file before any model sees it."""
from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from config import settings


def validate_image(name: str, data: bytes) -> tuple[bool, str]:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in settings.ALLOWED_TYPES:
        return False, f"unsupported file type '.{ext}' (use {', '.join(settings.ALLOWED_TYPES)})"
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        return False, f"file larger than {settings.MAX_UPLOAD_MB} MB"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            im.convert("RGB").load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        return False, f"not a readable image ({type(exc).__name__})"
    if min(w, h) < 32:
        return False, f"image too small ({w}x{h}); at least 32 px per side"
    return True, f"{w}x{h} px"
