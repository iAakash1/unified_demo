"""File logging that never writes the user's home folder, the repository path or tokens."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from config import settings

_TOKEN = re.compile(r"(hf_[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{10,})")


def redact(text: str) -> str:
    text = str(text)
    for path, tag in ((str(settings.REPO), "<repo>"), (str(Path.home()), "~")):
        text = text.replace(path, tag).replace(path.replace("\\", "/"), tag)
    return _TOKEN.sub("<token>", text)


class _Redact(logging.Filter):
    def filter(self, record):
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"tihdemo.{name}")
    if not logger.handlers:
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(settings.LOG_DIR / "app.log", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        h.addFilter(_Redact())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
