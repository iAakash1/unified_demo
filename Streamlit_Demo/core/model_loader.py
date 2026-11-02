"""Start and talk to one persistent worker process per environment.

A worker keeps its models loaded between requests, so only the first prediction of a model pays the loading
time. Workers that use the GPU (vlm, florence, tih) share one 24 GB card: before one of them loads a model, the
others are asked to free theirs, so at most one large model is in GPU memory.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from config import settings
from utils.logging_utils import get_logger, redact

log = get_logger("workers")


class WorkerError(RuntimeError):
    pass


class Worker:
    def __init__(self, env: str):
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.replies: queue.Queue = queue.Queue()
        self.holds_model = False

    def _start(self) -> None:
        py = settings.ENV_PYTHON[self.env]
        if not Path(py).exists():
            raise WorkerError(f"the '{self.env}' environment is not installed")
        env = {**os.environ, "HANDOVER_ROOT": str(settings.REPO), "PYTHONIOENCODING": "utf-8",
               "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2", "TOKENIZERS_PARALLELISM": "false"}
        if self.env in settings.CPU_ONLY_ENVS:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        if self.env in settings.TF_GROWTH_ENVS:
            env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        errlog = open(settings.LOG_DIR / f"worker_{self.env}.log", "a", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if settings.IS_WINDOWS else 0
        self.proc = subprocess.Popen([str(py), str(settings.WORKER_SCRIPT)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=errlog, cwd=str(settings.REPO), env=env,
                                     creationflags=flags)
        threading.Thread(target=self._reader, daemon=True).start()
        log.info("started worker %s (pid %s)", self.env, self.proc.pid)

    def _reader(self) -> None:
        for raw in self.proc.stdout:
            try:
                self.replies.put(json.loads(raw.decode("utf-8")))
            except ValueError:
                pass
        self.replies.put({"ok": False, "error": "worker exited", "_dead": True})

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def request(self, payload: dict, timeout: float = settings.REQUEST_TIMEOUT_S) -> dict:
        with self.lock:
            if not self.alive():
                self.replies = queue.Queue()
                self._start()
            self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            try:
                reply = self.replies.get(timeout=timeout)
            except queue.Empty:
                self.stop()
                raise WorkerError(f"no answer within {timeout:.0f} s; the worker was restarted")
            if reply.get("_dead"):
                raise WorkerError(f"the '{self.env}' worker stopped; see logs/worker_{self.env}.log")
            if not reply.get("ok"):
                raise WorkerError(redact(reply.get("error", "unknown error")))
            return reply.get("result", {})

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'{"op": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        self.proc = None
        self.holds_model = False


class WorkerPool:
    def __init__(self):
        self.workers: dict[str, Worker] = {}
        self.gpu_lock = threading.Lock()

    def get(self, env: str) -> Worker:
        if env not in self.workers:
            self.workers[env] = Worker(env)
        return self.workers[env]

    def predict(self, env: str, adapter: str, args: dict, image: str, heavy: bool) -> dict:
        worker = self.get(env)
        if env in settings.GPU_ENVS and heavy:
            with self.gpu_lock:
                for other in self.workers.values():
                    if other is not worker and other.env in settings.GPU_ENVS and other.holds_model and other.alive():
                        other.request({"op": "unload"})
                        other.holds_model = False
                        log.info("freed GPU memory in worker %s", other.env)
                result = worker.request({"op": "predict", "adapter": adapter, "args": args, "image": image})
                worker.holds_model = True
                return result
        return worker.request({"op": "predict", "adapter": adapter, "args": args, "image": image})

    def info(self, env: str) -> dict:
        return self.get(env).request({"op": "info"}, timeout=180)

    def status(self) -> dict:
        return {env: ("running" if w.alive() else "stopped") for env, w in self.workers.items()}

    def stop_all(self) -> None:
        for w in self.workers.values():
            w.stop()
