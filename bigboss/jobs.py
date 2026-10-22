"""Minimal background job runner for long tasks (ingestion, training, experiments).

Jobs run in a small thread pool; status/progress is kept in memory and persisted to data/jobs.jsonl
so the history survives restarts (running jobs do not).
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class JobManager:
    def __init__(self, log_path: Path, workers: int = 2):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bigboss-job")
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.log_path = log_path
        if log_path.exists():
            for line in log_path.read_text().splitlines()[-200:]:
                try:
                    j = json.loads(line)
                    if j.get("status") in ("queued", "running"):
                        j["status"] = "interrupted"
                    self.jobs[j["id"]] = j
                except json.JSONDecodeError:
                    pass

    def _persist(self, job: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps({k: v for k, v in job.items() if k != "result"} | {"has_result": job.get("result") is not None}, default=str) + "\n")

    def submit(self, kind: str, fn: Callable[[Callable[[float, str], None]], object], params: dict | None = None) -> str:
        jid = f"{kind}-{uuid.uuid4().hex[:8]}"
        job = {"id": jid, "kind": kind, "params": params or {}, "status": "queued", "progress": 0.0, "message": "",
               "created_at": datetime.now(timezone.utc).isoformat(), "started_at": None, "finished_at": None, "result": None, "error": None}
        with self.lock:
            self.jobs[jid] = job

        def progress(frac: float, msg: str) -> None:
            job["progress"] = round(float(frac), 3)
            job["message"] = msg

        def run():
            job["status"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            try:
                job["result"] = fn(progress)
                job["status"] = "succeeded"
                job["progress"] = 1.0
            except Exception as e:  # report every failure to the user
                job["status"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
                job["traceback"] = traceback.format_exc()[-4000:]
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._persist(job)

        self.pool.submit(run)
        return jid

    def get(self, jid: str) -> dict | None:
        return self.jobs.get(jid)

    def list(self, limit: int = 50) -> list[dict]:
        js = sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]
        return [{k: v for k, v in j.items() if k not in ("result", "traceback")} for j in js]
