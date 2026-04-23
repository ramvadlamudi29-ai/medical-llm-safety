"""Lightweight experiment tracking: appends JSON lines to experiments/runs.jsonl."""
from __future__ import annotations
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from core.config import settings


def log_run(name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    run_id = uuid.uuid4().hex[:12]
    record = {
        "run_id": run_id,
        "ts": round(time.time(), 3),
        "name": name,
        "params": params,
        "metrics": metrics,
        "versions": {
            "model": settings.model_version,
            "prompt": settings.prompt_version,
            "dataset": settings.dataset_version,
        },
    }
    out_dir = Path(settings.experiments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runs.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return run_id
