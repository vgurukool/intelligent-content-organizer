import os
import glob
import json
import time
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import config

logger = logging.getLogger(__name__)

class TaskRecord(BaseModel):
    id: str
    task_type: str        # "document_ingestion" | "financial_extraction" | "semantic_search" | "summarization" | "qa"
    source: str           # "UI Upload" | "REST API" | "MCP Client" | "Google Drive" | "S3 Inbox"
    name: str             # Filename, query or document title
    status: str = "processing"  # "processing" | "completed" | "failed"
    created_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    summary: str = "Processing started"
    output: Optional[Any] = None
    error: Optional[str] = None

class TaskService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TaskService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.config = config.config
        self.task_store_path = self.config.TASK_STORE_PATH
        os.makedirs(self.task_store_path, exist_ok=True)
        self.tasks_cache: Dict[str, TaskRecord] = {}
        self._load_recent_tasks_from_disk()
        self._initialized = True

    def _load_recent_tasks_from_disk(self):
        """Load recent task records from persistent disk cache into memory."""
        try:
            files = glob.glob(os.path.join(self.task_store_path, "*.json"))
            # Sort files by modification time descending
            files.sort(key=os.path.getmtime, reverse=True)
            for f in files[:200]:  # Load up to 200 most recent
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        task = TaskRecord(**data)
                        self.tasks_cache[task.id] = task
                except Exception as e:
                    logger.debug(f"Could not load task file {f}: {e}")
            logger.info(f"Loaded {len(self.tasks_cache)} task records into cache from {self.task_store_path}")
        except Exception as e:
            logger.warning(f"Error loading task records: {e}")

    def _persist_task(self, task: TaskRecord):
        try:
            filepath = os.path.join(self.task_store_path, f"{task.id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(task.model_dump(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist task {task.id}: {e}")

    def create_task(self, task_type: str, source: str, name: str, summary: str = "Processing started") -> TaskRecord:
        task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now_iso = datetime.utcnow().isoformat() + "Z"
        task = TaskRecord(
            id=task_id,
            task_type=task_type,
            source=source,
            name=name,
            status="processing",
            created_at=now_iso,
            summary=summary
        )
        self.tasks_cache[task_id] = task
        self._persist_task(task)
        return task

    def complete_task(self, task_id: str, summary: str, output: Optional[Any] = None, duration_ms: Optional[int] = None) -> Optional[TaskRecord]:
        task = self.tasks_cache.get(task_id)
        if not task:
            # Try to load from disk
            p = os.path.join(self.task_store_path, f"{task_id}.json")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    task = TaskRecord(**json.load(f))
            else:
                return None

        now_iso = datetime.utcnow().isoformat() + "Z"
        task.status = "completed"
        task.completed_at = now_iso
        task.summary = summary
        task.output = output
        if duration_ms is not None:
            task.duration_ms = duration_ms
        elif task.created_at:
            try:
                created_dt = datetime.fromisoformat(task.created_at.replace("Z", "+00:00"))
                now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
                task.duration_ms = int((now_dt - created_dt).total_seconds() * 1000)
            except Exception:
                pass

        self.tasks_cache[task_id] = task
        self._persist_task(task)
        return task

    def fail_task(self, task_id: str, error: str, duration_ms: Optional[int] = None) -> Optional[TaskRecord]:
        task = self.tasks_cache.get(task_id)
        if not task:
            return None

        now_iso = datetime.utcnow().isoformat() + "Z"
        task.status = "failed"
        task.completed_at = now_iso
        task.error = error
        task.summary = f"Failed: {error}"
        if duration_ms is not None:
            task.duration_ms = duration_ms
        elif task.created_at:
            try:
                created_dt = datetime.fromisoformat(task.created_at.replace("Z", "+00:00"))
                now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
                task.duration_ms = int((now_dt - created_dt).total_seconds() * 1000)
            except Exception:
                pass

        self.tasks_cache[task_id] = task
        self._persist_task(task)
        return task

    def list_tasks(self, limit: int = 100, status_filter: Optional[str] = None, source_filter: Optional[str] = None) -> List[TaskRecord]:
        tasks = list(self.tasks_cache.values())
        # Sort descending by created_at
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        filtered = []
        for t in tasks:
            if status_filter and status_filter.lower() != "all" and t.status.lower() != status_filter.lower():
                continue
            if source_filter and source_filter.lower() != "all" and t.source.lower() != source_filter.lower():
                continue
            filtered.append(t)
            if len(filtered) >= limit:
                break
        return filtered

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        if task_id in self.tasks_cache:
            return self.tasks_cache[task_id]
        p = os.path.join(self.task_store_path, f"{task_id}.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    t = TaskRecord(**json.load(f))
                    self.tasks_cache[task_id] = t
                    return t
            except Exception:
                pass
        return None

# Global singleton
task_service = TaskService()
