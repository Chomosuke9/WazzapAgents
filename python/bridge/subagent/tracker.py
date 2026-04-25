from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional

try:
  from .models import SubTask, ProgressEntry
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.subagent.models import SubTask, ProgressEntry  # type: ignore


class SubTaskTracker:
  def __init__(self) -> None:
    self._active: Dict[str, SubTask] = {}
    self._history: Dict[str, Deque[SubTask]] = {}

  def register(self, task: SubTask) -> None:
    self._active[task.session_id] = task
    self._history.setdefault(task.chat_id, deque(maxlen=50))

  def update_progress(self, session_id: str, step: str, detail: str) -> None:
    task = self._active.get(session_id)
    if task is None:
      return
    # Skip duplicate: if the last entry has identical step+detail, ignore
    if task.progress:
      last = task.progress[-1]
      if last.step == step and last.detail == detail:
        return
    task.progress.append(ProgressEntry(
      step=step,
      detail=detail,
      timestamp=time.time(),
    ))

  def finalize(self, session_id: str, result: dict) -> None:
    task = self._active.pop(session_id, None)
    if task is None:
      return
    task.end_time = time.time()
    task.result = result
    success = result.get("success", False)
    task.status = "completed" if success else "failed"
    task.report = result.get("report") or result.get("error") or None
    self._history.setdefault(task.chat_id, deque(maxlen=50)).append(task)

  def get_active_for_chat(self, chat_id: str) -> SubTask | None:
    for task in self._active.values():
      if task.chat_id == chat_id:
        return task
    return None

  def format_context(self, chat_id: str) -> str | None:
    task = self.get_active_for_chat(chat_id)
    if task is None:
      return None

    elapsed = task.elapsed_seconds
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    elapsed_text = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    lines: list[str] = []
    lines.append("## Task information")
    lines.append(f"- {task.instruction}")
    lines.append("")

    if task.progress:
      lines.append("## Sub-Agent history:")
      for entry in task.progress:
        lines.append(f"- {entry.step}: {entry.detail}")
      lines.append("")

    lines.append("## Information")
    lines.append(f"- Sub-Agent is currently running for {elapsed_text}")

    return "\n".join(lines)
