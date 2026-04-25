from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional

try:
  from .models import SubTask, ProgressEntry
  from .config import SUBAGENT_PROGRESS_DETAIL_MAX_CHARS, SUBAGENT_REPORT_MAX_CHARS
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.subagent.models import SubTask, ProgressEntry  # type: ignore
  from bridge.subagent.config import (  # type: ignore
    SUBAGENT_PROGRESS_DETAIL_MAX_CHARS,
    SUBAGENT_REPORT_MAX_CHARS,
  )


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
  if text is None or limit <= 0 or len(text) <= limit:
    return text
  # Reserve room for the marker so the final string never exceeds ``limit``.
  marker = " …[truncated]"
  keep = max(0, limit - len(marker))
  return text[:keep] + marker


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
    # Hard cap the detail length — we cannot trust the upstream cap to hold
    # if SUBAGENT_PROGRESS_DETAIL_MAX_CHARS gets tightened on this side.
    detail = _truncate(detail, SUBAGENT_PROGRESS_DETAIL_MAX_CHARS) or ""
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
    raw_report = result.get("report") or result.get("error") or None
    task.report = _truncate(raw_report, SUBAGENT_REPORT_MAX_CHARS)
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
    lines.append("## Active sub-agent task (already running for this chat)")
    lines.append(f"- Instruction: {task.instruction}")
    lines.append(f"- Running for: {elapsed_text}")

    if task.progress:
      lines.append("")
      lines.append("Progress so far:")
      for entry in task.progress:
        lines.append(f"- {entry.step}: {entry.detail}")

    lines.append("")
    lines.append("Rules while a sub-agent is in flight:")
    lines.append("- DO NOT call `execute_subtask` again for this chat — one is already running.")
    lines.append(
      "- DO NOT re-acknowledge with phrases like \"oke aku cek\" / "
      "\"sebentar ya\" / \"on it\". The user already saw your earlier "
      "acknowledgement and the typing indicator is on while the sub-agent "
      "works."
    )
    lines.append(
      "- If the user asks an unrelated question, answer that briefly. "
      "If they're just checking on progress, stay silent (no `reply_message`)."
    )
    lines.append(
      "- The sub-agent's final report will be delivered to you on the next "
      "turn as a `[SUBTASK FINISHED]` system message; that's when you "
      "summarise the result for the user."
    )

    return "\n".join(lines)

  def format_recent_finished(self, chat_id: str, *, max_age_seconds: float = 300.0) -> str | None:
    """Render the most recently finished sub-agent task for ``chat_id`` if it
    finished within ``max_age_seconds``.

    This exists so a follow-up message (e.g. user replying to the report)
    in a fresh burst still has a clear, prompt-level signal of what just
    happened. The persistent history already carries the [SUBTASK FINISHED]
    line, but the model is more reliable when we surface it as a dedicated
    context slot rather than relying on it being noticed inside a chat
    transcript.
    """
    history = self._history.get(chat_id)
    if not history:
      return None
    task = history[-1]
    if task.end_time is None:
      return None
    age = time.time() - task.end_time
    if age < 0 or age > max_age_seconds:
      return None
    success_text = "yes" if task.status == "completed" else "no"
    lines: list[str] = []
    lines.append("## Recently finished sub-agent task")
    lines.append(f"- Instruction: {task.instruction}")
    lines.append(f"- Success: {success_text}")
    if task.report:
      lines.append(f"- Report: {task.report}")
    lines.append("")
    lines.append(
      "This task has already been delivered to the user. DO NOT call "
      "`execute_subtask` again to redo it. If the user is referencing it, "
      "answer from the report above."
    )
    return "\n".join(lines)
