"""Pin the context-block contracts that prevent the post-task delivery loop.

The user-visible bug these tests guard against:

  - The bot keeps replying "Siap, aku cek dokumennya" over and over even
    after the sub-agent has finished, and the actual report never reaches
    the user.

The fixes rely on three context blocks that LLM2 sees:

  1. ``format_context()`` — when a task is in flight, must tell LLM2 not to
     re-acknowledge or re-spawn the same task.
  2. ``format_recent_finished()`` — for follow-up bursts after delivery,
     must mark the task as already-delivered so LLM2 does not redo it.
  3. The post-task re-invoke ``subagent_result_block`` (built in main.py)
     uses ``[SUBTASK FINISHED]`` from the same tracker state. The recent-
     finished helper must keep that report intact.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge.subagent.tracker import SubTaskTracker  # noqa: E402
from bridge.subagent.models import SubTask  # noqa: E402


def _make_tracker_with_active(chat_id: str = "chat-1") -> SubTaskTracker:
    tracker = SubTaskTracker()
    tracker.register(SubTask(
        session_id="sess-1",
        chat_id=chat_id,
        instruction="Summarise the attached PDF in one paragraph.",
    ))
    return tracker


def test_active_context_warns_against_reack_and_respawn():
    tracker = _make_tracker_with_active()
    block = tracker.format_context("chat-1")
    assert block is not None
    # Active-task header + instruction should be present.
    assert "Active sub-agent task" in block
    assert "Summarise the attached PDF" in block
    # The new anti-loop guidance MUST be present — without it the model
    # happily emits another "oke aku cek dulu" while the task is in flight.
    assert "DO NOT call `execute_subtask` again" in block
    assert "DO NOT re-acknowledge" in block


def test_active_context_returns_none_when_no_task():
    tracker = SubTaskTracker()
    assert tracker.format_context("chat-1") is None


def test_recent_finished_surfaces_report_for_followup_bursts():
    tracker = _make_tracker_with_active()
    tracker.finalize(
        "sess-1",
        {
            "success": True,
            "report": "PDF says: project deadline is next Friday.",
            "output_files": [],
        },
    )
    block = tracker.format_recent_finished("chat-1")
    assert block is not None
    assert "Recently finished sub-agent task" in block
    assert "Summarise the attached PDF" in block
    assert "next Friday" in block
    # Must explicitly tell LLM2 the task is delivered so a follow-up burst
    # doesn't trigger another execute_subtask.
    assert "DO NOT call `execute_subtask` again" in block


def test_recent_finished_expires_after_max_age():
    tracker = _make_tracker_with_active()
    tracker.finalize("sess-1", {"success": True, "report": "ok"})
    # Force an old finish time so the helper treats it as expired.
    history = tracker._history.get("chat-1")
    assert history
    history[-1].end_time = time.time() - 10_000
    assert tracker.format_recent_finished("chat-1", max_age_seconds=300.0) is None


def test_recent_finished_returns_none_when_no_history():
    tracker = SubTaskTracker()
    assert tracker.format_recent_finished("chat-1") is None
