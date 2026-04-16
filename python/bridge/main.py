from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Set
from urllib.parse import urlsplit

import websockets
from dotenv import load_dotenv

try:
  from .history import (
    WhatsAppMessage,
    assistant_name,
    assistant_sender_ref,
    assistant_name_pattern,
  )
  from .log import setup_logging, set_chat_log_context, reset_chat_log_context
  from .llm.llm1 import call_llm1, LLM1Decision
  from .llm.llm2 import generate_reply
  from .db import (
    get_mode as db_get_mode,
    get_triggers as db_get_triggers,
    is_muted as db_is_muted,
    is_mute_notified as db_is_mute_notified,
    mark_mute_notified as db_mark_mute_notified,
    add_mute as db_add_mute,
    remove_mute as db_remove_mute,
    clear_mutes as db_clear_mutes,
    get_mute_remaining_minutes as db_get_mute_remaining,
    set_permission as db_set_permission,
  )
  from .dashboard import record_stat, record_user_invoke, flush_to_db, start_flush_loop
  from .stickers import resolve_sticker
  from .tools.sticker import create_sticker_file
  from .messaging.processing import (
    _append_history,
    _append_or_merge_history_payload,
    _build_burst_current,
    _clean_text,
    _collect_context_ids,
    _extract_send_ack_context_msg_id,
    _hydrate_provisional_context_id_from_ack,
    _infer_media,
    _is_context_only_payload,
    _make_request_id,
    _normalize_context_msg_id,
    _normalize_preview_text,
    _payload_to_message,
    _quoted_preview,
    _reply_signature,
  )
  from .messaging.filtering import (
    _chat_state_from_payload,
    _message_matches_prefix,
    _payload_has_meaningful_content,
    _payload_triggers_llm1,
  )
  from .llm.metadata import (
    _build_llm1_context_metadata,
    _resolve_group_prompt_context,
  )
  from .messaging.moderation import (
    _merge_payload_attachments,
  )
  from .messaging.actions import (
    _extract_actions,
    _extract_actions_from_tool_calls,
    _extract_reply_text,
  )
  from .messaging.gateway import (
    send_delete_message,
    send_kick_member,
    send_mark_read,
    send_message,
    send_react_message,
    send_sticker,
    send_typing,
    typing_indicator,
  )
except ImportError:  # allow running as `python python/bridge/main.py`
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import (  # type: ignore
    WhatsAppMessage,
    assistant_name,
    assistant_sender_ref,
    assistant_name_pattern,
  )
  from bridge.log import setup_logging, set_chat_log_context, reset_chat_log_context  # type: ignore
  from bridge.llm.llm1 import call_llm1, LLM1Decision  # type: ignore
  from bridge.llm.llm2 import generate_reply  # type: ignore
  from bridge.commands import parse_command, handle_command, CommandResult  # type: ignore
  from bridge.db import (  # type: ignore
    get_mode as db_get_mode,
    get_triggers as db_get_triggers,
    is_muted as db_is_muted,
    is_mute_notified as db_is_mute_notified,
    mark_mute_notified as db_mark_mute_notified,
    add_mute as db_add_mute,
    remove_mute as db_remove_mute,
    clear_mutes as db_clear_mutes,
    get_mute_remaining_minutes as db_get_mute_remaining,
    set_permission as db_set_permission,
  )
  from bridge.dashboard import record_stat, record_user_invoke, flush_to_db, start_flush_loop  # type: ignore
  from bridge.stickers import resolve_sticker  # type: ignore
  from bridge.tools.sticker import create_sticker_file  # type: ignore
  from bridge.messaging.processing import (  # type: ignore
    _append_history,
    _append_or_merge_history_payload,
    _build_burst_current,
    _clean_text,
    _collect_context_ids,
    _extract_send_ack_context_msg_id,
    _hydrate_provisional_context_id_from_ack,
    _infer_media,
    _is_context_only_payload,
    _make_request_id,
    _normalize_context_msg_id,
    _normalize_preview_text,
    _payload_to_message,
    _quoted_preview,
    _reply_signature,
  )
  from bridge.messaging.filtering import (  # type: ignore
    _chat_state_from_payload,
    _message_matches_prefix,
    _payload_has_meaningful_content,
    _payload_triggers_llm1,
  )
  from bridge.llm.metadata import (  # type: ignore
    _build_llm1_context_metadata,
    _resolve_group_prompt_context,
  )
  from bridge.messaging.moderation import (  # type: ignore
    _merge_payload_attachments,
  )
  from bridge.messaging.actions import (  # type: ignore
    _extract_actions,
    _extract_actions_from_tool_calls,
    _extract_reply_text,
  )
  from bridge.messaging.gateway import (  # type: ignore
    send_delete_message,
    send_kick_member,
    send_mark_read,
    send_message,
    send_react_message,
    send_sticker,
    send_typing,
    typing_indicator,
  )

load_dotenv()

try:
  from .config import (
    SLOW_BATCH_LOG_MS,
    MAX_TRIGGER_BATCH_AGE_MS,
    REPLY_DEDUP_WINDOW_MS,
    REPLY_DEDUP_MIN_CHARS,
    ASSISTANT_ECHO_MERGE_WINDOW_MS,
    INCOMING_DEBOUNCE_SECONDS,
    INCOMING_BURST_MAX_SECONDS,
  )
except ImportError:
  from bridge.config import (  # type: ignore
    SLOW_BATCH_LOG_MS,
    MAX_TRIGGER_BATCH_AGE_MS,
    REPLY_DEDUP_WINDOW_MS,
    REPLY_DEDUP_MIN_CHARS,
    ASSISTANT_ECHO_MERGE_WINDOW_MS,
    INCOMING_DEBOUNCE_SECONDS,
    INCOMING_BURST_MAX_SECONDS,
  )

logger = setup_logging()

# Backward compat re-exports (used by tests).
# _quoted_preview, _build_burst_current, and _build_llm1_context_metadata
# are imported above and are available as module-level names.


# ---------------------------------------------------------------------------
# Sticker command helpers
# ---------------------------------------------------------------------------

def _parse_sticker_args(args: str) -> tuple[str | None, str | None]:
  """Parse '/sticker upper#lower' args → (upper_text, lower_text). Either may be None."""
  if "#" in args:
    upper, _, lower = args.partition("#")
    return upper.strip() or None, lower.strip() or None
  return args.strip() or None, None


def _store_media_path(media_paths_by_chat: dict, payload: dict) -> None:
  """Record attachment file path keyed by (chat_id, context_msg_id) for later lookup."""
  ctx_id = payload.get("contextMsgId")
  chat_id = payload.get("chatId")
  atts = payload.get("attachments") or []
  if ctx_id and chat_id and atts and isinstance(atts[0], dict):
    path = atts[0].get("path")
    if path:
      media_paths_by_chat.setdefault(chat_id, {})[ctx_id] = path


def _resolve_sticker_media(
  media_paths_by_chat: dict,
  payload: dict,
  chat_id: str,
) -> str | None:
  """
  Find the media file path to use for sticker creation.
  First checks the current payload's attachments; falls back to the
  quoted message's tracked path (populated when the original image arrived).
  """
  atts = payload.get("attachments") or []
  if atts and isinstance(atts[0], dict):
    path = atts[0].get("path")
    if path and os.path.isfile(path):
      return path
  # Fall back to quoted message's previously tracked path
  quoted = payload.get("quoted") or {}
  quoted_ctx_id = quoted.get("contextMsgId")
  if quoted_ctx_id:
    path = media_paths_by_chat.get(chat_id, {}).get(quoted_ctx_id)
    if path and os.path.isfile(path):
      return path
  return None


def _append_sticker_log_to_history(
  history: deque,
  log_text: str,
) -> None:
  """Append a synthetic assistant entry to the conversation history for sticker creation."""
  history.append(WhatsAppMessage(
    timestamp_ms=int(time.time() * 1000),
    sender="bot",
    text=log_text,
    role="assistant",
  ))


@dataclass
class PendingChat:
  payloads: list[dict] = field(default_factory=list)
  burst_started_at: float | None = None
  last_event_at: float | None = None
  wake_event: asyncio.Event = field(default_factory=asyncio.Event)
  prefix_interrupt: asyncio.Event = field(default_factory=asyncio.Event)
  task: asyncio.Task | None = None
  lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def handle_socket(ws):
  per_chat: Dict[str, Deque[WhatsAppMessage]] = defaultdict(deque)
  per_chat_lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
  pending_by_chat: Dict[str, PendingChat] = defaultdict(PendingChat)
  pending_send_request_chat: Dict[str, str] = {}
  recent_reply_signatures_by_chat: Dict[str, Deque[tuple[int, str]]] = defaultdict(deque)
  media_paths_by_chat: Dict[str, Dict[str, str]] = defaultdict(dict)
  tasks: Set[asyncio.Task] = set()
  logger.info("Gateway connected")

  # Start dashboard flush loop
  dashboard_task = await start_flush_loop()
  tasks.add(dashboard_task)

  def _track_task(task: asyncio.Task) -> None:
    tasks.add(task)
    task.add_done_callback(tasks.discard)

  def _is_duplicate_reply(chat_id: str, text: str | None) -> bool:
    if REPLY_DEDUP_WINDOW_MS <= 0:
      return False

    signature = _reply_signature(text)
    if len(signature) < REPLY_DEDUP_MIN_CHARS:
      return False

    now_ms = int(time.time() * 1000)
    cutoff = now_ms - REPLY_DEDUP_WINDOW_MS
    items = recent_reply_signatures_by_chat[chat_id]
    while items and items[0][0] < cutoff:
      items.popleft()

    if any(prev_sig == signature for _, prev_sig in items):
      return True

    items.append((now_ms, signature))
    while len(items) > 24:
      items.popleft()
    return False

  async def process_message_batch(payloads: list[dict]):
    if not payloads:
      return

    non_empty_payloads = [payload for payload in payloads if _payload_has_meaningful_content(payload)]
    if not non_empty_payloads:
      chat_id = payloads[-1].get("chatId") if payloads else "unknown"
      logger.debug(
        "skipped empty batch",
        extra={
          "chat_id": chat_id,
          "batch_size": len(payloads),
          "message_ids": [p.get("messageId") for p in payloads],
        },
      )
      return

    # --- Slash command handling ---
    # Commands are now handled by Node.js (commandHandler.js).
    # Python only adds commands to history for context and handles /sticker (PIL) and /reset (memory clear).
    remaining_payloads: list[dict] = []
    for payload in non_empty_payloads:
      _store_media_path(media_paths_by_chat, payload)

      slash_cmd = payload.get("slashCommand")
      cmd_handled = bool(payload.get("commandHandled"))

      if not slash_cmd or not isinstance(slash_cmd, dict):
        remaining_payloads.append(payload)
        continue

      cmd_name = slash_cmd.get("command") or ""
      cmd_args = slash_cmd.get("args") or ""
      p_chat_id = payload.get("chatId") or "unknown"
      history = per_chat[p_chat_id]

      # Add command message to history (for LLM context)
      _append_or_merge_history_payload(history, payload)

      # If command already handled by Node.js, skip further processing
      if cmd_handled:
        logger.debug("command %s handled by gateway, skipping", cmd_name, extra={"chat_id": p_chat_id})
        continue

      # Handle /sticker: create meme-style sticker (requires PIL, stays in Python)
      if cmd_name == "sticker":
        p_chat_type, p_bot_is_admin, _ = _chat_state_from_payload(payload)
        upper_text, lower_text = _parse_sticker_args(cmd_args)
        media_path = _resolve_sticker_media(media_paths_by_chat, payload, p_chat_id)
        reply_to = _normalize_context_msg_id(payload.get("contextMsgId"))
        if not media_path:
          await send_message(
            ws, p_chat_id,
            "Send an image with the /sticker caption or reply to an image.",
            reply_to, request_id=_make_request_id("cmd"),
          )
        else:
          try:
            sticker_path = create_sticker_file(media_path, upper_text, lower_text)
            await send_sticker(ws, p_chat_id, sticker_path, reply_to, request_id=_make_request_id("sticker"))
            record_stat(p_chat_id, "stickers_sent")
            log_parts = ["Successfully created sticker"]
            if upper_text:
              log_parts.append(f"upper_text: {upper_text}")
            if lower_text:
              log_parts.append(f"lower_text: {lower_text}")
            if upper_text or lower_text:
              log_parts.append("font_size: 150")
            _append_sticker_log_to_history(history, ", ".join(log_parts))
          except Exception as err:
            logger.exception("sticker creation failed: %s", err, extra={"chat_id": p_chat_id})
            await send_message(
              ws, p_chat_id, f"Failed to create sticker: {err}",
              reply_to, request_id=_make_request_id("cmd"),
            )
        continue

      # Handle /reset: clear memory (Python manages per_chat history)
      if cmd_name == "reset":
        per_chat[p_chat_id].clear()
        logger.info("Memory cleared for chat_id=%s via /reset", p_chat_id)
        continue

      # All other commands are handled by Node.js, just skip
      logger.debug("command %s not handled in Python, skipping", cmd_name, extra={"chat_id": p_chat_id})
      continue

    non_empty_payloads = remaining_payloads
    if not non_empty_payloads:
      return

    context_only_payloads = [payload for payload in non_empty_payloads if _is_context_only_payload(payload)]
    trigger_indexes = [
      idx for idx, payload in enumerate(non_empty_payloads) if _payload_triggers_llm1(payload)
    ]
    llm1_trigger_payloads = [non_empty_payloads[idx] for idx in trigger_indexes]
    last_trigger_index = trigger_indexes[-1] if trigger_indexes else None
    passive_context_payloads = [
      payload for payload in context_only_payloads if not _payload_triggers_llm1(payload)
    ]

    last_payload = llm1_trigger_payloads[-1] if llm1_trigger_payloads else non_empty_payloads[-1]
    chat_id = last_payload["chatId"]
    history = per_chat[chat_id]
    lock = per_chat_lock[chat_id]
    batch_started = time.perf_counter()
    last_payload_ts = None
    raw_ts = last_payload.get("timestampMs")
    try:
      ts = int(raw_ts)
      last_payload_ts = ts if ts > 0 else None
    except (TypeError, ValueError):
      last_payload_ts = None
    lock_wait_started = time.perf_counter()
    async with lock:
      lock_wait_ms = int((time.perf_counter() - lock_wait_started) * 1000)
      llm1_ms = 0
      llm2_ms = 0
      action_send_ms = 0

      def _log_slow_batch(outcome: str, *, action_counts: dict | None = None, action_total: int = 0):
        total_ms = int((time.perf_counter() - batch_started) * 1000)
        if total_ms < SLOW_BATCH_LOG_MS and lock_wait_ms < SLOW_BATCH_LOG_MS:
          return
        payload_age_ms = None
        if last_payload_ts is not None:
          payload_age_ms = max(0, int(time.time() * 1000) - last_payload_ts)
        logger.info(
          "slow batch observed",
          extra={
            "chat_id": chat_id,
            "outcome": outcome,
            "batch_size": len(payloads),
            "non_empty_batch_size": len(non_empty_payloads),
            "llm1_trigger_batch_size": len(llm1_trigger_payloads),
            "context_only_batch_size": len(context_only_payloads),
            "passive_context_batch_size": len(passive_context_payloads),
            "lock_wait_ms": lock_wait_ms,
            "llm1_ms": llm1_ms,
            "llm2_ms": llm2_ms,
            "action_send_ms": action_send_ms,
            "total_ms": total_ms,
            "payload_age_ms": payload_age_ms,
            "history_len": len(history),
            "last_message_id": last_payload.get("messageId"),
            "last_type": last_payload.get("messageType"),
            "last_sender": last_payload.get("senderName") or last_payload.get("senderId"),
            "action_counts": action_counts,
            "action_total": action_total,
          },
        )

      try:
        logger.debug(
          "incoming_batch",
          extra={
            "chat_id": chat_id,
            "batch_size": len(payloads),
            "non_empty_batch_size": len(non_empty_payloads),
            "llm1_trigger_batch_size": len(llm1_trigger_payloads),
            "context_only_batch_size": len(context_only_payloads),
            "passive_context_batch_size": len(passive_context_payloads),
            "message_ids": [p.get("messageId") for p in non_empty_payloads],
            "last_message_id": last_payload.get("messageId"),
            "type": last_payload.get("messageType"),
            "text": last_payload.get("text"),
            "attachments": len(last_payload.get("attachments") or []),
            "quoted": bool(last_payload.get("quoted")),
            "location": bool(last_payload.get("location")),
            "history_len": len(history),
            "sender": last_payload.get("senderName") or last_payload.get("senderId"),
            "raw_payload": last_payload,
          },
        )
        if not llm1_trigger_payloads:
          for payload in non_empty_payloads:
            _append_or_merge_history_payload(history, payload)
          logger.debug("stored context-only updates", extra={"chat_id": chat_id})
          _log_slow_batch("context_only")
          return

        trigger_window_payloads = non_empty_payloads[: (last_trigger_index + 1)]
        prefix_payloads = trigger_window_payloads[:-1]
        passive_prefix_payloads = [
          payload for payload in prefix_payloads if not _payload_triggers_llm1(payload)
        ]

        history_before_current = list(history)
        current = _build_burst_current(llm1_trigger_payloads)
        llm1_history = list(history_before_current)
        # LLM1 should evaluate the pending window as a single "current" burst
        # so one trailing sticker does not overshadow earlier questions.
        llm1_current = _build_burst_current(trigger_window_payloads)
        llm2_history = list(history_before_current)
        llm2_history.extend(_payload_to_message(payload) for payload in passive_prefix_payloads)
        batch_payload_age_ms = None
        if last_payload_ts is not None:
          batch_payload_age_ms = max(0, int(time.time() * 1000) - last_payload_ts)
        if (
          MAX_TRIGGER_BATCH_AGE_MS > 0
          and batch_payload_age_ms is not None
          and batch_payload_age_ms > MAX_TRIGGER_BATCH_AGE_MS
        ):
          for payload in non_empty_payloads:
            _append_or_merge_history_payload(history, payload)
          logger.info(
            "skipped stale trigger batch",
            extra={
              "chat_id": chat_id,
              "payload_age_ms": batch_payload_age_ms,
              "max_trigger_batch_age_ms": MAX_TRIGGER_BATCH_AGE_MS,
              "trigger_batch_size": len(llm1_trigger_payloads),
            },
          )
          _log_slow_batch("stale_skip")
          return
        group_description, db_prompt = _resolve_group_prompt_context(last_payload)
        chat_type, bot_is_admin, bot_is_super_admin = _chat_state_from_payload(last_payload)
        llm_context_metadata = _build_llm1_context_metadata(
          history_before_current,
          trigger_window_payloads,
        )
        llm1_payload = dict(last_payload)
        llm1_payload.update(llm_context_metadata)

        # --- Dashboard: record messages processed ---
        for _dp in llm1_trigger_payloads:
          record_stat(chat_id, "messages_processed")
          if bool(_dp.get("botMentioned")):
            record_stat(chat_id, "bot_tags")
          _dp_text = _clean_text(_dp.get("text"))
          if _dp_text and assistant_name_pattern().search(_dp_text):
            record_stat(chat_id, "bot_name_mentions")

        # --- Mode-aware LLM1 decision ---
        chat_mode = db_get_mode(chat_id) if chat_type == "group" else "auto"
        triggers = db_get_triggers(chat_id) if chat_mode in ("prefix", "hybrid") else set()

        if chat_type == "private":
          decision = LLM1Decision(
            should_response=True,
            confidence=100,
            reason="Private chat: always respond to direct messages.",
          )
          llm1_ms = 0
          logger.info("private chat; skipping LLM1", extra={"chat_id": chat_id})
        elif chat_mode == "prefix":
          # Prefix mode: check if any trigger payload matches prefix
          prefix_matched_payloads = [p for p in llm1_trigger_payloads if _message_matches_prefix(p, triggers)]
          if not prefix_matched_payloads:
            # No prefix match — store history and skip
            for payload in non_empty_payloads:
              _append_or_merge_history_payload(history, payload)
            logger.info(
              "prefix mode: no match; skipping",
              extra={"chat_id": chat_id, "triggers": sorted(triggers), "batch_size": len(llm1_trigger_payloads)},
            )
            _log_slow_batch("prefix_no_match")
            return
          # Prefix matched — skip LLM1, go straight to LLM2
          decision = LLM1Decision(
            should_response=True,
            confidence=100,
            reason="Prefix mode: bot was explicitly invoked.",
          )
          llm1_ms = 0
          # Record invoking user for dashboard
          for _pp in prefix_matched_payloads:
            _pp_ref = _clean_text(_pp.get("senderRef"))
            _pp_name = _clean_text(_pp.get("senderName"))
            if _pp_ref:
              record_user_invoke(chat_id, _pp_ref, _pp_name)
          logger.info(
            "prefix mode: matched %d/%d payloads; skipping LLM1",
            len(prefix_matched_payloads), len(llm1_trigger_payloads),
            extra={"chat_id": chat_id, "triggers": sorted(triggers)},
          )
        elif chat_mode == "hybrid":
          # Hybrid mode: check prefix triggers first, fall back to auto (LLM1)
          prefix_matched_payloads = [p for p in llm1_trigger_payloads if _message_matches_prefix(p, triggers)]
          if prefix_matched_payloads:
            # Prefix matched in current batch — skip LLM1, go straight to LLM2
            decision = LLM1Decision(
              should_response=True,
              confidence=100,
              reason="Hybrid mode: bot was explicitly invoked (prefix trigger in batch).",
            )
            llm1_ms = 0
            for _pp in prefix_matched_payloads:
              _pp_ref = _clean_text(_pp.get("senderRef"))
              _pp_name = _clean_text(_pp.get("senderName"))
              if _pp_ref:
                record_user_invoke(chat_id, _pp_ref, _pp_name)
            logger.info(
              "hybrid mode: prefix matched %d/%d payloads; skipping LLM1",
              len(prefix_matched_payloads), len(llm1_trigger_payloads),
              extra={"chat_id": chat_id, "triggers": sorted(triggers)},
            )
          else:
            # No prefix match in batch — run LLM1 with cancellation support
            pending = pending_by_chat[chat_id]
            pending.prefix_interrupt.clear()
            llm1_started = time.perf_counter()

            llm1_task = asyncio.create_task(call_llm1(
              llm1_history,
              llm1_current,
              current_payload=llm1_payload,
              group_description=group_description,
              prompt_override=db_prompt,
            ))
            interrupt_wait = asyncio.create_task(pending.prefix_interrupt.wait())

            done, _pending_tasks = await asyncio.wait(
              {llm1_task, interrupt_wait},
              return_when=asyncio.FIRST_COMPLETED,
            )

            if interrupt_wait in done:
              # Prefix trigger arrived while LLM1 was running — cancel LLM1
              llm1_task.cancel()
              try:
                await llm1_task
              except (asyncio.CancelledError, Exception):
                pass
              llm1_ms = int((time.perf_counter() - llm1_started) * 1000)

              # Drain new prefix-trigger payloads from pending
              async with pending.lock:
                new_payloads = list(pending.payloads)
                pending.payloads.clear()
                pending.burst_started_at = None
                pending.last_event_at = None
                pending.prefix_interrupt.clear()

              if new_payloads:
                # Merge new payloads into current batch for LLM2
                non_empty_payloads.extend(new_payloads)
                new_trigger_payloads = [p for p in new_payloads if _payload_triggers_llm1(p)]
                llm1_trigger_payloads.extend(new_trigger_payloads)
                # Rebuild burst context for LLM2 with merged payloads
                trigger_window_payloads = list(non_empty_payloads)
                current = _build_burst_current(llm1_trigger_payloads)

              decision = LLM1Decision(
                should_response=True,
                confidence=100,
                reason="Hybrid mode: prefix trigger interrupted LLM1; responding immediately.",
              )
              # Record invoking users from new payloads
              for _np in new_payloads:
                if _message_matches_prefix(_np, triggers):
                  _np_ref = _clean_text(_np.get("senderRef"))
                  _np_name = _clean_text(_np.get("senderName"))
                  if _np_ref:
                    record_user_invoke(chat_id, _np_ref, _np_name)
              logger.info(
                "hybrid mode: prefix trigger interrupted LLM1 after %dms; merged %d new payloads",
                llm1_ms, len(new_payloads),
                extra={"chat_id": chat_id, "triggers": sorted(triggers)},
              )
            else:
              # LLM1 finished before any prefix interrupt
              interrupt_wait.cancel()
              try:
                await interrupt_wait
              except (asyncio.CancelledError, Exception):
                pass
              decision = llm1_task.result()
              llm1_ms = int((time.perf_counter() - llm1_started) * 1000)
              record_stat(chat_id, "llm1_calls")
              if decision.input_tokens:
                record_stat(chat_id, "llm1_input_tokens", decision.input_tokens)
              if decision.output_tokens:
                record_stat(chat_id, "llm1_output_tokens", decision.output_tokens)
              if decision.should_response:
                for _ap in llm1_trigger_payloads:
                  _ap_ref = _clean_text(_ap.get("senderRef"))
                  _ap_name = _clean_text(_ap.get("senderName"))
                  if _ap_ref:
                    record_user_invoke(chat_id, _ap_ref, _ap_name)
              logger.info(
                "hybrid mode: LLM1 completed in %dms (no prefix interrupt); should_response=%s",
                llm1_ms, decision.should_response,
                extra={"chat_id": chat_id, "confidence": decision.confidence},
              )
        else:
          llm1_started = time.perf_counter()
          decision = await call_llm1(
            llm1_history,
            llm1_current,
            current_payload=llm1_payload,
            group_description=group_description,
            prompt_override=db_prompt,
          )
          llm1_ms = int((time.perf_counter() - llm1_started) * 1000)
          record_stat(chat_id, "llm1_calls")
          if decision.input_tokens:
            record_stat(chat_id, "llm1_input_tokens", decision.input_tokens)
          if decision.output_tokens:
            record_stat(chat_id, "llm1_output_tokens", decision.output_tokens)
          # Record invoking user for auto mode too
          if decision.should_response:
            for _ap in llm1_trigger_payloads:
              _ap_ref = _clean_text(_ap.get("senderRef"))
              _ap_name = _clean_text(_ap.get("senderName"))
              if _ap_ref:
                record_user_invoke(chat_id, _ap_ref, _ap_name)

        # Send read receipt after LLM1 processes (regardless of decision)
        for _p in trigger_window_payloads:
          _msg_id = _p.get("messageId")
          _participant = _p.get("senderId") if _p.get("isGroup") else None
          await send_mark_read(ws, chat_id, _msg_id, _participant)

        for payload in non_empty_payloads:
          _append_or_merge_history_payload(history, payload)
        # Handle express decision from LLM1 (skip LLM2 entirely)
        if decision.react_expression and decision.react_context_msg_id:
          sticker_path = resolve_sticker(decision.react_expression)
          if sticker_path:
            logger.info(
              "llm1 express; sending sticker directly (skipping llm2)",
              extra={
                "chat_id": chat_id,
                "sticker_name": decision.react_expression,
                "react_context_msg_id": decision.react_context_msg_id,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "llm1_ms": llm1_ms,
              },
            )
            await send_sticker(
              ws,
              chat_id,
              sticker_path,
              decision.react_context_msg_id,
              request_id=_make_request_id("sticker"),
            )
            record_stat(chat_id, "stickers_sent")
          else:
            logger.info(
              "llm1 express; sending emoji react directly (skipping llm2)",
              extra={
                "chat_id": chat_id,
                "react_expression": decision.react_expression,
                "react_context_msg_id": decision.react_context_msg_id,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "llm1_ms": llm1_ms,
              },
            )
            await send_react_message(
              ws,
              chat_id,
              decision.react_context_msg_id,
              decision.react_expression,
              request_id=_make_request_id("react"),
            )
          _log_slow_batch("llm1_express")
          return

        if not decision.should_response:
          logger.info(
            "llm1 skip; no response sent",
            extra={"chat_id": chat_id},
          )
          _log_slow_batch("llm1_skip")
          return

        allowed_context_ids = _collect_context_ids(history)
        fallback_reply_to = _normalize_context_msg_id(last_payload.get("contextMsgId"))
        llm2_payload = _merge_payload_attachments(trigger_window_payloads, last_payload)
        llm2_payload.update(llm_context_metadata)
        llm2_payload.update(
          {
            "llm1ShouldResponse": decision.should_response,
            "llm1Confidence": decision.confidence,
            "llm1Reason": " ".join((decision.reason or "").split()),
          }
        )

        # Keep typing indicator alive while LLM2 generates (refreshes every 8s)
        llm2_started = time.perf_counter()
        async with typing_indicator(ws, chat_id):
          def _validate_llm2_result(result) -> bool:
            """Return True if the LLM2 output contains at least one usable action."""
            tool_calls = getattr(result, 'tool_calls', None) or []
            if tool_calls:
              test_actions = _extract_actions_from_tool_calls(
                tool_calls,
                fallback_reply_to=fallback_reply_to,
                allowed_context_ids=allowed_context_ids,
              )
              return len(test_actions) > 0
            # Fallback: parse text content (legacy)
            test_actions = _extract_actions(
              result,
              fallback_reply_to=fallback_reply_to,
              allowed_context_ids=allowed_context_ids,
            )
            return len(test_actions) > 0

          reply_msg = await generate_reply(
            llm2_history,
            current,
            current_payload=llm2_payload,
            group_description=group_description,
            prompt_override=db_prompt,
            chat_type=chat_type,
            bot_is_admin=bot_is_admin,
            bot_is_super_admin=bot_is_super_admin,
            result_validator=_validate_llm2_result,
          )

        llm2_ms = int((time.perf_counter() - llm2_started) * 1000)
        record_stat(chat_id, "llm2_calls")
        # Track LLM2 token usage if available
        if reply_msg is not None:
          _usage = getattr(reply_msg, "usage_metadata", None)
          if isinstance(_usage, dict):
            _in_tok = _usage.get("input_tokens", 0)
            _out_tok = _usage.get("output_tokens", 0)
            if _in_tok:
              record_stat(chat_id, "llm2_input_tokens", _in_tok)
            if _out_tok:
              record_stat(chat_id, "llm2_output_tokens", _out_tok)
        if reply_msg is None:
          record_stat(chat_id, "errors")
          logger.warning("llm2 failed to produce reply", extra={"chat_id": chat_id})
          _log_slow_batch("llm2_none")
          return
        tool_calls = getattr(reply_msg, 'tool_calls', None) or []
        if tool_calls:
          actions = _extract_actions_from_tool_calls(
            tool_calls,
            fallback_reply_to=fallback_reply_to,
            allowed_context_ids=allowed_context_ids,
          )
        else:
          # Fallback: parse text content (legacy)
          actions = _extract_actions(
            reply_msg,
            fallback_reply_to=fallback_reply_to,
            allowed_context_ids=allowed_context_ids,
          )
        if not actions:
          logger.warning(
            "llm2 returned no executable action",
            extra={
              "chat_id": chat_id,
              "reply_preview": _extract_reply_text(reply_msg),
              "fallback_reply_to": fallback_reply_to,
              "tool_calls": len(tool_calls),
            },
          )
          _log_slow_batch("no_action")
          return

        action_counts: dict[str, int] = defaultdict(int)
        action_send_started = time.perf_counter()
        for action in actions:
          action_type = action.get("type")
          if action_type == "send_message":
            action_text = action.get("text") or ""
            if _is_duplicate_reply(chat_id, action_text):
              logger.info(
                "dropped duplicate reply",
                extra={
                  "chat_id": chat_id,
                  "reply_preview": _normalize_preview_text(action_text, limit=180),
                  "reply_dedup_window_ms": REPLY_DEDUP_WINDOW_MS,
                },
              )
              continue
            request_id = _make_request_id("send")
            await send_message(
              ws,
              chat_id,
              action_text,
              action.get("replyTo"),
              request_id=request_id,
            )
            record_stat(chat_id, "responses_sent")
            pending_send_request_chat[request_id] = chat_id
            if len(pending_send_request_chat) > 4096:
              pending_send_request_chat.pop(next(iter(pending_send_request_chat)))
            _append_history(
              history,
              WhatsAppMessage(
                timestamp_ms=int(time.time() * 1000),
                sender=assistant_name(),
                context_msg_id="pending",
                sender_ref=assistant_sender_ref(),
                sender_is_admin=False,
                text=action_text or None,
                media=None,
                quoted_message_id=_normalize_context_msg_id(action.get("replyTo")),
                quoted_sender=None,
                quoted_text=None,
                quoted_media=None,
                message_id=f"local-send-{request_id}",
                role="assistant",
              ),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "delete_message":
            await send_delete_message(
              ws,
              chat_id,
              action.get("contextMsgId"),
              request_id=_make_request_id("delete"),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "kick_member":
            await send_kick_member(
              ws,
              chat_id,
              action.get("targets") or [],
              request_id=_make_request_id("kick"),
              mode=action.get("mode") or "partial_success",
              auto_reply_anchor=bool(action.get("autoReplyAnchor", False)),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "react_message":
            await send_react_message(
              ws,
              chat_id,
              action.get("contextMsgId"),
              action.get("emoji"),
              request_id=_make_request_id("react"),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "send_sticker":
            sticker_name = action.get("stickerName", "")
            sticker_path = resolve_sticker(sticker_name)
            if sticker_path:
              request_id = _make_request_id("sticker")
              await send_sticker(
                ws,
                chat_id,
                sticker_path,
                action.get("replyTo"),
                request_id=request_id,
              )
              record_stat(chat_id, "stickers_sent")
              action_counts[action_type] += 1
            else:
              logger.warning(
                "sticker not found: %s",
                sticker_name,
                extra={"chat_id": chat_id},
              )
            continue
          if action_type == "create_sticker":
            ctx_id = action.get("contextMsgId")
            media_path = media_paths_by_chat.get(chat_id, {}).get(ctx_id) if ctx_id else None
            if not media_path:
              logger.warning(
                "create_sticker: no media for context_msg_id=%s",
                ctx_id,
                extra={"chat_id": chat_id},
              )
              continue
            try:
              upper = action.get("upperText") or None
              lower = action.get("lowerText") or None
              fsize = int(action.get("fontSize") or 150)
              sticker_path = create_sticker_file(media_path, upper, lower, fsize)
              await send_sticker(
                ws,
                chat_id,
                sticker_path,
                action.get("replyTo"),
                request_id=_make_request_id("sticker"),
              )
              record_stat(chat_id, "stickers_sent")
              action_counts[action_type] += 1
              log_parts = ["Successfully created sticker"]
              if upper:
                log_parts.append(f"upper_text: {upper}")
              if lower:
                log_parts.append(f"lower_text: {lower}")
              if upper or lower:
                log_parts.append(f"font_size: {fsize}")
              _append_sticker_log_to_history(history, ", ".join(log_parts))
            except Exception as err:
              logger.exception(
                "create_sticker action failed: %s", err,
                extra={"chat_id": chat_id},
              )
            continue
          if action_type == "mute_member":
            sender_ref = action.get("senderRef", "")
            anchor_id = action.get("anchorContextMsgId")
            duration = action.get("durationMinutes", 30)
            if duration == 0:
              db_remove_mute(chat_id, sender_ref)
            else:
              db_add_mute(chat_id, sender_ref, duration)
              # Delete the anchor message immediately
              if anchor_id:
                await send_delete_message(
                  ws,
                  chat_id,
                  anchor_id,
                  request_id=_make_request_id("mute_del"),
                )
            action_counts[action_type] += 1
            continue
          logger.warning(
            "unknown action type from parser: %s",
            action_type,
            extra={"chat_id": chat_id},
          )
        action_send_ms = int((time.perf_counter() - action_send_started) * 1000)
        logger.info(
          "executed actions",
          extra={
            "chat_id": chat_id,
            "action_counts": action_counts,
            "batch_size": len(llm1_trigger_payloads),
            "action_total": len(actions),
          },
        )
        _log_slow_batch("actions_executed", action_counts=action_counts, action_total=len(actions))
      except Exception as err:
        _log_slow_batch("handler_error")
        logger.exception("handler error: %s", err, extra={"chat_id": chat_id})

  async def flush_pending(chat_id: str):
    pending = pending_by_chat[chat_id]
    while True:
      async with pending.lock:
        if not pending.payloads:
          pending.task = None
          return

        now = time.monotonic()
        last_event_at = pending.last_event_at or now
        burst_started_at = pending.burst_started_at or now

        # Skip debounce for private chats and prefix/hybrid mode matches.
        _skip_debounce = False
        _last_p = pending.payloads[-1] if pending.payloads else {}
        _flush_chat_type, _, _ = _chat_state_from_payload(_last_p)
        if _flush_chat_type == "private":
          _skip_debounce = True
        elif db_get_mode(chat_id) in ("prefix", "hybrid"):
          _flush_triggers = db_get_triggers(chat_id)
          for _fp in pending.payloads:
            if _message_matches_prefix(_fp, _flush_triggers):
              _skip_debounce = True
              break

        if _skip_debounce:
          timeout_s = 0.0
        else:
          quiet_deadline = last_event_at + INCOMING_DEBOUNCE_SECONDS
          hard_deadline = burst_started_at + INCOMING_BURST_MAX_SECONDS
          timeout_s = max(0.0, min(quiet_deadline, hard_deadline) - now)
        pending.wake_event.clear()
        wake_event = pending.wake_event

      try:
        await asyncio.wait_for(wake_event.wait(), timeout=timeout_s)
        continue
      except asyncio.TimeoutError:
        pass

      async with pending.lock:
        payloads = list(pending.payloads)
        pending.payloads.clear()
        pending.burst_started_at = None
        pending.last_event_at = None

      if payloads:
        context_payload = payloads[-1] if payloads else {}
        context_chat_type, _, _ = _chat_state_from_payload(context_payload)
        context_chat_name = _clean_text(context_payload.get("chatName")) if context_chat_type == "group" else None
        context_token = set_chat_log_context(
          chat_id=_clean_text(context_payload.get("chatId")) or None,
          chat_name=context_chat_name or None,
        )
        try:
          await process_message_batch(payloads)
        finally:
          reset_chat_log_context(context_token)
      # Keep the same worker task alive so new payloads for the same chat
      # are drained sequentially without spawning extra waiters.

  try:
    async for raw in ws:
      try:
        event = json.loads(raw)
      except json.JSONDecodeError:
        logger.warning("Dropping non-JSON payload")
        continue

      event_type = event.get("type")

      if event_type == "hello":
        logger.info("Handshake: %s", event.get("payload"))
        continue

      if event_type in {"send_ack", "action_ack"}:
        payload = event.get("payload")
        if (
          event_type == "action_ack"
          and isinstance(payload, dict)
          and str(payload.get("action") or "") == "send_message"
        ):
          request_id = _clean_text(payload.get("requestId"))
          chat_id_for_request = pending_send_request_chat.pop(request_id, None)
          if request_id and chat_id_for_request:
            context_msg_id = _extract_send_ack_context_msg_id(payload)
            if context_msg_id:
              history = per_chat[chat_id_for_request]
              lock = per_chat_lock[chat_id_for_request]
              async with lock:
                updated = _hydrate_provisional_context_id_from_ack(
                  history,
                  request_id=request_id,
                  context_msg_id=context_msg_id,
                )
              if updated:
                logger.debug(
                  "hydrated provisional send context id from action_ack",
                  extra={
                    "chat_id": chat_id_for_request,
                    "request_id": request_id,
                    "context_msg_id": context_msg_id,
                  },
                )
              else:
                logger.debug(
                  "action_ack arrived but provisional send not found",
                  extra={
                    "chat_id": chat_id_for_request,
                    "request_id": request_id,
                    "context_msg_id": context_msg_id,
                  },
                )
        logger.debug("Gateway ack: %s", event.get("payload"))
        continue

      if event_type == "error":
        logger.warning("Gateway error: %s", event.get("payload"))
        continue

      # Handle clear_history message from Node.js (after /reset)
      if event_type == "clear_history":
        clear_chat_id = event.get("chatId")
        if clear_chat_id and clear_chat_id in per_chat:
          per_chat[clear_chat_id].clear()
          logger.info("History cleared for chat_id=%s via clear_history message", clear_chat_id)
        continue

      if event_type != "incoming_message":
        continue

      payload = event["payload"]
      chat_id = payload.get("chatId")
      if not chat_id:
        logger.warning("Dropping incoming_message without chatId")
        continue

      # --- Mute enforcement (before debounce, instant) ---
      _mute_sender_ref = (payload.get("senderRef") or "").strip().lower()
      _mute_context_only = bool(payload.get("contextOnly"))
      _mute_msg_type = str(payload.get("messageType") or "").strip().lower()
      if (
        _mute_sender_ref
        and not _mute_context_only
        and _mute_msg_type not in ("groupparticipantsupdate", "actionlog", "botrolechange")
        and db_is_muted(chat_id, _mute_sender_ref)
      ):
        _mute_ctx_id = payload.get("contextMsgId")
        if _mute_ctx_id:
          await send_delete_message(
            ws,
            chat_id,
            _mute_ctx_id,
            request_id=_make_request_id("mute_enforce"),
          )
        # First-delete notification
        if not db_is_mute_notified(chat_id, _mute_sender_ref):
          db_mark_mute_notified(chat_id, _mute_sender_ref)
          _remaining = db_get_mute_remaining(chat_id, _mute_sender_ref)
          _mute_name = payload.get("senderName") or _mute_sender_ref
          await send_message(
            ws,
            chat_id,
            f"Pesan dari {_mute_name} dihapus (muted, {_remaining}m tersisa).",
            None,
            request_id=_make_request_id("mute_notify"),
          )
        logger.debug(
          "mute enforcement: deleted message from muted user",
          extra={"chat_id": chat_id, "sender_ref": _mute_sender_ref},
        )
        continue  # skip all further processing

      # --- Bot role change (promote/demote) handling ---
      if _mute_msg_type == "botrolechange":
        _role_action = (payload.get("groupEvent") or {}).get("action", "")
        if _role_action == "promote":
          await send_message(
            ws,
            chat_id,
            "Bot is now an admin! Moderation features (/permission) can now be enabled by group admins.",
            None,
            request_id=_make_request_id("role_notify"),
          )
          logger.info("bot promoted in chat_id=%s", chat_id)
        elif _role_action == "demote":
          db_set_permission(chat_id, 0)
          db_clear_mutes(chat_id)
          await send_message(
            ws,
            chat_id,
            "Bot is no longer an admin. Moderation permissions have been reset to 0 (disabled).",
            None,
            request_id=_make_request_id("role_notify"),
          )
          logger.info("bot demoted in chat_id=%s; permission reset to 0", chat_id)
        continue

      pending = pending_by_chat[chat_id]
      now = time.monotonic()
      async with pending.lock:
        if pending.burst_started_at is None:
          pending.burst_started_at = now
        pending.last_event_at = now
        pending.payloads.append(payload)
        # Signal hybrid mode: if a prefix trigger arrives while LLM1 is running
        if db_get_mode(chat_id) == "hybrid":
          _hybrid_triggers = db_get_triggers(chat_id)
          if _message_matches_prefix(payload, _hybrid_triggers):
            pending.prefix_interrupt.set()
        if pending.task is None or pending.task.done():
          task = asyncio.create_task(flush_pending(chat_id))
          pending.task = task
          _track_task(task)
        else:
          pending.wake_event.set()
  except websockets.ConnectionClosed:
    logger.info("Gateway disconnected")
  finally:
    # Flush dashboard stats before shutting down
    flush_to_db()
    for task in tasks:
      task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)


def _parse_endpoint(url: str):
  parsed = urlsplit(url)
  host = parsed.hostname or "0.0.0.0"
  port = parsed.port or (443 if parsed.scheme == "wss" else 8080)
  return host, port


async def main():
  endpoint = os.getenv("LLM_WS_ENDPOINT", "ws://0.0.0.0:8080/ws")
  host, port = _parse_endpoint(endpoint)
  logger.info("Listening for gateway on %s (host=%s port=%s)", endpoint, host, port)
  server = await websockets.serve(
    handle_socket,
    host=host,
    port=port,
    max_size=20 * 1024 * 1024,
    ping_interval=20,
    ping_timeout=20,
  )
  await server.wait_closed()


if __name__ == "__main__":
  asyncio.run(main())
