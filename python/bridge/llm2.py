from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
  from .history import WhatsAppMessage, format_history
  from .log import setup_logging, trunc, dump_json, env_flag
  from .media import build_visual_parts, llm2_media_enabled, redact_multimodal_content
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage, format_history  # type: ignore
  from bridge.log import setup_logging, trunc, dump_json, env_flag  # type: ignore
  from bridge.media import build_visual_parts, llm2_media_enabled, redact_multimodal_content  # type: ignore

logger = setup_logging()
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "systemprompt.txt"
_SYSTEM_PROMPT_CACHE: str | None = None


@dataclass(frozen=True)
class LLM2Target:
  name: str
  model: str
  base_url: str | None
  api_key: str


def _parse_positive_float(raw: str | None, default: float) -> float:
  if raw is None:
    return default
  try:
    parsed = float(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


def _parse_non_negative_int(raw: str | None, default: int) -> int:
  if raw is None:
    return default
  try:
    parsed = int(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed >= 0 else default


def _llm2_timeout() -> float:
  return _parse_positive_float(os.getenv("LLM2_TIMEOUT"), 20.0)


def _llm2_retry_max() -> int:
  return _parse_non_negative_int(os.getenv("LLM2_RETRY_MAX"), 0)


def _llm2_retry_backoff_seconds() -> float:
  return _parse_positive_float(os.getenv("LLM2_RETRY_BACKOFF_SECONDS"), 0.8)


def _llm2_sdk_max_retries() -> int:
  return _parse_non_negative_int(os.getenv("LLM2_SDK_MAX_RETRIES"), 0)


def _llm2_reasoning_effort() -> str | None:
  raw = os.getenv("LLM2_REASONING_EFFORT")
  if raw is None:
    return "medium"
  cleaned = raw.strip().lower()
  if not cleaned:
    return None
  return cleaned


def _clean_env(raw: str | None) -> str | None:
  if raw is None:
    return None
  cleaned = raw.strip()
  return cleaned or None


def _llm2_targets() -> list[LLM2Target]:
  primary_model = _clean_env(os.getenv("LLM2_MODEL")) or "gpt-4.1"
  primary_endpoint = _clean_env(os.getenv("LLM2_ENDPOINT"))
  primary_api_key = _clean_env(os.getenv("LLM2_API_KEY")) or ""

  targets = [
    LLM2Target(
      name="primary",
      model=primary_model,
      base_url=primary_endpoint,
      api_key=primary_api_key,
    )
  ]

  fallback_model_raw = _clean_env(os.getenv("LLM2_FALLBACK_MODEL"))
  fallback_endpoint_raw = _clean_env(os.getenv("LLM2_FALLBACK_ENDPOINT"))
  fallback_api_key_raw = _clean_env(os.getenv("LLM2_FALLBACK_API_KEY"))
  fallback_enabled = any((fallback_model_raw, fallback_endpoint_raw, fallback_api_key_raw))
  if not fallback_enabled:
    return targets

  fallback_target = LLM2Target(
    name="fallback",
    model=fallback_model_raw or primary_model,
    base_url=fallback_endpoint_raw if fallback_endpoint_raw is not None else primary_endpoint,
    api_key=fallback_api_key_raw if fallback_api_key_raw is not None else primary_api_key,
  )

  primary_target = targets[0]
  if (
    fallback_target.model == primary_target.model
    and fallback_target.base_url == primary_target.base_url
    and fallback_target.api_key == primary_target.api_key
  ):
    return targets

  targets.append(fallback_target)
  return targets


def _is_timeout_error(err: Exception) -> bool:
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < 8:
    if "timeout" in type(current).__name__.lower():
      return True
    current = current.__cause__ or current.__context__
    depth += 1
  return False


def _error_chain(err: Exception, limit: int = 8) -> list[str]:
  chain: list[str] = []
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < limit:
    chain.append(type(current).__name__)
    current = current.__cause__ or current.__context__
    depth += 1
  return chain


def _load_system_prompt() -> str:
  global _SYSTEM_PROMPT_CACHE
  if _SYSTEM_PROMPT_CACHE is not None:
    return _SYSTEM_PROMPT_CACHE
  text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
  _SYSTEM_PROMPT_CACHE = text
  return _SYSTEM_PROMPT_CACHE


def _render_system_prompt(
  base_system: str,
  *,
  prompt_override: str | None = None,
) -> str:
  overide_text = (prompt_override or "").strip()
  return (
    base_system
    .replace("{{prompt_override}}", overide_text)
    .replace("{{ prompt_override }}", overide_text)
  )


def _group_description_block(group_description: str | None) -> str:
  cleaned = (group_description or "").strip()
  if cleaned:
    return cleaned
  return "(none)"


def _format_current_window(msg: WhatsAppMessage) -> str:
  # Burst windows are already serialized as multi-line chat entries.
  text = (msg.text or "").strip()
  if text.startswith("Burst messages ("):
    return text
  return format_history([msg])


def _normalize_chat_type(chat_type: str | None) -> str:
  lowered = (chat_type or "").strip().lower()
  if lowered in {"private", "group"}:
    return lowered
  return "private"


def _chat_state_header(chat_type: str, bot_is_admin: bool, bot_is_super_admin: bool) -> str:
  normalized_type = _normalize_chat_type(chat_type)
  if normalized_type == "group":
    scope_line = "This is a group chat. You're in a chat with multiple people at once."
  else:
    scope_line = "This is a private chat. You're directly chatting with one other person."
  if bot_is_super_admin:
    role_line = "Bot is a super admin (owner)."
  elif bot_is_admin:
    role_line = "Bot is an admin."
  else:
    role_line = "Bot is a normal member."
  return (
    f"{scope_line}\n"
    f"{role_line}"
  )


def _context_injection_block(
  current_payload: dict | None,
  *,
  chat_type: str,
  bot_is_admin: bool,
  bot_is_super_admin: bool,
) -> str:
  payload = current_payload if isinstance(current_payload, dict) else {}
  bot_mentioned = bool(payload.get("botMentionedInWindow", payload.get("botMentioned")))
  replied_to_bot = bool(payload.get("repliedToBotInWindow", payload.get("repliedToBot")))
  mention_count = payload.get("botMentionCountInWindow")
  if mention_count is None:
    mentioned = payload.get("mentionedJids")
    if isinstance(mentioned, list):
      mention_count = len(mentioned)
    elif payload.get("botMentioned") is not None:
      mention_count = 1 if bool(payload.get("botMentioned")) else 0
    else:
      mention_count = None
  since_assistant = payload.get("messagesSinceAssistantReply")
  assistant_replies_by_window = payload.get("assistantRepliesByWindow")
  human_window = payload.get("humanMessagesInWindow")
  explicit_join_events = payload.get("explicitJoinEventsInWindow")
  explicit_join_participants = payload.get("explicitJoinParticipantsInWindow")
  llm1_reason_raw = payload.get("llm1Reason")

  def _count_phrase(value, singular: str, plural: str) -> str:
    if value is None:
      return f"unknown {plural}"
    if isinstance(value, int):
      return f"{value} {singular if value == 1 else plural}"
    return f"{value} {plural}"

  def _is_singular_count(value) -> bool:
    return isinstance(value, int) and value == 1

  try:
    mention_count = int(mention_count)
  except (TypeError, ValueError):
    pass

  mention_count_text = _count_phrase(mention_count, "time", "times")
  if isinstance(mention_count, int):
    if mention_count > 0:
      mention_line = f"- Bot is mentioned {mention_count_text} in this current message window."
    elif bot_mentioned:
      mention_line = "- Bot is mentioned in this current message window."
    else:
      mention_line = "- Bot is not mentioned in this current message window."
  elif bot_mentioned:
    mention_line = "- Bot is mentioned in this current message window."
  else:
    mention_line = "- Bot is not mentioned in this current message window."

  if replied_to_bot:
    reply_line = "- A message in this current message window replies to the bot."
  else:
    reply_line = "- No message in this current message window replies to the bot."

  since_assistant_text = _count_phrase(since_assistant, "message", "messages")
  human_window_text = _count_phrase(human_window, "human message", "human messages")

  assistant_reply_lines: list[str] = []
  if isinstance(assistant_replies_by_window, dict):
    assistant_reply_values: list[tuple[int, int | str]] = []
    for raw_window, raw_count in assistant_replies_by_window.items():
      try:
        window = int(raw_window)
      except (TypeError, ValueError):
        continue
      assistant_reply_values.append((window, raw_count))
    assistant_reply_values.sort(key=lambda item: item[0])
    for window, count in assistant_reply_values:
      count_text = _count_phrase(count, "reply", "replies")
      assistant_reply_lines.append(
        f"- Assistant has sent {count_text} in the last {window} messages."
      )

  if not assistant_reply_lines:
    fallback_recent = payload.get("assistantRepliesInLast20")
    fallback_text = _count_phrase(fallback_recent, "reply", "replies")
    assistant_reply_lines.append(
      f"- Assistant has sent {fallback_text} in the last 20 messages."
    )

  if _is_singular_count(human_window):
    human_window_line = f"- There is {human_window_text} in this current message window."
  else:
    human_window_line = f"- There are {human_window_text} in this current message window."

  join_event_text = _count_phrase(explicit_join_events, "event", "events")
  join_participant_text = _count_phrase(explicit_join_participants, "participant", "participants")
  if isinstance(explicit_join_events, int):
    if explicit_join_events > 0:
      join_event_line = (
        "- Explicit system member-join signals in this current message window: "
        f"{join_event_text} ({join_participant_text})."
      )
    else:
      join_event_line = "- No explicit system member-join signal in this current message window."
  else:
    join_event_line = "- Explicit system member-join signal count is unknown for this current message window."

  llm1_reason = ""
  if isinstance(llm1_reason_raw, str):
    llm1_reason = " ".join(llm1_reason_raw.split())
  elif llm1_reason_raw is not None:
    llm1_reason = " ".join(str(llm1_reason_raw).split())

  if llm1_reason:
    llm1_reason_line = f"\nLLM1 routing reason: {llm1_reason}\n\n"
  else:
    llm1_reason_line = "\n"

  assistant_reply_block = "\n".join(assistant_reply_lines)
  chat_state_text = _chat_state_header(chat_type, bot_is_admin, bot_is_super_admin)
  return (
    "Current message metadata:\n"
    "Helper:\n"
    "- `current message window` = only `current messages(burst)` (exclude `older messages`).\n"
    f"{mention_line}\n"
    f"{reply_line}\n"
    f"- The last assistant reply was {since_assistant_text} ago.\n"
    f"{assistant_reply_block}\n"
    f"{human_window_line}\n"
    f"{join_event_line}\n"
    f"{llm1_reason_line}"
    "Chat state:\n"
    f"{chat_state_text}"
  )


def get_llm2(
  *,
  model: str | None = None,
  base_url: str | None = None,
  api_key: str | None = None,
  include_reasoning: bool = True,
) -> ChatOpenAI:
  resolved_model = model or (_clean_env(os.getenv("LLM2_MODEL")) or "gpt-4.1")
  temperature = float(os.getenv("LLM2_TEMPERATURE", "0.5"))
  timeout = _llm2_timeout()
  max_retries = _llm2_sdk_max_retries()
  reasoning_effort = _llm2_reasoning_effort() if include_reasoning else None
  resolved_base_url = base_url if base_url is not None else _clean_env(os.getenv("LLM2_ENDPOINT"))
  resolved_api_key = api_key if api_key is not None else (_clean_env(os.getenv("LLM2_API_KEY")) or "")
  kwargs = {
    "model": resolved_model,
    "temperature": temperature,
    "base_url": resolved_base_url,
    "api_key": resolved_api_key,
    "timeout": timeout,
    "max_retries": max_retries,
  }
  if reasoning_effort:
    kwargs["reasoning_effort"] = reasoning_effort
  return ChatOpenAI(
    **kwargs,
  )


async def generate_reply(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  system: str | None = None,
  tools: Optional[list] = None,
  current_payload: dict | None = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
  chat_type: str | None = None,
  bot_is_admin: bool = False,
  bot_is_super_admin: bool = False,
):
  targets = _llm2_targets()
  payload = current_payload if isinstance(current_payload, dict) else {}
  log_chat_id = payload.get("chatId") or payload.get("chat_id") or current.sender
  payload_chat_type = _normalize_chat_type(
    chat_type
    or payload.get("chatType")
    or payload.get("chat_type")
    or ("group" if bool(payload.get("isGroup")) else "private")
  )
  log_chat_name = (payload.get("chatName") or payload.get("chat_name")) if payload_chat_type == "group" else None
  timeout_s = _llm2_timeout()
  retry_max = _llm2_retry_max()
  retry_backoff_s = _llm2_retry_backoff_seconds()
  sdk_max_retries = _llm2_sdk_max_retries()
  reasoning_effort = _llm2_reasoning_effort()
  base_system = (system or _load_system_prompt()).strip()
  rendered_system = _render_system_prompt(
    base_system,
    prompt_override=prompt_override,
  )
  history_list = list(history)
  hist_text = format_history(history_list) or "(no older messages)"
  current_line = _format_current_window(current) or "(no current messages)"
  group_text = _group_description_block(group_description)
  context_injection = _context_injection_block(
    current_payload,
    chat_type=chat_type or "private",
    bot_is_admin=bot_is_admin,
    bot_is_super_admin=bot_is_super_admin,
  )
  messages_content_text = (
    "older messages:\n"
    f"{hist_text}\n\n"
    "current messages(burst):\n"
    f"{current_line}"
  )
  media_parts: list[dict] = []
  media_notes: list[str] = []
  if llm2_media_enabled():
    media_parts, media_notes = build_visual_parts(current_payload)
  if media_notes:
    messages_content_text += "\n\nVisual attachments:\n" + "\n".join(
      f"- {note}" for note in media_notes
    )

  messages_content: str | list[dict]
  if media_parts:
    messages_content = [{"type": "text", "text": messages_content_text}]
    messages_content.extend(media_parts)
  else:
    messages_content = messages_content_text

  msgs = [SystemMessage(content=rendered_system)]
  msgs.append(HumanMessage(content=f"Group description:\n{group_text}"))
  msgs.append(HumanMessage(content=context_injection))
  msgs.append(HumanMessage(content=messages_content))
  if env_flag("BRIDGE_LOG_PROMPT_FULL"):
    first_target = targets[0]
    logger.info(
      "LLM2 prompt full",
      extra={
        "chat_id": log_chat_id,
        "chat_name": log_chat_name,
        "provider": first_target.name,
        "model": first_target.model,
        "endpoint": first_target.base_url,
        "messages": [
          {"role": "system", "content": rendered_system},
          {"role": "user", "content": f"Group description:\n{group_text}"},
          {"role": "user", "content": context_injection},
          {"role": "user", "content": redact_multimodal_content(messages_content)},
        ],
      },
    )
  prompt_preview = trunc(
    (
      group_text
      + '\n'
      + context_injection
      + '\nolder messages:\n'
      + hist_text
      + '\n\ncurrent messages(burst):\n'
      + current_line
      + f"\n[visual_attachments={len(media_parts)}]"
    ),
    800,
  )

  total_targets = len(targets)
  for idx, target in enumerate(targets):
    has_next_target = idx < (total_targets - 1)
    llm = get_llm2(
      model=target.model,
      base_url=target.base_url,
      api_key=target.api_key,
      include_reasoning=bool(reasoning_effort),
    )
    if tools:
      llm = llm.bind_tools(tools)

    logger.debug(
      "LLM2 invoke",
      extra={
        "chat_id": log_chat_id,
        "chat_name": log_chat_name,
        "provider": target.name,
        "history_len": len(history_list),
        "system_chars": len(rendered_system),
        "prompt_preview": prompt_preview,
        "model": target.model,
        "endpoint": target.base_url,
        "timeout_s": timeout_s,
        "retry_max": retry_max,
        "retry_backoff_s": retry_backoff_s,
        "sdk_max_retries": sdk_max_retries,
        "reasoning_effort": reasoning_effort,
      },
    )

    async def _invoke_with_retry(prompt_msgs, *, mode: str):
      attempts_total = retry_max + 1
      last_failure_kind: str | None = None
      for attempt in range(1, attempts_total + 1):
        started = time.perf_counter()
        logger.info(
          "LLM2 invoke start (provider=%s, mode=%s, attempt=%s/%s, model=%s)",
          target.name,
          mode,
          attempt,
          attempts_total,
          target.model,
          extra={
            "chat_id": log_chat_id,
            "chat_name": log_chat_name,
            "provider": target.name,
            "model": target.model,
            "endpoint": target.base_url,
            "mode": mode,
            "attempt": attempt,
            "attempts_total": attempts_total,
            "timeout_s": timeout_s,
            "sdk_max_retries": sdk_max_retries,
          },
        )
        try:
          response = await llm.ainvoke(prompt_msgs)
          elapsed_ms = int((time.perf_counter() - started) * 1000)
          logger.info(
            "LLM2 invoke success (provider=%s, mode=%s, attempt=%s/%s, elapsed=%sms)",
            target.name,
            mode,
            attempt,
            attempts_total,
            elapsed_ms,
            extra={
              "chat_id": log_chat_id,
              "chat_name": log_chat_name,
              "provider": target.name,
              "model": target.model,
              "endpoint": target.base_url,
              "mode": mode,
              "attempt": attempt,
              "attempts_total": attempts_total,
              "elapsed_ms": elapsed_ms,
              "timeout_s": timeout_s,
              "sdk_max_retries": sdk_max_retries,
            },
          )
          return response, None
        except Exception as err:
          elapsed_ms = int((time.perf_counter() - started) * 1000)
          timeout_error = _is_timeout_error(err)
          last_failure_kind = "timeout" if timeout_error else "error"
          can_retry = timeout_error and attempt < attempts_total
          logger.warning(
            "LLM2 invoke failed",
            exc_info=not can_retry,
            extra={
              "chat_id": log_chat_id,
              "chat_name": log_chat_name,
              "provider": target.name,
              "model": target.model,
              "endpoint": target.base_url,
              "mode": mode,
              "attempt": attempt,
              "attempts_total": attempts_total,
              "elapsed_ms": elapsed_ms,
              "timeout_s": timeout_s,
              "retry_backoff_s": retry_backoff_s,
              "will_retry": can_retry,
              "error_type": type(err).__name__,
              "error_chain": _error_chain(err),
              "sdk_max_retries": sdk_max_retries,
            },
          )
          if not can_retry:
            return None, last_failure_kind
          await asyncio.sleep(retry_backoff_s * attempt)
      return None, last_failure_kind

    result, failure_kind = await _invoke_with_retry(msgs, mode="multimodal" if media_parts else "text")
    if result is None and media_parts:
      if failure_kind == "timeout":
        logger.warning(
          "LLM2 multimodal timeout; skipping text-only fallback on this provider",
          extra={
            "chat_id": log_chat_id,
            "chat_name": log_chat_name,
            "provider": target.name,
            "model": target.model,
            "endpoint": target.base_url,
            "media_parts": len(media_parts),
            "timeout_s": timeout_s,
            "retry_max": retry_max,
            "sdk_max_retries": sdk_max_retries,
            "will_try_fallback_target": has_next_target,
          },
        )
      else:
        logger.warning(
          "LLM2 multimodal failed; retrying text-only prompt",
          extra={
            "chat_id": log_chat_id,
            "chat_name": log_chat_name,
            "provider": target.name,
            "model": target.model,
            "endpoint": target.base_url,
            "media_parts": len(media_parts),
          },
        )
        result, _ = await _invoke_with_retry(
          [
            SystemMessage(content=rendered_system),
            HumanMessage(content=f"Group description:\n{group_text}"),
            HumanMessage(content=context_injection),
            HumanMessage(content=messages_content_text),
          ],
          mode="text_fallback",
        )

    if result is not None:
      logger.debug(
        "LLM2 result",
        extra={
          "chat_id": log_chat_id,
          "chat_name": log_chat_name,
          "provider": target.name,
          "model": target.model,
          "endpoint": target.base_url,
          "reply_preview": trunc(getattr(result, 'content', ''), 800),
          "raw": dump_json(getattr(result, "model_dump", lambda: str(result))()),
        },
      )
      return result

    if has_next_target:
      logger.warning(
        "LLM2 provider failed; trying fallback target",
        extra={
          "chat_id": log_chat_id,
          "chat_name": log_chat_name,
          "provider": target.name,
          "model": target.model,
          "endpoint": target.base_url,
          "next_provider": targets[idx + 1].name,
          "next_model": targets[idx + 1].model,
          "next_endpoint": targets[idx + 1].base_url,
        },
      )

  return None
