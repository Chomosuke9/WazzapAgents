from __future__ import annotations

import os
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


def _load_system_prompt() -> str:
  global _SYSTEM_PROMPT_CACHE
  if _SYSTEM_PROMPT_CACHE is not None:
    return _SYSTEM_PROMPT_CACHE
  text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
  return text


def _render_system_prompt(
  base_system: str,
  reply_candidates: Optional[list[dict[str, str]]],
) -> str:
  if reply_candidates:
    candidate_ids = [x.get("message_id", "") for x in reply_candidates if x.get("message_id")]
    allowed_ids = ", ".join(candidate_ids) if candidate_ids else "(none)"
    lines: list[str] = []
    for item in reply_candidates:
      message_id = item.get("message_id")
      if not message_id:
        continue
      sender = item.get("sender", "unknown")
      preview = item.get("preview", "(no text)")
      lines.append(f"- {message_id} | {sender} | {preview}")
    context = "\n".join(lines) if lines else "(no candidate context)"
  else:
    allowed_ids = "(none)"
    context = "(no candidate context)"

  return (
    base_system
    .replace("{{ allowed_message_ids }}", allowed_ids)
    .replace("{{ message_id_context }}", context)
  )


def get_llm2() -> ChatOpenAI:
  model = os.getenv("LLM2_MODEL", "gpt-4.1")
  temperature = float(os.getenv("LLM2_TEMPERATURE", "0.5"))
  timeout = float(os.getenv("LLM2_TIMEOUT", "20"))
  base_url = os.getenv("LLM2_ENDPOINT")  # optional: custom base URL / proxy
  api_key = os.getenv("LLM2_API_KEY", "")  # optional: override OPENAI_API_KEY
  return ChatOpenAI(
    model=model,
    temperature=temperature,
    base_url=base_url,
    api_key=api_key,
    timeout=timeout,
    reasoning_effort="high",
  )


async def generate_reply(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  system: str | None = None,
  tools: Optional[list] = None,
  reply_candidates: Optional[list[dict[str, str]]] = None,
  current_payload: dict | None = None,
):
  llm = get_llm2()
  base_system = (system or _load_system_prompt()).strip()
  rendered_system = _render_system_prompt(base_system, reply_candidates)
  history_list = list(history)
  hist_text = format_history(history_list)
  current_line = format_history([current])
  user_content_text = (
    "WhatsApp context (latest last):\n"
    f"{hist_text}\n"
    "Current WhatsApp message:\n"
    f"{current_line}"
  )
  media_parts: list[dict] = []
  media_notes: list[str] = []
  if llm2_media_enabled():
    media_parts, media_notes = build_visual_parts(current_payload)
  if media_notes:
    user_content_text += "\n\nVisual attachments:\n" + "\n".join(
      f"- {note}" for note in media_notes
    )

  user_content: str | list[dict]
  if media_parts:
    user_content = [{"type": "text", "text": user_content_text}]
    user_content.extend(media_parts)
  else:
    user_content = user_content_text

  msgs = [SystemMessage(content=rendered_system)]
  msgs.append(HumanMessage(content=user_content))
  if tools:
    llm = llm.bind_tools(tools)
  if env_flag("BRIDGE_LOG_PROMPT_FULL"):
    logger.info(
      "LLM2 prompt full",
      extra={
        "chat_id": current.sender,
        "model": os.getenv("LLM2_MODEL", "gpt-4.1"),
        "messages": [
          {"role": "system", "content": rendered_system},
          {"role": "user", "content": redact_multimodal_content(user_content)},
        ],
        "reply_candidates": reply_candidates or [],
      },
    )
  logger.debug(
    "LLM2 invoke",
    extra={
      "chat_id": current.sender,
      "history_len": len(history_list),
      "reply_candidates_count": len(reply_candidates or []),
      "system_chars": len(rendered_system),
      "prompt_preview": trunc(
        hist_text + '\n' + current_line + f"\n[visual_attachments={len(media_parts)}]",
        800,
      ),
      "model": os.getenv("LLM2_MODEL", "gpt-4.1"),
    },
  )
  try:
    result = await llm.ainvoke(msgs)
  except Exception as err:
    if media_parts:
      logger.warning(
        "LLM2 multimodal invoke failed; retrying text-only",
        exc_info=err,
        extra={
          "chat_id": current.sender,
          "model": os.getenv("LLM2_MODEL", "gpt-4.1"),
          "media_parts": len(media_parts),
        },
      )
      try:
        result = await llm.ainvoke(
          [SystemMessage(content=rendered_system), HumanMessage(content=user_content_text)]
        )
      except Exception as retry_err:
        logger.warning(
          "LLM2 invoke failed",
          exc_info=retry_err,
          extra={
            "chat_id": current.sender,
            "model": os.getenv("LLM2_MODEL", "gpt-4.1"),
            "timeout": os.getenv("LLM2_TIMEOUT", "20"),
          },
        )
        return None
    else:
      logger.warning(
        "LLM2 invoke failed",
        exc_info=err,
        extra={
          "chat_id": current.sender,
          "model": os.getenv("LLM2_MODEL", "gpt-4.1"),
          "timeout": os.getenv("LLM2_TIMEOUT", "20"),
        },
      )
      return None

  logger.debug(
    "LLM2 result",
    extra={
      "chat_id": current.sender,
      "reply_preview": trunc(getattr(result, 'content', ''), 800),
      "raw": dump_json(getattr(result, "model_dump", lambda: str(result))()),
    },
  )
  return result
