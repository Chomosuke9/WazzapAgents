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


def _normalize_chat_type(chat_type: str | None) -> str:
  lowered = (chat_type or "").strip().lower()
  if lowered in {"private", "group"}:
    return lowered
  return "private"


def _chat_state_header(chat_type: str, bot_is_admin: bool, bot_is_super_admin: bool) -> str:
  normalized_type = _normalize_chat_type(chat_type)
  return (
    f"CHAT_TYPE: {normalized_type}\n"
    f"BOT_ROLE: botIsAdmin={'true' if bot_is_admin else 'false'} "
    f"botIsSuperAdmin={'true' if bot_is_super_admin else 'false'}"
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
  current_payload: dict | None = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
  chat_type: str | None = None,
  bot_is_admin: bool = False,
  bot_is_super_admin: bool = False,
):
  llm = get_llm2()
  base_system = (system or _load_system_prompt()).strip()
  rendered_system = _render_system_prompt(
    base_system,
    prompt_override=prompt_override,
  )
  history_list = list(history)
  hist_text = format_history(history_list) or "(no history)"
  current_line = format_history([current])
  group_text = _group_description_block(group_description)
  chat_state_text = _chat_state_header(chat_type or "private", bot_is_admin, bot_is_super_admin)
  older_messages_content = f"Chat state:\n{chat_state_text}\n\nOlder messages:\n{hist_text}"
  current_content_text = f"Chat state:\n{chat_state_text}\n\nCurrent messages:\n{current_line}"
  media_parts: list[dict] = []
  media_notes: list[str] = []
  if llm2_media_enabled():
    media_parts, media_notes = build_visual_parts(current_payload)
  if media_notes:
    current_content_text += "\n\nVisual attachments:\n" + "\n".join(
      f"- {note}" for note in media_notes
    )

  current_content: str | list[dict]
  if media_parts:
    current_content = [{"type": "text", "text": current_content_text}]
    current_content.extend(media_parts)
  else:
    current_content = current_content_text

  msgs = [SystemMessage(content=rendered_system)]
  msgs.append(HumanMessage(content=f"Group description:\n{group_text}"))
  msgs.append(HumanMessage(content=older_messages_content))
  msgs.append(HumanMessage(content=current_content))
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
          {"role": "user", "content": f"Group description:\n{group_text}"},
          {"role": "user", "content": older_messages_content},
          {"role": "user", "content": redact_multimodal_content(current_content)},
        ],
      },
    )
  logger.debug(
    "LLM2 invoke",
    extra={
      "chat_id": current.sender,
      "history_len": len(history_list),
      "system_chars": len(rendered_system),
      "prompt_preview": trunc(
        group_text + '\n' + chat_state_text + '\n' + hist_text + '\n' + current_line + f"\n[visual_attachments={len(media_parts)}]",
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
          [
            SystemMessage(content=rendered_system),
            HumanMessage(content=f"Group description:\n{group_text}"),
            HumanMessage(content=older_messages_content),
            HumanMessage(content=current_content_text),
          ]
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
