from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
  from .history import WhatsAppMessage, format_history
  from .log import setup_logging, trunc, dump_json
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage, format_history  # type: ignore
  from bridge.log import setup_logging, trunc, dump_json  # type: ignore

logger = setup_logging()
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "systemprompt.txt"
SYSTEM_PROMPT_FALLBACK = (
  "You are Vivy, a concise WhatsApp assistant.\n"
  "Reply in the user's language.\n"
  "You may output one or more reply blocks.\n"
  "Each block must start with: REPLY_TO:<messageId|none>\n"
  "Then write user-facing text.\n"
  "Allowed messageId values: {{ allowed_message_ids }}\n"
  "MessageId context:\n"
  "{{ message_id_context }}\n"
  "Use REPLY_TO:none to send without quoting.\n"
  "Do not mention messageId in user-facing text."
)
_SYSTEM_PROMPT_CACHE: str | None = None


def _load_system_prompt() -> str:
  global _SYSTEM_PROMPT_CACHE
  if _SYSTEM_PROMPT_CACHE is not None:
    return _SYSTEM_PROMPT_CACHE

  try:
    text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
  except Exception as err:
    logger.warning("Failed reading system prompt file; using fallback", exc_info=err)
    text = ""

  if not text:
    text = SYSTEM_PROMPT_FALLBACK
  _SYSTEM_PROMPT_CACHE = text
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
    reasoning_effort="low",
  )


async def generate_reply(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  system: str | None = None,
  tools: Optional[list] = None,
  reply_candidates: Optional[list[dict[str, str]]] = None,
):
  llm = get_llm2()
  base_system = (system or _load_system_prompt()).strip()
  rendered_system = _render_system_prompt(base_system, reply_candidates)
  msgs = [SystemMessage(content=rendered_system)]
  hist_text = format_history(history)
  current_line = format_history([current])
  msgs.append(
    HumanMessage(
      content=(
        "WhatsApp context (latest last):\n"
        f"{hist_text}\n"
        "Current WhatsApp message:\n"
        f"{current_line}"
      )
    )
  )
  if tools:
    llm = llm.bind_tools(tools)
  logger.debug(
    "LLM2 invoke",
    extra={
      "chat_id": current.sender,
      "history_len": len(list(history)),
      "reply_candidates_count": len(reply_candidates or []),
      "system_chars": len(rendered_system),
      "prompt_preview": trunc(hist_text + '\n' + current_line, 800),
      "model": os.getenv("LLM2_MODEL", "gpt-4.1"),
    },
  )
  try:
    result = await llm.ainvoke(msgs)
  except Exception as err:
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
