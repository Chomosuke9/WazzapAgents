from __future__ import annotations

import os
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
):
  llm = get_llm2()
  msgs = []
  if system:
    msgs.append(SystemMessage(content=system))
  hist_text = format_history(history)
  current_line = format_history([current])
  msgs.append(
    HumanMessage(
      content=(
        "Context from WhatsApp (latest last):\n"
        f"{hist_text}\n"
        "New message:\n"
        f"{current_line}\n"
        "Reply concisely in the user's language."
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
