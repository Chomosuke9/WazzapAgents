from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict, deque
from typing import Deque, Dict, Set
from urllib.parse import urlsplit

import websockets
from dotenv import load_dotenv

try:
  from .history import WhatsAppMessage
  from .log import setup_logging
  from .llm1 import call_llm1
  from .llm2 import generate_reply
except ImportError:  # allow running as `python python/bridge/main.py`
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage  # type: ignore
  from bridge.log import setup_logging  # type: ignore
  from bridge.llm1 import call_llm1  # type: ignore
  from bridge.llm2 import generate_reply  # type: ignore

load_dotenv()

HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
logger = setup_logging()


def _append_history(buf: Deque[WhatsAppMessage], msg: WhatsAppMessage) -> None:
  buf.append(msg)
  while len(buf) > HISTORY_LIMIT:
    buf.popleft()


async def handle_socket(ws):
  per_chat: Dict[str, Deque[WhatsAppMessage]] = defaultdict(deque)
  per_chat_lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
  tasks: Set[asyncio.Task] = set()
  logger.info("Gateway connected")

  async def process_message(payload: dict):
    chat_id = payload["chatId"]
    history = per_chat[chat_id]
    lock = per_chat_lock[chat_id]
    async with lock:
      try:
        logger.debug(
          "[%s] incoming",
          chat_id,
          extra={
            "message_id": payload.get("messageId"),
            "type": payload.get("messageType"),
            "text": payload.get("text"),
            "attachments": len(payload.get("attachments") or []),
            "quoted": bool(payload.get("quoted")),
            "location": bool(payload.get("location")),
            "history_len": len(history),
            "sender": payload.get("senderName") or payload.get("senderId"),
            "raw_payload": payload,
          },
        )
        current = WhatsAppMessage(
          timestamp_ms=int(payload["timestampMs"]),
          sender=payload.get("senderName") or payload.get("senderId") or payload.get("chatId"),
          text=payload.get("text"),
          media=_infer_media(payload),
          role="user",
        )

        decision = await call_llm1(history, current)
        _append_history(history, current)
        if not decision.should_response:
          logger.info(
            "[%s] skipped (llm1=%s, conf=%s%%)", chat_id, decision.reason, decision.confidence
          )
          return

        reply_msg = await generate_reply(history, current)
        reply_text = _extract_reply_text(reply_msg) if reply_msg else None
        if not reply_text:
          logger.warning("[%s] llm2 returned empty reply", chat_id)
          return

        await send_message(ws, chat_id, reply_text, payload.get("messageId"))
        _append_history(
          history,
          WhatsAppMessage(
            timestamp_ms=int(payload["timestampMs"]),
            sender="LLM",
            text=reply_text,
            role="assistant",
          ),
        )
        logger.info("[%s] replied", chat_id, extra={"reply_preview": reply_text[:120]})
      except Exception as err:
        logger.exception("[%s] handler error: %s", chat_id, err)

  try:
    async for raw in ws:
      try:
        event = json.loads(raw)
      except json.JSONDecodeError:
        logger.warning("Dropping non-JSON payload")
        continue

      if event.get("type") == "hello":
        logger.info("Handshake: %s", event.get("payload"))
        continue

      if event.get("type") != "incoming_message":
        continue

      payload = event["payload"]
      task = asyncio.create_task(process_message(payload))
      tasks.add(task)
      task.add_done_callback(tasks.discard)
  except websockets.ConnectionClosed:
    logger.info("Gateway disconnected")
  finally:
    for task in tasks:
      task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)


async def send_message(ws, chat_id: str, text: str, reply_to: str | None):
  logger.debug(
    "[%s] outbound",
    chat_id,
    extra={"reply_to": reply_to, "text_preview": text[:200], "text_len": len(text or "")},
  )
  await ws.send(
    json.dumps(
      {
        "type": "send_message",
        "payload": {
          "chatId": chat_id,
          "text": text,
          "replyTo": reply_to,
        },
      }
    )
  )


def _infer_media(payload: dict) -> str | None:
  atts = payload.get("attachments") or []
  if not atts:
    if payload.get("messageType") == "stickerMessage":
      return "sticker"
    return None
  return atts[0].get("kind") or atts[0].get("mime") or "media"


def _extract_reply_text(msg) -> str | None:
  if hasattr(msg, "content") and isinstance(msg.content, str):
    return msg.content.strip()
  if hasattr(msg, "content") and isinstance(msg.content, list):
    parts = [part for part in msg.content if isinstance(part, str)]
    return "\n".join(parts).strip() if parts else None
  return None


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
