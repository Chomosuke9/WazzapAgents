from __future__ import annotations

import asyncio
import contextlib
import json
import time

try:
  from .log import setup_logging
  from .message_processing import _normalize_context_msg_id
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.log import setup_logging  # type: ignore
  from bridge.message_processing import _normalize_context_msg_id  # type: ignore

logger = setup_logging()


async def send_message(
  ws,
  chat_id: str,
  text: str,
  reply_to: str | None,
  *,
  request_id: str,
):
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "send_message",
      "request_id": request_id,
      "reply_to": reply_to,
      "text_preview": text[:200],
      "text_len": len(text or ""),
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "send_message",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "text": text,
          "replyTo": reply_to,
        },
      }
    )
  )


async def send_delete_message(
  ws,
  chat_id: str,
  context_msg_id: str | None,
  *,
  request_id: str,
):
  normalized_context_msg_id = _normalize_context_msg_id(context_msg_id)
  if not normalized_context_msg_id:
    return
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "delete_message",
      "request_id": request_id,
      "context_msg_id": normalized_context_msg_id,
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "delete_message",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "contextMsgId": normalized_context_msg_id,
        },
      }
    )
  )


async def send_kick_member(
  ws,
  chat_id: str,
  targets: list[dict[str, str]],
  *,
  request_id: str,
  mode: str = "partial_success",
  auto_reply_anchor: bool = False,
):
  if not targets:
    return
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "kick_member",
      "request_id": request_id,
      "targets": targets,
      "mode": mode,
      "auto_reply_anchor": auto_reply_anchor,
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "kick_member",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "targets": targets,
          "mode": mode,
          "autoReplyAnchor": auto_reply_anchor,
        },
      }
    )
  )


async def send_react_message(
  ws,
  chat_id: str,
  context_msg_id: str | None,
  emoji: str | None,
  *,
  request_id: str,
):
  normalized_context_msg_id = _normalize_context_msg_id(context_msg_id)
  if not normalized_context_msg_id or not emoji:
    return
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "react_message",
      "request_id": request_id,
      "context_msg_id": normalized_context_msg_id,
      "emoji": emoji,
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "react_message",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "contextMsgId": normalized_context_msg_id,
          "emoji": emoji,
        },
      }
    )
  )


async def send_sticker(
  ws,
  chat_id: str,
  sticker_path: str,
  reply_to: str | None,
  *,
  request_id: str,
):
  normalized_reply_to = _normalize_context_msg_id(reply_to) if reply_to else None
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "send_sticker",
      "request_id": request_id,
      "sticker_path": sticker_path,
      "reply_to": normalized_reply_to,
    },
  )
  payload: dict = {
    "requestId": request_id,
    "chatId": chat_id,
    "attachments": [{"kind": "sticker", "path": sticker_path}],
  }
  if normalized_reply_to:
    payload["replyTo"] = normalized_reply_to
  await ws.send(
    json.dumps({"type": "send_message", "payload": payload})
  )


async def send_mark_read(
  ws,
  chat_id: str,
  message_id: str | None,
  participant: str | None = None,
):
  """Send a read receipt signal to the gateway."""
  if not message_id:
    return
  payload: dict = {"chatId": chat_id, "messageId": message_id}
  if participant:
    payload["participant"] = participant
  try:
    await ws.send(json.dumps({"type": "mark_read", "payload": payload}))
  except Exception as err:
    logger.debug("send_mark_read failed: %s", err)


async def send_typing(ws, chat_id: str, composing: bool = True):
  """Send typing presence to the gateway."""
  presence_type = "composing" if composing else "paused"
  try:
    await ws.send(
      json.dumps({"type": "send_presence", "payload": {"chatId": chat_id, "type": presence_type}})
    )
  except Exception as err:
    logger.debug("send_typing failed: %s", err)


@contextlib.asynccontextmanager
async def typing_indicator(ws, chat_id: str, interval: float = 8.0):
  """Context manager that keeps the typing indicator alive by refreshing it periodically.

  WhatsApp's typing indicator expires after ~10-15 seconds on the client side.
  This sends a fresh 'composing' presence every *interval* seconds so the
  indicator stays visible even when LLM2 takes a long time to respond.
  """

  async def _keep_alive():
    try:
      while True:
        await asyncio.sleep(interval)
        await send_typing(ws, chat_id, composing=True)
    except asyncio.CancelledError:
      pass

  await send_typing(ws, chat_id, composing=True)
  task = asyncio.create_task(_keep_alive())
  try:
    yield
  finally:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await task
    await send_typing(ws, chat_id, composing=False)
