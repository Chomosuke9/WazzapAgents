from __future__ import annotations

import asyncio
import contextlib
import json
import time

try:
  from ..log import setup_logging
  from .processing import _normalize_context_msg_id
  from .format import sanitize_whatsapp_text
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.log import setup_logging  # type: ignore
  from bridge.messaging.processing import _normalize_context_msg_id  # type: ignore
  from bridge.messaging.format import sanitize_whatsapp_text  # type: ignore

logger = setup_logging()


async def send_message(
  ws,
  chat_id: str,
  text: str,
  reply_to: str | None,
  *,
  request_id: str,
):
  # Sanitize WhatsApp formatting before sending: LLMs sometimes produce
  # **bold** (Markdown) which renders as literal asterisks in WhatsApp
  # instead of bold text. Convert to *bold* (WhatsApp-compatible).
  text = sanitize_whatsapp_text(text) if text else text
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


async def send_attachment(
  ws,
  chat_id: str,
  attachment_path: str,
  kind: str,
  *,
  request_id: str,
  file_name: str | None = None,
  reply_to: str | None = None,
  caption: str | None = None,
  mime: str | None = None,
):
  """Send a single attachment to a chat as its own WhatsApp message.

  ``kind`` must be one of: ``image``, ``video``, ``audio``, ``sticker``,
  ``document``. The Node gateway (``src/wa/outbound.js::sendOutgoing``) already
  accepts an ``attachments`` array on the ``send_message`` payload — this
  helper just builds a payload with exactly one attachment so each file lands
  in its own bubble.

  ``mime`` is forwarded to Node so the gateway can set Baileys'
  ``content.mimetype`` explicitly. Without it Baileys falls back to its own
  guess, which for unfamiliar files is ``application/pdf`` — that produces
  WhatsApp messages that can't be opened. Pass the value returned by
  :func:`bridge.subagent.output.detect_kind` whenever possible.

  The Node side re-validates ``attachment_path`` against ``MEDIA_DIR`` /
  ``STICKERS_DIR`` via ``resolveAllowedAttachmentPath``, so a path outside the
  sandbox will be rejected by the action even though we don't check here.
  """
  if not attachment_path or not kind:
    return
  # Sanitize WhatsApp formatting in caption before sending, same as
  # send_message() does for plain text.
  if caption:
    caption = sanitize_whatsapp_text(caption)
  normalized_reply_to = _normalize_context_msg_id(reply_to) if reply_to else None
  attachment: dict = {"kind": kind, "path": attachment_path}
  if file_name:
    attachment["fileName"] = file_name
  if caption:
    attachment["caption"] = caption
  if mime:
    attachment["mime"] = mime
  payload: dict = {
    "requestId": request_id,
    "chatId": chat_id,
    "attachments": [attachment],
  }
  if normalized_reply_to:
    payload["replyTo"] = normalized_reply_to
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "send_attachment",
      "request_id": request_id,
      "kind": kind,
      "attachment_path": attachment_path,
      "file_name": file_name,
      "mime": mime,
      "reply_to": normalized_reply_to,
    },
  )
  await ws.send(json.dumps({"type": "send_message", "payload": payload}))


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
