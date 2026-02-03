"""
Minimal LLM-side WebSocket server example for the WhatsApp gateway.

What it does:
- Accepts a single gateway connection on ws://0.0.0.0:8080/ws
- Logs incoming messages.
- Echoes back text replies to the same chat, quoting the original message when present.

Requirements:
- Python 3.10+
- websockets==12.*, pydantic>=2 (install with `pip install websockets==12.* pydantic`)

Run:
    python examples/llm_ws_echo.py
"""

import asyncio
import json
from typing import Any, Dict

import websockets
from pydantic import BaseModel


class IncomingMessage(BaseModel):
    instanceId: str
    messageId: str
    chatId: str
    chatName: str
    senderId: str
    senderName: str
    isGroup: bool
    timestampMs: int
    messageType: str
    text: str | None = None
    quoted: Dict[str, Any] | None = None
    attachments: list[Dict[str, Any]] = []


async def handler(websocket: websockets.WebSocketServerProtocol):
    async for raw in websocket:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            print("Received non-JSON payload, ignoring")
            continue

        if event.get("type") == "hello":
            print(f"Gateway connected: {event.get('payload')}")
            continue

        if event.get("type") != "incoming_message":
            print(f"Other event: {event.get('type')}")
            continue

        try:
            msg = IncomingMessage.model_validate(event["payload"])
        except Exception as exc:
            print(f"Failed to parse incoming_message: {exc}")
            continue

        print(f"[{msg.chatName}] {msg.senderName}: {msg.text!r} attachments={len(msg.attachments)}")

        # Build reply
        reply_payload = {
            "type": "send_message",
            "payload": {
                "requestId": msg.messageId,
                "chatId": msg.chatId,
                "text": f"Echo: {msg.text or '(no text)'}",
                "replyTo": msg.messageId,  # quote the incoming message
                "attachments": [],
            },
        }

        await websocket.send(json.dumps(reply_payload))


async def main():
    server = await websockets.serve(
        handler,
        host="0.0.0.0",
        port=8080,
        path="/ws",
        max_size=20 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    )
    print("LLM WS server listening on ws://0.0.0.0:8080/ws")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
