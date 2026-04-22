# WazzapAgents

[![Node 18+](https://img.shields.io/badge/node-%3E%3D18-brightgreen)](https://nodejs.org/)
[![Python 3.10+](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://python.org/)
[![License](https://img.shields.io/badge/license-ISC-lightgrey)](./package.json)

WhatsApp AI agent system: a Node.js gateway (Baileys v7) connects a WhatsApp account and forwards messages to a Python LLM bridge over WebSocket. The bridge runs a two-stage LLM pipeline (routing + response generation) and sends moderation/action commands back to the gateway.

> **For full architecture, concepts, and developer context**, see [AGENTS.md](./AGENTS.md).

## Architecture

```
WhatsApp (phone)
      ↕  Baileys v7 socket
┌─────────────────────────────┐
│  Node.js Gateway  (src/)    │   Python Bridge  (python/bridge/)
│  WhatsApp ↔ WS bridge       │   LLM1 routing → LLM2 response
│  Slash commands, actions     │   Debounce, tool calls, dispatch
└──────────┬──────────────────┘
           │ WebSocket
┌──────────▼──────────────────┐
│  Python LLM Bridge          │
│  Message batching, LLM1/2, │
│  tool extraction, actions   │
└─────────────────────────────┘
```

See [AGENTS.md](./AGENTS.md) for ADRs, terminology, and detailed module descriptions.

## Prerequisites
- Node.js 18+ (tested with Node 25).
- Python 3.10+.
- pnpm 9+ (`npm i -g pnpm` or `corepack enable pnpm`).
- Internet access to install dependencies.

## Quick Start
1. Copy `.env.example` to `.env` and set **required** `LLM_WS_ENDPOINT` (e.g., `ws://localhost:8080/ws`). Adjust other values as needed.
2. Install Node deps: `pnpm install`.
3. Install Python deps: `pip install -r requirements.txt`.
4. Start the Python bridge: `python -m python.bridge.main`.
5. Start the Node gateway: `pnpm dev`.
6. Scan the QR code in the terminal to pair your WhatsApp account (auth stored in `data/auth`).

## Detailed Setup
1. Copy `.env.example` to `.env`, fill required `LLM_WS_ENDPOINT` first, then adjust optional values as needed.
2. Install Node deps: `pnpm install` (Baileys v7 is ESM-only; this project is `type: module`).
3. Install Python deps: `pip install -r requirements.txt` (Python 3.10+).
4. Run the gateway: `pnpm dev`.
5. Run the Python bridge: `python -m python.bridge.main`.
6. Scan the QR in the terminal to pair the WhatsApp account (auth is stored in `data/auth`).

## Runtime folders
- `data/auth`: WhatsApp session files.
- `data/media`: Media downloaded from incoming messages; paths are sent to the LLM.

## WebSocket protocol (gateway ↔ LLM)

### Gateway -> LLM: `incoming_message`
```json
{
  "type": "incoming_message",
  "payload": {
    "contextMsgId": "000125",
    "messageId": "wamid-abc",
    "instanceId": "dev-gateway-1",
    "chatId": "12345@g.us",
    "chatName": "Group Name",
    "chatType": "group",
    "senderId": "98765@s.whatsapp.net",
    "senderRef": "u8k2d1",
    "senderName": "Alice",
    "senderIsAdmin": false,
    "senderIsOwner": false,
    "isGroup": true,
    "botIsAdmin": true,
    "botIsSuperAdmin": false,
    "fromMe": false,
    "contextOnly": false,
    "triggerLlm1": false,
    "timestampMs": 1738560000000,
    "messageType": "extendedTextMessage",
    "text": "Hello world",
    "quoted": {
      "messageId": "wamid-quoted",
      "contextMsgId": "000124",
      "senderId": "555@s.whatsapp.net",
      "text": "Previous message",
      "type": "conversation"
    },
    "attachments": [
      {
        "kind": "image",
        "mime": "image/jpeg",
        "fileName": "wamid_image.jpg",
        "size": 12345,
        "path": "data/media/wamid_image.jpg",
        "isAnimated": false
      }
    ],
    "mentionedJids": ["123@s.whatsapp.net"],
    "mentionedParticipants": [
      {
        "jid": "123@s.whatsapp.net",
        "senderRef": "u1m9qa",
        "name": "Bob"
      }
    ],
    "botMentioned": false,
    "repliedToBot": false,
    "location": null,
    "groupDescription": "Rules and context for this group",
    "slashCommand": null
  }
}
```

Notes:
- `contextMsgId` is a 6-digit per-chat sequence (`000000..999999`, wraps after `999999`).
- `senderRef` is a short deterministic reference per sender in each chat; LLM moderation must use this, not JIDs.
- `senderIsOwner` indicates whether the sender is a bot owner (configured via `BOT_OWNER_JIDS`).
- `mentionedParticipants` resolves mentions into `{ jid, senderRef, name }` when available; keep using `mentionedJids` for backwards compatibility if needed.
- `botMentioned` / `repliedToBot` signal whether the bot was explicitly mentioned or replied to.
- `location` contains location data when a location message is shared, otherwise `null`.
- `slashCommand` contains `{ command, args }` when the message is a recognized slash command, otherwise `null`.
- Bot messages are forwarded as `contextOnly: true` and `triggerLlm1: false` so they enrich context without causing loops.
- Gateway may emit synthetic bot context events with `messageType: "actionLog"` and `actionLog` details after successful moderation actions (`delete_message`, `kick_member`).
- Backend bridge enforces moderation permissions via `/permission` command: `DELETE`/`KICK` actions are dropped unless the chat's permission level allows them and bot role is admin/superadmin.

### LLM -> Gateway: `send_message`
```json
{
  "type": "send_message",
  "payload": {
    "requestId": "req-send-001",
    "chatId": "12345@g.us",
    "text": "Reply text @whoami (u8k2d1) @everyone (everyone)",
    "replyTo": "000124",
    "attachments": [
      {
        "kind": "image",
        "path": "data/media/to-send.jpg",
        "caption": "optional"
      }
    ]
  }
}
```

Notes:
- Mention one user inside outgoing text/caption with `@Name (senderRef)`.
- Mention all group members with `@everyone (everyone)`.
- Invalid `senderRef` mention tokens are silently skipped (message still sent).

### LLM -> Gateway: `react_message`
```json
{
  "type": "react_message",
  "payload": {
    "requestId": "req-react-001",
    "chatId": "12345@g.us",
    "contextMsgId": "000125",
    "emoji": "👍"
  }
}
```

### LLM -> Gateway: `delete_message`
```json
{
  "type": "delete_message",
  "payload": {
    "requestId": "req-del-001",
    "chatId": "12345@g.us",
    "contextMsgId": "000125"
  }
}
```

### LLM -> Gateway: `kick_member`
```json
{
  "type": "kick_member",
  "payload": {
    "requestId": "req-kick-001",
    "chatId": "12345@g.us",
    "targets": [
      { "senderRef": "u8k2d1", "anchorContextMsgId": "000125" },
      { "senderRef": "u1m9qa", "anchorContextMsgId": "000124" }
    ],
    "mode": "partial_success",
    "autoReplyAnchor": true
  }
}
```

### LLM -> Gateway: `mark_read`
```json
{
  "type": "mark_read",
  "payload": {
    "chatId": "12345@g.us",
    "messageId": "wamid-abc",
    "participant": "98765@s.whatsapp.net"
  }
}
```

Notes:
- `participant` is optional; include it for group messages.

### LLM -> Gateway: `send_presence`
```json
{
  "type": "send_presence",
  "payload": {
    "chatId": "12345@g.us",
    "type": "composing"
  }
}
```

Notes:
- `type` can be `"composing"` (typing indicator) or `"paused"` (stop typing). Defaults to `"composing"`.

## Acknowledgements and errors

### Gateway -> LLM: `action_ack`
```json
{
  "type": "action_ack",
  "payload": {
    "requestId": "req-del-001",
    "action": "delete_message",
    "ok": true,
    "detail": "deleted",
    "result": {
      "contextMsgId": "000125",
      "messageId": "wamid-abc"
    }
  }
}
```

### Legacy compatibility
- `send_ack` is still emitted for successful `send_message`.
- `error` is emitted for command failures with stable `code` values where possible (`not_found`, `not_group`, `permission_denied`, `invalid_target`, `send_failed`).

## Notes
- Attachment paths are local; if your LLM service runs elsewhere, you need a file-serving layer or shared volume.
- `delete_message` runs in strict mode: unresolved `contextMsgId` fails without speculative fallback.
- `kick_member` resolves targets via backend senderRef registry and validates each `senderRef@anchorContextMsgId` pair before removal.
- If the WhatsApp session logs out, delete `data/auth` and re-run to re-pair.
- Multi-account: run multiple gateway instances with different `INSTANCE_ID` and separate `DATA_DIR`/`MEDIA_DIR`.
- Baileys version pinned to `7.0.0-rc.9` (package name `baileys`); ensure Node 18+ with ESM support.
- **Interactive messages** (`viewOnceMessage` + `additionalNodes`) only render on mobile clients, not WhatsApp Web.
- **LLM1 is skipped in private chats** — all DMs get a full LLM2 response (confidence 100).

## Example LLM WebSocket (Python)
See `examples/llm_ws_echo.py` for a minimal server that:
- Listens on `ws://0.0.0.0:8080/ws`.
- Logs `incoming_message` payload including `contextMsgId` and `senderRef`.
- Sends `send_message`, `delete_message`, `react_message`, and `kick_member` examples.

Run:
```bash
pip install websockets==12.* pydantic
python examples/llm_ws_echo.py
```
