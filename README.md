# WazzapAgents Gateway (Baileys -> LLM)

Node.js (ESM) gateway that connects a WhatsApp account via Baileys v7 and forwards messages to an external LLM service over an outbound WebSocket. It accepts structured moderation commands from the LLM bridge.

## Prerequisites
- Node.js 18+ (tested with Node 25).
- pnpm 9+ (`npm i -g pnpm` or `corepack enable pnpm`).
- Internet access to install dependencies (`pnpm install`).

## Setup
1. Copy `.env.example` to `.env`, fill required `LLM_WS_ENDPOINT` first, then adjust optional values as needed.
2. Install deps with `pnpm install` (Baileys v7 is ESM-only; this project is `type: module`).
3. Run the gateway: `pnpm dev`.
4. Scan the QR in the terminal to pair the WhatsApp account (auth is stored in `data/auth`).

## Runtime folders
- `data/auth`: WhatsApp session files.
- `data/media`: Media downloaded from incoming messages; paths are sent to the LLM.

## WebSocket protocol (gateway <-> LLM)

### Gateway -> LLM: `incoming_message`
```json
{
  "type": "incoming_message",
  "payload": {
    "instanceId": "dev-gateway-1",
    "contextMsgId": "000125",
    "messageId": "wamid-abc",
    "chatId": "12345@g.us",
    "chatName": "Group Name",
    "chatType": "group",
    "isGroup": true,
    "botIsAdmin": true,
    "botIsSuperAdmin": false,
    "senderId": "98765@s.whatsapp.net",
    "senderRef": "u8k2d1",
    "senderName": "Alice",
    "senderIsAdmin": false,
    "fromMe": false,
    "contextOnly": false,
    "triggerLlm1": false,
    "timestampMs": 1738560000000,
    "messageType": "extendedTextMessage",
    "text": "Hello world",
    "mentionedJids": ["123@s.whatsapp.net"],
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
    "groupDescription": "Rules and context for this group (without prompt_override block)",
    "groupPromptOveride": "Extra instructions extracted from <prompt_override>...</prompt_override>"
  }
}
```

Notes:
- `contextMsgId` is a 6-digit per-chat sequence (`000000..999999`, wraps after `999999`).
- `senderRef` is a short deterministic reference per sender in each chat; LLM moderation must use this, not JIDs.
- Bot messages are forwarded as `contextOnly: true` and `triggerLlm1: false` so they enrich context without causing loops.
- Gateway may emit synthetic bot context events with `messageType: "actionLog"` and `actionLog` details after successful moderation actions (`delete_message`, `kick_member`).

### LLM -> Gateway: `send_message`
```json
{
  "type": "send_message",
  "payload": {
    "requestId": "req-send-001",
    "chatId": "12345@g.us",
    "text": "Reply text",
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

## Example LLM WebSocket (Python)
See `examples/llm_ws_echo.py` for a minimal server that:
- Listens on `ws://0.0.0.0:8080/ws`.
- Logs `incoming_message` payload including `contextMsgId` and `senderRef`.
- Sends `send_message`, `delete_message`, and `kick_member` examples.

Run:
```bash
pip install websockets==12.* pydantic
python examples/llm_ws_echo.py
```
