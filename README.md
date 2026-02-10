# WazzapAgents Gateway (Baileys → LLM)

Node.js (ESM) gateway that connects a WhatsApp account via Baileys v7 and forwards messages to an external LLM service over an outbound WebSocket. It also accepts outbound instructions from the LLM to send messages or media back to WhatsApp.

## Prerequisites
- Node.js 18+ (tested with Node 25).
- pnpm 9+ (`npm i -g pnpm` or `corepack enable pnpm`).
- Internet access to install dependencies (`pnpm install`).

## Setup
1. Copy `.env.example` to `.env` and fill `LLM_WS_ENDPOINT` (your WebSocket server URL) and optional `LLM_WS_TOKEN`.
2. Install deps with `pnpm install` (Baileys v7 is ESM-only; this project is `type: module`).
3. Run the gateway: `pnpm dev`.
4. Scan the QR in the terminal to pair the WhatsApp account (auth is stored in `data/auth`).

## Runtime folders
- `data/auth`: WhatsApp session files.
- `data/media`: Media downloaded from incoming messages; paths are sent to the LLM.

## WebSocket protocol (gateway ↔ LLM)
- Gateway → LLM: `incoming_message`
```json
{
  "type": "incoming_message",
  "payload": {
    "instanceId": "dev-gateway-1",
    "messageId": "wamid",
    "chatId": "12345@g.us",
    "chatName": "Group Name",
    "senderId": "98765@s.whatsapp.net",
    "senderName": "Alice",
    "isGroup": true,
    "groupDescription": "Rules and context for this group (without prompt_overide block)",
    "groupPromptOveride": "Extra instructions extracted from <prompt_overide>...</prompt_overide>",
    "contextOnly": false,
    "triggerLlm1": false,
    "timestampMs": 1738560000000,
    "messageType": "extendedTextMessage",
    "text": "Hello world",
    "mentionedJids": ["123@s.whatsapp.net"],
    "quoted": {
      "messageId": "wamid-quoted",
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
    ]
  }
}
```

- LLM → Gateway: `send_message`
```json
{
  "type": "send_message",
  "payload": {
    "requestId": "optional-correlation-id",
    "chatId": "12345@g.us",
    "text": "Reply text",
    "replyTo": "optional-wamid-to-quote",
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

- Gateway → LLM acknowledgements:
  - `send_ack` with `requestId` when an outbound send succeeds.
  - `error` with `requestId` when a send fails.
  - `whatsapp_status` when the WA connection opens or closes.

## Notes
- Attachment paths are local; if your LLM service runs elsewhere, you’ll need a file-serving layer or shared volume.
- Quoting outbound works only for messages still cached (in-memory ~200 messages).
- Group participant join notifications are sent as `incoming_message` with `contextOnly: true` and `triggerLlm1: true`, so they enrich context and still let LLM1 decide whether to respond.
- If the WhatsApp session logs out, delete `data/auth` and re-run to re-pair.
- Multi-account: run multiple gateway instances with different `INSTANCE_ID` and separate `DATA_DIR`/`MEDIA_DIR` to keep sessions isolated.
- Baileys version pinned to `7.0.0-rc.9` (package name `baileys`); ensure Node 18+ with ESM support.

## Example LLM WebSocket (Python)
See `examples/llm_ws_echo.py` for a minimal echo server that:
- Listens on `ws://0.0.0.0:8080/ws`.
- Logs `incoming_message`.
- Replies with `send_message` quoting the original message.

Run:
```bash
pip install websockets==12.* pydantic
python examples/llm_ws_echo.py
```
