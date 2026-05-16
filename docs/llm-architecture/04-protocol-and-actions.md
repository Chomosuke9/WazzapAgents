# 04 - Protocol and Actions

## Node → Python events

### `incoming_message`
The primary event containing a normalized chat message payload.

Key fields:
- `chatId`, `chatType`, `chatName` — Chat identification
- `senderId`, `senderRef`, `senderName` — Sender identification (senderRef is the short LLM-friendly reference)
- `contextMsgId`, `messageId`, `timestampMs` — Message identification (contextMsgId is 6-digit per-chat sequence)
- `text`, `quoted`, `attachments`, `mentionedJids` — Message content
- `botMentioned`, `repliedToBot` — Bot-mention and reply signals for LLM1 routing
- `slashCommand`, `commandHandled` — Slash command detection and whether it was already handled by Node
- `groupDescription` — Group description for LLM context
- `location` — Location data if present

### Control events (sent via `sendReliable()`)

| Event | Direction | Purpose |
|-------|-----------|---------|
| `clear_history` | Node → Python | Clear per-chat history |
| `set_llm2_model` | Node → Python | Authoritative model sync for a chat |
| `invalidate_llm2_model` | Node → Python | Clear cached model for a chat |
| `invalidate_default_model` | Node → Python | Clear cached default model |
| `whatsapp_status` | Node → Python | WhatsApp connection state (open/closed) |
| `hello` | Node → Python | Handshake after WS connection opens |

## Python → Node actions

| Action | Required fields | Description |
|--------|----------------|-------------|
| `send_message` | `chatId`, `text` | Send text/media reply |
| `react_message` | `chatId`, `contextMsgId`, `emoji` | React to a message |
| `delete_message` | `chatId`, `contextMsgId` | Delete a message |
| `kick_member` | `chatId`, `targets[]` | Remove members from group |
| `mark_read` | `chatId`, `messageId` | Mark messages as read |
| `send_presence` | `chatId`, `type` | Send typing/paused indicator |
| `send_buttons` | `chatId`, `text`, `buttons` | NativeFlow button message |
| `send_carousel` | `chatId`, `cards[]` | Carousel card message |
| `run_command` | `chatId`, `command`, `contextMsgId?` | Execute a slash command silently on the gateway (not posted to WhatsApp). Used by `reply_message`'s optional `command` parameter. |

## Ack/Error responses (Node → Python)

| Type | Fields | Description |
|------|--------|-------------|
| `action_ack` | `requestId`, `action`, `ok`, `detail`, `result?` | Action result confirmation |
| `send_ack` | `requestId` | Legacy `send_message` confirmation |
| `error` | `message`, `detail`, `code`, `requestId?`, `action?` | Action failure (stable codes: `not_found`, `not_group`, `permission_denied`, `invalid_target`, `send_failed`) |

## Reliability contract
- Critical control events from Node to Python **must** use `sendReliable()` to survive WS reconnects.
- If WS is not open, reliable events are stored in an in-memory queue (max 1000 entries, oldest dropped on overflow).
- The queue is flushed when the connection reopens.
- Regular `incoming_message` events use `send()` (best-effort) because they're transient.

## Full payload reference
See `README.md` for the complete `incoming_message` and `send_message` payload contracts.