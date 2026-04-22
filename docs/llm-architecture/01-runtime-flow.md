# 01 - Runtime Flow

## A. Startup flow
1. Node bootstrap (`src/index.js`): init DB, start WhatsApp socket, connect WS client.
2. Python bridge (`python/bridge/main.py`): start WS server, init in-memory state.
3. Node sends `hello` to Python after WS connection opens.

## B. Incoming message flow (WhatsApp → Node → Python)
1. Node receives `messages.upsert` event from Baileys.
2. Node command listener processes slash commands and button responses first.
3. Node inbound pipeline builds the normalized `incoming_message` payload.
4. Node sends payload to Python via WS (`wsClient.send()` — best-effort).
5. Python places the payload into a per-chat pending buffer.
6. After the debounce/burst window expires, Python processes the batch:
   - Trigger check (prefix/hybrid/auto mode)
   - LLM1 decision (should-respond / express-only / skip)
   - LLM2 generation + tool calls (if LLM1 decides to respond)
7. Python sends action commands to Node.
8. Node executes actions on WhatsApp and sends `action_ack`/`error` back to Python.

## C. Model switching flow
1. User selects model from interactive menu (`model_select:<modelId>`).
2. Node sets `chat_settings.llm2_model` in the local SQLite DB.
3. Node sends WS control events via `sendReliable()`:
   - `set_llm2_model` (authoritative sync)
   - `invalidate_llm2_model` (fallback cache clear)
4. Python updates its DB/cache for the chat's model.
5. Subsequent LLM requests use the new model.

## D. Dashboard flow
1. Python records counters (in-memory buffer) during message processing.
2. Python periodically flushes counters to the `stats` SQLite DB.
3. `/dashboard` command in Node reads the stats table and sends a summary to the chat.

## E. Reset flow
1. User runs `/reset`.
2. Node sends `clear_history` via `sendReliable()`.
3. Python clears the per-chat history buffer.

## F. Failure / reconnect behavior
- If WS drops, non-reliable events (`incoming_message`) may be lost — the next burst will include newer state anyway.
- Reliable events are stored in an in-memory queue and re-sent when the connection reopens.
- If the WhatsApp session is logged out, the gateway stops reconnecting and logs `"Logged out from WhatsApp"`. Delete `data/auth/` and restart to re-pair.