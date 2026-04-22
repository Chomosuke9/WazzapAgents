# 00 - System Overview

## Core components

### 1) Node Gateway (`src/`)
Primary responsibilities:
- WhatsApp connection via Baileys v7 socket.
- Incoming message parsing + media normalization.
- Slash command handling and interactive message buttons/lists.
- Sending `incoming_message` events to the Python bridge over WebSocket.
- Executing actions received from Python (`send_message`, `delete_message`, `kick_member`, etc.).

### 2) Python Bridge (`python/bridge/`)
Primary responsibilities:
- Receiving `incoming_message` payloads from Node.
- Batching/debouncing per chat, trigger filtering, and LLM1 routing.
- Generating replies and tool calls via LLM2.
- Sending action commands back to Node over WebSocket.
- Writing dashboard statistics (in-memory buffer + periodic DB flush).

## Data flow (high level)
1. User sends a message on WhatsApp.
2. Node parses the message and sends `incoming_message` to Python.
3. Python decides whether to respond (LLM1/LLM2) and what moderation actions to take.
4. Python sends action commands to Node.
5. Node executes actions on WhatsApp and sends `action_ack`/`error` back to Python.

## Command design
- Slash commands are parsed in Node (`src/wa/commands.js`) with singular/plural aliases.
- Aliases are normalized to the canonical command name:
  - `/model` and `/models` → `model`
  - `/setting` and `/settings` → `setting`
- The main command handler lives in `src/wa/commandHandler.js` with per-command modules in `src/wa/command/`.

## Interactive UI
- `/setting` and `/model` menus use NativeFlow buttons/lists (mobile-only, not visible on WhatsApp Web).
- Button clicks produce a `selectedId` like `model_select:<id>`.
- See ADR-1 in AGENTS.md for why `relayMessage` + `additionalNodes` is required.

## Key reliability points
- `wsClient.send()` = best-effort (drops if WS not open).
- `wsClient.sendReliable()` = queued, then flushed on reconnect.
- Critical control events (clear history, invalidate cache, set model, status) **must** use the reliable path.