# 02 - Modules Map

## Node side (`src/`)

### Core
- `src/index.js` — Bootstrap app + action dispatcher from Python.
- `src/wsClient.js` — WS client to Python (`send`, `sendReliable`, flush queue).
- `src/config.js` — Env parsing + runtime paths (data/auth/media).
- `src/db.js` — SQLite via sql.js (settings/models/stats reads).
- `src/logger.js` — Structured pino logger.
- `src/identifiers.js` — contextMsgId management, senderRef registry, message index.
- `src/caches.js` — In-memory LRU caches (messages, groups, participants).
- `src/mediaHandler.js` — Media download from Baileys, validation, path resolution.
- `src/messageParser.js` — Baileys message unwrapping (viewOnce, interactive, buttons).
- `src/participants.js` — Group role/name caching, owner detection.
- `src/groupContext.js` — Group metadata caching + invalidation.

### WhatsApp integration (`src/wa/`)
- `connection.js` — Socket init, lifecycle, button/list response handling, command routing.
- `inbound.js` — Normalize incoming WA messages → `incoming_message` payload.
- `outbound.js` — Send text/media/mentions to WhatsApp.
- `actions.js` — React and delete message wrappers.
- `moderation.js` — Kick members from groups.
- `presence.js` — Mark read + typing presence.
- `commands.js` — Slash command parsing + alias normalization.
- `commandHandler.js` — Central command dispatcher.
- `events.js` — Synthetic context events (action log, group join, role change).
- `utils.js` — Concurrency helpers (semaphore, withRetry, escapeRegex).

### Per-command handlers (`src/wa/command/`)
- `parseCommand.js` — Parse `/command args` strings.
- `help.js`, `prompt.js`, `reset.js`, `permission.js`, `mode.js`, `trigger.js` — Configuration commands.
- `dashboard.js` — Stats display.
- `broadcast.js` — Owner-only broadcast to all chats.
- `info.js`, `debug.js` — Diagnostic commands.
- `join.js` — Join group via invite link.
- `sticker.js` — Create sticker from image/video.
- `model.js`, `modelcfg.js` — Per-chat and global model management.
- `setting.js` — Interactive settings menu.
- `groupStatus.js` — Group info.
- `catch.js` — Edit message.

### Interactive messages (`src/wa/interactive/`)
- `sendInteractive.js` — Low-level helper: viewOnce wrapper, device metadata, `relayMessage` with `additionalNodes`.
- `sendButtons.js` — Quick reply, CTA URL, copy code, call, combined buttons.
- `sendCarousel.js` — Swipeable carousel cards.

## Python side (`python/bridge/`)

### Core
- `main.py` — WS handler, batching, main processing loop.
- `db.py` — SQLite access + caches (settings/models/stats/mutes).
- `dashboard.py` — Stats buffer + periodic flush + dashboard text formatting.
- `commands.py` — Legacy slash command parser/handler (Python side).
- `history.py` — `WhatsAppMessage` dataclass, history formatting.
- `config.py` — Env parsing + bridge-level constants.
- `log.py` — Structured logging setup.
- `media.py` — Visual attachment processing (base64, size limits).
- `stickers.py` — Sticker catalog scanning from `data/stickers/`.

### LLM pipeline (`python/bridge/llm/`)
- `llm1.py` — Routing/decision: should the bot respond, express-only, or skip?
- `llm2.py` — Reply generation + tool invocation.
- `schemas.py` — Tool schema definitions (JSON Schema / OpenAI function calling).
- `prompt.py` — Prompt assembly: history, metadata, stickers, prompt override.
- `client.py` — LLM client factory, fallback targets.
- `metadata.py` — Context metadata: bot mention, reply signals, window stats.
- `tool_utils.py` — Cross-provider tool-call extraction.

### Messaging pipeline (`python/bridge/messaging/`)
- `processing.py` — Burst processing, payload normalization, media dedup.
- `filtering.py` — Trigger check, prefix/trigger mode, echo filtering.
- `actions.py` — Parse action lines from LLM2 model output.
- `gateway.py` — Send action commands over WS to Node.
- `moderation.py` — Permission checks, moderation payload merge.

### Tools (`python/bridge/tools/`)
- `sticker.py` — PIL-based sticker creation (text overlay, EXIF metadata).