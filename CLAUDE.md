# CLAUDE.md

## Project Overview

WazzapAgents is a WhatsApp-to-LLM gateway. It connects a WhatsApp account via Baileys v7 (Node.js) and forwards messages to an LLM service over WebSocket. A Python bridge handles LLM decision-routing and response generation.

## Architecture

Two runtime components communicate over WebSocket:

1. **Node.js Gateway** (`src/`) — Connects to WhatsApp via Baileys, forwards messages to LLM bridge, executes moderation actions (send, react, delete, kick, mark read, typing indicator).
2. **Python LLM Bridge** (`python/bridge/`) — Receives messages, runs two-stage LLM pipeline (LLM1 for gating/decision, LLM2 for response generation), handles slash commands, manages chat history and settings via SQLite.

Additional directories and files:
- `examples/` — Example LLM WebSocket echo server (`llm_ws_echo.py`).
- `python/systemprompt.txt` — LLM2 system prompt template (reply protocol, identity, formatting rules).
- `website/` — Docusaurus documentation site (Indonesian primary, English i18n). Deployed via GitHub Actions to GitHub Pages.
- `data/` — Runtime artifacts (created automatically): `auth/` (Baileys session), `media/` (downloaded media), `stickers/` (sticker images for bot to send).

## Key Source Files

### Node.js Gateway (`src/`)
| File | Purpose |
|------|---------|
| `index.js` | Bootstrap, routes LLM commands to WhatsApp actions |
| `wsClient.js` | Outbound WebSocket to LLM bridge |
| `config.js` | Environment variable loading |
| `logger.js` | Pino-based structured logging |
| `utils.js` | Stream utilities (`streamToBuffer`, `streamToFile`) |
| `mediaHandler.js` | Media download/validation, MIME inference, `MEDIA_DIR` sandboxing |
| `messageParser.js` | Baileys message unwrapping; extracts mentions, quoted messages, media, locations, vCards |
| `identifiers.js` | `contextMsgId` (6-digit per-chat counter) and `senderRef` (deterministic short IDs) management |
| `participants.js` | Group participant role mapping, name caching, JID normalization |
| `groupContext.js` | Group metadata caching (60s TTL), bot role tracking |
| `caches.js` | In-memory caches: message cache, metadata TTL, participant names, sender ref registry |

#### WhatsApp modules (`src/wa/`)
| File | Purpose |
|------|---------|
| `index.js` | Barrel re-export for all WhatsApp functionality |
| `connection.js` | WhatsApp connection via Baileys v7, socket lifecycle |
| `inbound.js` | Incoming message handling, mention resolution |
| `outbound.js` | Outgoing message sending, mention rendering. LLM text replies use `sendRichMessage` with AI footer; fallback to `sock.sendMessage` on failure |
| `actions.js` | Reaction and delete actions |
| `moderation.js` | Kick members, moderation workflows |
| `commands.js` | Slash command parsing, `/broadcast`, `/info`, `/debug` |
| `events.js` | Synthetic event emission (group join, bot action context) |
| `presence.js` | Mark read and typing presence |
| `utils.js` | Concurrency, timeout, and regex utilities |

#### Interactive messages (`src/wa/interactive/`)

> **⚠️ Read `src/wa/interactive/README.md` before editing any file in this folder.**
> Interactive messages in Baileys v7 require specific binary node injection and proto
> wrapping — many things silently break without it.

| File | Purpose |
|------|---------|
| `README.md` | Full implementation notes — required reading |
| `sendInteractive.js` | Core `_sendInteractive` helper + NativeFlow functions: `sendRichMessage`, `sendQuickReply`, `sendUrlButtons`, `sendCopyCode`, `sendCombinedButtons`, `sendNativeFlow`, `sendList` |
| `sendCarousel.js` | Carousel / swipeable cards (⚠️ error 479, experimental) |
| `sendButtons.js` | Legacy button formats (`sendLegacyButtons`, `sendTemplate`) |
| `index.js` | Barrel re-export |

### Python Bridge (`python/bridge/`)
| File | Purpose |
|------|---------|
| `main.py` | WebSocket handler, message batching, burst debounce |
| `commands.py` | Slash command parsing (`/prompt`, `/reset`, `/permission`, `/broadcast`, `/mode`, `/trigger`, `/dashboard`) |
| `config.py` | Shared env variable parsing with type-safe helpers |
| `db.py` | SQLite settings storage (prompts, permissions, mode, triggers), dashboard stats tables, thread-safe with in-memory cache |
| `history.py` | `WhatsAppMessage` dataclass, history formatting, `assistant_aliases()` / `assistant_name_pattern()` for prefix mode |
| `media.py` | Visual attachment processing for multimodal LLM input (`build_visual_parts()`) |
| `dashboard.py` | Usage stats tracking (RAM buffer + periodic DB flush), `/dashboard` output formatting |
| `stickers.py` | Sticker catalog scanning from `data/stickers/`, name-to-path resolution |
| `log.py` | Structured logging with contextvars, configurable extras and chat labels |

#### LLM pipeline (`python/bridge/llm/`)
| File | Purpose |
|------|---------|
| `llm1.py` | LLM1 decision/gating stage (LangChain + OpenAI SDK, multimodal input) |
| `llm2.py` | LLM2 response generation (system prompt from `python/systemprompt.txt`) |
| `schemas.py` | LLM1 tool/function schemas, `LLM1Decision` pydantic model |
| `prompt.py` | LLM1 prompt construction and metadata blocks |
| `client.py` | LLM1 client config, target resolution, OpenAI SDK setup |
| `metadata.py` | LLM1 context metadata (mention counts, reply windows, group prompts) |

#### Messaging pipeline (`python/bridge/messaging/`)
| File | Purpose |
|------|---------|
| `processing.py` | Message normalization, history append, burst building, context ID management |
| `filtering.py` | Payload filtering: prefix matching, content checks, LLM1 trigger logic |
| `actions.py` | LLM2 tool-call parsing: `_extract_actions_from_tool_calls` (primary) + legacy control-line fallback `_extract_actions` |
| `gateway.py` | Outbound WebSocket actions: send, delete, kick, react, sticker, typing |
| `moderation.py` | Attachment merging (`_merge_payload_attachments`) |

## Development Commands

### Node.js Gateway
```bash
pnpm install          # Install dependencies (Node 18+, pnpm 9+)
pnpm dev              # Run gateway (same as pnpm start)
pnpm lint             # Placeholder — not configured yet
```

### Python Bridge
```bash
pip install -r requirements.txt        # Install Python deps (Python 3.10+)
python -m python.bridge.main           # Run the bridge
```

### Tests
```bash
python -m pytest python/tests/         # Run Python tests
python -m unittest python/tests/test_llm_context_serialization.py  # Specific test file
```
No Node.js test framework is configured. If adding tests, use `vitest`.

### Documentation Site
```bash
cd website && npm ci && npm run build   # Build Docusaurus site
cd website && npm start                 # Local dev server
```

## Coding Conventions

### JavaScript (Node.js Gateway)
- ESM modules (`"type": "module"` in package.json). Use `import`/`export`, not `require`.
- 2-space indentation, single quotes, no trailing commas.
- Async/await for all asynchronous operations.
- Use `logger` from `src/logger.js` for logging with structured context objects.
- No formatter/linter configured — match existing style and keep diffs minimal.

### Python (Bridge)
- Python 3.10+ with `from __future__ import annotations`.
- Type hints used throughout. Dataclasses for data structures.
- Relative imports within `python/bridge/` package.
- Dependencies: `websockets>=12.0`, `httpx>=0.27.0`, `pydantic>=2.7.0`, `langchain>=0.2.0`, `langchain-openai>=0.1.0`, `python-dotenv>=1.0.1`.

### General
- Paths in payloads stay workspace-relative (`data/media/...`).
- Use `senderRef` (short deterministic ID) for user references, never raw JIDs in LLM-facing code.
- `contextMsgId` is a 6-digit per-chat sequence number used to reference messages.

## Environment Configuration

Copy `.env.example` to `.env`. Required variable:
- `LLM_WS_ENDPOINT` — WebSocket URL for the LLM bridge (e.g., `ws://localhost:8080/ws`).

Key optional variables:
- `INSTANCE_ID` — Gateway instance identifier.
- `BOT_OWNER_JIDS` — Comma-separated owner JIDs for `/broadcast`, `/mode`, `/trigger`.
- `LLM1_*` — LLM1 (gating) provider config: endpoint, model, API key, temperature, timeout, reasoning effort.
- `LLM2_*` — LLM2 (responder) provider config: endpoint, model, API key, temperature, timeout, reasoning effort.
- `LLM1_REASONING_EFFORT`, `LLM2_REASONING_EFFORT` — `low`/`medium`/`high` for models that support reasoning.
- `ASSISTANT_NAME` — Bot display name. Supports comma-separated aliases for prefix mode (e.g., `vivy,ivy,vivi`). First name = display name, rest = trigger aliases.
- `HISTORY_LIMIT`, `INCOMING_DEBOUNCE_SECONDS`, `INCOMING_BURST_MAX_SECONDS` — Bridge batching tuning. Note: debounce is bypassed in prefix mode and private chats.
- `BRIDGE_MAX_TRIGGER_BATCH_AGE_MS` — Max age of a batch in prefix mode before forcing send (default: 45s).
- `BRIDGE_REPLY_DEDUP_WINDOW_MS`, `BRIDGE_REPLY_DEDUP_MIN_CHARS` — Reply deduplication window and minimum character threshold.
- `BRIDGE_ASSISTANT_ECHO_MERGE_WINDOW_MS` — Merge bot's own consecutive messages within this window (default: 180s).
- `BRIDGE_LOG_*` — Python bridge logging configuration.
- `BOT_DB_PATH` — SQLite database path (default: `data/bot.db`).

See `.env.example` for the full list with descriptions.

## Bot Modes

Three response modes per chat, controlled via `/mode` (owner only):

- **`prefix`** (default) — Bot only responds when explicitly invoked: @tagged, replied to, or name mentioned in text. LLM1 is skipped entirely (straight to LLM2). Debounce is bypassed for immediate response. Saves tokens and reduces latency.
- **`auto`** — LLM1 decides whether to respond based on context analysis. Full debounce/burst batching applies.
- **`hybrid`** — Checks prefix triggers first; if matched, responds immediately (skipping LLM1). If no prefix trigger, falls back to LLM1 gating (auto behavior).

**Triggers** (`/trigger`, owner only): Configures what activates the bot in prefix mode.
- `tag` — bot @mentioned
- `reply` — replied to bot message
- `join` — new member joins group
- `name` — bot name/alias found in message text (case-insensitive, uses `ASSISTANT_NAME` aliases)

Default: `tag`, `reply`, `name` enabled (`join` disabled). Private chats always auto-respond regardless of mode.

**Dashboard** (`/dashboard`): Shows per-chat usage stats (daily/weekly/monthly) including messages processed, bot tags, name mentions, LLM1/LLM2 calls and token usage, responses sent, stickers sent, errors, and top users. Stats are buffered in RAM and periodically flushed to SQLite.

## Interactive Messages

Interactive messages (buttons, menus, carousels) require special handling in Baileys v7.
**Always read `src/wa/interactive/README.md` before working on this area.**

Key points:
- `sock.sendMessage` cannot be used for `interactiveMessage` — use `generateWAMessageFromContent` + `sock.relayMessage`.
- `additionalNodes` with `{ biz > interactive(type=native_flow) > native_flow(v=9,name=mixed) }` must be passed to every `relayMessage` call, or WhatsApp shows a "version not supported" error.
- Use `proto.Message.InteractiveMessage.create()` — `.fromObject()` was removed in Baileys v7.
- Wrap content in `viewOnceMessage.message.messageContextInfo.interactiveMessage`.

### `sendRichMessage` — Universal Styled Message

The primary function for sending styled messages from the bot:

```js
import { sendRichMessage } from './wa/interactive/index.js';

await sendRichMessage(sock, jid, {
  title: 'Judul',        // optional — rendered in header (requires media to show visually)
  text: 'Isi pesan',     // body text
  footer: 'Footer',      // optional footer bar
  image: { url: '...' }, // optional header media
  buttons: [...],        // optional NativeFlow buttons
  mentions: [jid1],      // optional @mentions via contextInfo.mentionedJid
  badge: true,           // AI badge (default true; only visible in private chats)
  quoted: msg,           // optional quoted message
});
```

All LLM text replies pass through `sendRichMessage` with `footer: 'Pesan ini dibuat oleh AI'`.

### `/debug` Command (owner only)

```
/debug buttons      → quick_reply, cta_url, cta_copy, cta_call
/debug menu         → single_select dropdown
/debug list         → listMessage
/debug rich         → sendRichMessage (tanpa & dengan tombol)
/debug combined     → semua tipe tombol
/debug broadcast    → preview format pesan broadcast
/debug all          → semua 6 tipe di atas
/debug carousel     → carousel (eksperimental, error 479)
```

## Sticker Action

Place sticker images (`.webp`, `.png`, `.jpg`, `.gif`) in `data/stickers/`. Filenames (without extension) become the sticker catalog, injected into the LLM2 system prompt under `<sticker>`.

LLM2 sends stickers via the `send_sticker` tool call (with `sticker_name` and optional `context_msg_id`). Stickers are sent as standalone messages through the existing `send_message` WebSocket action with a sticker attachment. No Node.js changes required.

## Security Rules

- **Never commit** `data/auth/`, `.env`, or any API keys.
- Treat `LLM_WS_TOKEN`, LLM API keys, and Baileys auth as secrets.
- Media size limits should be respected to avoid OOM.
- Moderation actions (delete/mute/kick) are gated by `/permission` command (DB-backed levels 0–3) and require the bot to be admin.

## WebSocket Protocol

The gateway and LLM bridge communicate via JSON messages over WebSocket. Key message types:

**Gateway → LLM:**
- `incoming_message` — User/group messages with full chat context, attachments, mentions, quoted messages. Key payload fields:
  - `contextOnly` — `true` for bot's own messages (enriches context without triggering response loops).
  - `triggerLlm1` — Whether to run LLM1 gating (`false` for bot context, reactions).
  - `senderIsOwner` — Whether sender is in `BOT_OWNER_JIDS`.
  - `botMentioned`, `repliedToBot` — Prefix mode trigger signals.
  - `mentionedParticipants` — Resolved mentions: `[{ jid, senderRef, name, isBot }]`.
- `incoming_message` with `messageType: "groupParticipantsUpdate"` — Synthetic event when members join/leave/are added. Includes `groupEvent: { action, participants, actorId, actorName, source }`.
- `incoming_message` with `messageType: "actionLog"` — Synthetic bot context event after successful moderation actions.
- `incoming_message` with `messageType: "botRoleChange"` — Emitted when the bot is promoted or demoted in a group. Python bridge sends a notification and resets permissions to 0 on demotion.
- `action_ack` — Success/failure acknowledgement for actions.

**LLM → Gateway:** `send_message`, `react_message`, `delete_message`, `kick_member`, `mark_read`, `send_presence`

**LLM2 Output:** LLM2 uses tool calls for all actions: `reply_message`, `react_to_message`, `send_sticker`, `delete_messages`, `kick_members`, `mute_member`. Moderation tools are injected dynamically based on permission level and bot admin status. A legacy text control-line parser (`_extract_actions`) is kept as a fallback.

See `README.md` for full protocol schema and payload examples.

## CI/CD

- GitHub Actions workflow (`.github/workflows/deploy-docs.yml`) deploys the Docusaurus site to GitHub Pages on push to `main`/`master` when `website/` changes.
- No CI for tests or linting is currently configured.

## Commit Guidelines

- Imperative mood, short prefix: `add`, `fix`, `refactor`.
- If changing the WebSocket protocol, prefix with `protocol:`.
- PRs should include: summary, testing performed, notes on protocol/schema changes.
