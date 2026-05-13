# WazzapAgents — Developer Context

> This file is read by AI coding agents at the start of every session. It is the
> canonical reference for understanding this project without rediscovery.

---

## Project Overview

**WazzapAgents** is a WhatsApp AI agent system that connects a WhatsApp account to
an LLM service, enabling automated conversation, moderation, and interactive
features in group and private chats.

**Tech stack:**
- **Node.js 18+** (ESM) — WhatsApp gateway via Baileys v7
- **Python 3.10+** — LLM bridge with LangChain / ChatOpenAI
- **SQLite** — per-chat settings, model configs, moderation state, dashboard stats
- **WebSocket** — Node ↔ Python protocol (JSON over WS)

**Architecture at a glance:**

```
WhatsApp (phone)
      ↕  (Baileys v7 socket, multi-file auth)
┌─────────────────────────────────────────┐
│  Node.js Gateway  (src/)                │
│  ├─ WhatsApp socket lifecycle            │
│  ├─ Inbound: message parser → WS send   │
│  ├─ Outbound: WS recv → WA send         │
│  ├─ Slash commands & interactive msgs    │
│  └─ Action dispatcher (react/delete/kick)│
└──────────┬────────────────────────────────┘
           │ WebSocket (LLM_WS_ENDPOINT)
┌──────────▼────────────────────────────────┐
│  Python Bridge  (python/bridge/)          │
│  ├─ Message debounce/batch (5s default)    │
│  ├─ LLM1: should-respond gating router    │
│  ├─ LLM2: response generation             │
│  ├─ Tool extraction → action commands       │
│  └─ Action dispatch back to Node           │
└───────────────────────────────────────────┘
```

---

## Directory Structure

```
src/                          Node.js gateway runtime
  index.js                    Bootstrap: DB init, WA socket, WS client, action dispatcher
  wsClient.js                 WS client: send() (best-effort) / sendReliable() (queued)
  config.js                   Env parsing, runtime paths (data/auth, data/media, etc.)
  logger.js                   Structured pino logger
  utils.js                    Text normalization, ID helpers
  db.js                       SQLite via better-sqlite3: settings, models, stats
  caches.js                   In-memory LRU caches: groups, messages, participants
  mediaHandler.js             Media download from Baileys, validation, path resolution
  messageParser.js            Baileys message unwrapping (viewOnce, interactive, buttons)
  identifiers.js              contextMsgId (6-digit per-chat sequence), senderRef management
  participants.js             Group role/name caching, owner detection
  groupContext.js              Group metadata caching + invalidation
  src/wa/                     WhatsApp modules
    index.js                  Barrel re-export
    connection.js             Baileys v7 socket lifecycle, button/list response handler
    inbound.js                Incoming WA → normalized incoming_message payload
    outbound.js               Send text/media/mentions to WhatsApp
    actions.js                React / delete message wrappers
    moderation.js              Kick members from group
    presence.js                Mark read, typing indicator
    commandHandler.js          Central slash command dispatcher
    commands.js               Command alias normalization
    events.js                 Synthetic context events (action log, group join, role change)
    utils.js                  Concurrency helpers: semaphore, withRetry, escapeRegex
    command/                  Per-command handler modules
      index.js, parseCommand.js
      broadcast.js, info.js, debug.js, join.js, help.js
      prompt.js, reset.js, sticker.js, permission.js
      mode.js, trigger.js, dashboard.js, model.js
      modelcfg.js, setting.js, groupStatus.js, catch.js
    interactive/              Interactive message modules (NativeFlow)
      index.js                Barrel re-export
      sendInteractive.js      Internal: viewOnce + device metadata + relayMessage
      sendButtons.js          Quick reply, CTA URL, copy, call, combined buttons
      sendCarousel.js         Swipeable carousel cards
python/bridge/                Python LLM bridge
  main.py                     WS server on :8080, message batching, debounce, main loop
  config.py                   Env parsing, debounce/burst constants
  db.py                       SQLite CRUD: settings, models, stats, mutes
  history.py                  WhatsAppMessage dataclass, history formatting
  media.py                    Visual attachment processing (base64, size limits)
  stickers.py                 Sticker catalog scanning (data/stickers/)
  commands.py                  Legacy slash command handler (Python side)
  dashboard.py                Stats buffer, 60s flush, dashboard text formatting
  log.py                      Structured logging setup
  llm/                        LLM pipeline
    llm1.py                   Decision router: should-respond / express-only
    llm2.py                   Response generation: reply + tool calls
    schemas.py                Tool schemas (JSON Schema / OpenAI function calling)
    prompt.py                  System prompt assembly, history, metadata injection
    client.py                 LLM client factory, fallback targets
    metadata.py               Context metadata: bot mention, reply signals, window stats
    tool_utils.py             Cross-provider tool-call extraction
  messaging/                  Message processing pipeline
    processing.py             Burst building, payload normalization, dedup
    filtering.py              Trigger check, prefix/trigger mode, echo filtering
    actions.py                 Control line parsing from LLM2 text output
    gateway.py                Send action commands over WS to Node
    moderation.py             Permission checks, moderation payload merge
  tools/                      Tool implementations
    sticker.py                PIL-based sticker creation (text overlay, EXIF metadata)
python/systemprompt.txt       LLM2 system prompt template
data/                         Runtime artifacts (git-ignored)
  auth/                       Baileys multi-file auth state
  media/                      Downloaded inbound media
  stickers/                   Sticker catalog for LLM2 tool
examples/                     Example LLM WebSocket server (llm_ws_echo.py)
website/                     Docusaurus documentation site (Indonesian + English)
docs/llm-architecture/         Architecture docs for LLM/agent developers
  00-overview.md → 05-state-data-and-db.md
```

---

## Key Concepts & Terminology

| Term | Definition |
|------|-----------|
| **contextMsgId** | A 6-digit per-chat monotonically increasing sequence number (`000000`–`999999`). Used as the canonical message reference across the system instead of WhatsApp's opaque `wamid-*` IDs. |
| **senderRef** | A short, deterministic reference string per sender in each chat (e.g., `u8k2d1`). LLM moderation uses this instead of JIDs. |
| **LLM1** | Decision/gating model. Determines whether the bot should respond, express-only (emoji/sticker), or skip. Also called "the router". |
| **LLM2** | Response generation model. Produces the actual text reply plus tool calls (`reply_message`, `delete_messages`, etc.). Also called "the responder". |
| **burst** | A group of messages collected during the debounce window before processing as a batch. |
| **session** | A WhatsApp session (Baileys multi-file auth stored in `data/auth/`). Deleting this forces re-pairing via QR code. |
| **tool** | A function the LLM can invoke, defined as JSON Schema. Permission-gated: `reply_message` and `llm_express` are always available; `delete_messages`, `mute_member`, `kick_members` depend on chat permission level. |
| **route** | Not a formal concept in this codebase. When you see "routing" it refers to LLM1's decision of whether to respond. |
| **context window** | The rolling history of messages passed to the LLM (capped by `HISTORY_LIMIT`, default 20). |
| **interactive message** | A WhatsApp NativeFlow message (buttons, carousels, lists). Requires special protobuf wrapping and binary XML nodes. |
| **action** | A command from Python to Node via WS: `send_message`, `react_message`, `delete_message`, `kick_member`, `mark_read`, `send_presence`, `send_buttons`, `send_carousel`. |
| **action_ack** | Node's confirmation response to an action, containing `{ requestId, action, ok, detail, result? }`. |

---

## Architecture Decisions (ADRs)

### ADR-1: Why `relayMessage()` with `additionalNodes` instead of `sendMessage()`

WhatsApp interactive messages (NativeFlow) don't render correctly via
`sock.sendMessage()`. The `sendMessage` path routes through
`prepareWAMessageMedia`, which throws "Invalid media type" for
`interactiveMessage` content. Instead, we must:

1. Construct the proto using `generateWAMessageFromContent` with a
   `viewOnceMessage` wrapper (not `viewOnceMessageV2` — Baileys v7 removed
   the `fromObject` helper).
2. Inject binary XML nodes via `additionalNodes` in `relayMessage()`. The
   `biz` node marks the message as a business native flow, and the `bot` node
   adds the AI badge in private chats.

Without these nodes, the message sends but WhatsApp renders it as plain text
or silently drops it.

### ADR-2: Why LLM1 is a separate router instead of a tool call

LLM1 runs on every incoming message burst in group chats. It uses a cheap,
fast model to make a binary decision: respond or skip. Keeping this as a
separate LLM call rather than making it a tool within LLM2 was chosen because:

- **Cost**: Most group messages don't need a full LLM2 response. Running a
  cheap model first saves expensive LLM2 tokens on ~70-80% of messages.
- **Latency**: LLM1 is tuned for sub-2s responses; LLM2 can take 5-20s.
- **Isolation**: LLM1's prompt is specialized for routing (confidence scoring,
  express-only detection). Mixing this into LLM2's system prompt would make
  both harder to tune.

If `LLM1_ENDPOINT` is empty, LLM1 is disabled and all messages go to LLM2.

### ADR-3: Why the sticker pipeline uses Pillow (PIL) instead of ffmpeg-only

The Python sticker tool (`python/bridge/tools/sticker.py`) uses Pillow for
image manipulation (square-padding, text overlay with outline, font rendering)
because:

- **Text rendering**: ffmpeg's `drawtext` filter doesn't support multi-line
  word wrapping with per-line measurement. Pillow's `ImageDraw.textbbox`
  allows precise layout.
- **EXIF metadata**: WhatsApp stickers require a custom EXIF payload
  (`sticker-pack-id`, `sticker-pack-name`) that ffmpeg can't embed correctly.
  The Python code builds this binary TIFF structure manually.
- **The Node sticker path** (`src/wa/command/sticker.js`) does use ffmpeg for
  animated stickers (video → WebP conversion) and sharp for static resizing.
  Both approaches converge on the same WhatsApp EXIF format.

### ADR-4: Why `wsClient.sendReliable()` for state-sync events

The WebSocket connection between Node and Python can drop during reconnection.
`send()` drops messages if disconnected; `sendReliable()` queues them and
flushes on reconnect. Used for events that must not be lost:

- `whatsapp_status` (connection state changes)
- `clear_history` (history invalidation)
- `set_llm2_model` / `invalidate_llm2_model` (model changes)
- `invalidate_default_model`

Regular `incoming_message` events use `send()` because they're transient — the
next burst will include newer state anyway.

### ADR-5: Why contextMsgId is a per-chat 6-digit counter

WhatsApp's native message IDs (`wamid-...`) are long, opaque strings that are
hard for LLMs to reference and easy to hallucinate. The contextMsgId system
creates a short, predictable, per-chat monotonically increasing counter that
LLMs can reliably use in tool calls like `reply_message(context_msg_id="000125")`.

### ADR-6: Why separate SQLite databases (settings/stats/moderation)

Three separate SQLite databases avoid locking contention:

- `settings.db` — read-heavy, written by both Node and Python
- `stats.db` — written frequently by Python, read by Node for dashboard
- `moderation.db` — read-heavy, occasional mutes from Python

Each uses `WAL` mode for concurrent reads.

---

## Development Conventions

### How to add a new tool

1. **Define the schema** in `python/bridge/llm/schemas.py` — add a JSON Schema
   function definition following the existing pattern (e.g., `REPLY_MESSAGE_TOOL`).
   Set `strict: true` and include all parameters.
2. **Register the tool** — add it to `build_llm2_tools()` in `schemas.py`,
   gated by the appropriate permission flag if it's a moderation tool.
3. **Parse the tool call** — add extraction logic in
   `python/bridge/messaging/actions.py` (`_extract_actions_from_tool_calls`).
   Map the tool call to an action dict with `type`, `chatId`, etc.
4. **Implement the action handler** — in `python/bridge/messaging/gateway.py`,
   add a `send_<action>()` function that sends the action over WS to Node.
5. **Handle the action in Node** — in `src/index.js`, add a case to
   `dispatchCommand()` that calls the appropriate `src/wa/` module.
6. **Update the protocol** — add the action to the README protocol section
   and document `action_ack`/`error` responses.

### How to add a new LLM provider

1. **LLM1**: Set `LLM1_ENDPOINT` to the provider's OpenAI-compatible base URL
   (e.g., `https://openrouter.ai/api/v1`). Both base URL and full URL formats
   (with `/chat/completions`) are accepted — the suffix is stripped automatically
   if present. Set `LLM1_MODEL` and `LLM1_API_KEY`. For fallback, set
   `LLM1_FALLBACK_ENDPOINT/MODEL/API_KEY`.
2. **LLM2**: Same pattern with `LLM2_*` env vars. Both base URL and full URL
   formats are accepted. The bridge uses `ChatOpenAI` from LangChain, so any
   OpenAI-compatible API works.
3. **Custom providers**: If the provider doesn't follow OpenAI's tool call
   schema, add extraction logic in `python/bridge/llm/tool_utils.py`.

### Environment variables

See `.env.example` for the full list. Required:

| Variable | Description |
|----------|-------------|
| `LLM_WS_ENDPOINT` | WebSocket URL to Python bridge (e.g., `ws://localhost:8080/ws`) |

Key optional variables — grouped by concern:

**Node Gateway:**
`INSTANCE_ID`, `BOT_OWNER_JIDS`, `ASSISTANT_NAME`, `LLM_WS_TOKEN`,
`DATA_DIR`, `MEDIA_DIR`, `LOG_LEVEL`, `WS_RECONNECT_MS`,
`WS_RECONNECT_MAX_MS` (cap for exponential backoff, default 60000),
`WS_RECONNECT_JITTER_RATIO` (+/- jitter fraction 0..1, default 0.2),
`WS_HEARTBEAT_INTERVAL_MS` (ping cadence and detection granularity when connected, default 20000),
`WS_HEARTBEAT_TIMEOUT_MS` (deprecated no-op under the `isAlive` heartbeat pattern; parsed but unused, default 20000),
`GROUP_METADATA_TIMEOUT_MS`, `DOWNLOAD_TIMEOUT_MS`, `SEND_TIMEOUT_MS`,
`UPSERT_CONCURRENCY`, `PERF_LOG_ENABLED`, `PERF_LOG_THRESHOLD_MS`

**Python Bridge:**
`HISTORY_LIMIT`, `INCOMING_DEBOUNCE_SECONDS`, `INCOMING_BURST_MAX_SECONDS`,
`BRIDGE_SLOW_BATCH_LOG_MS`, `BRIDGE_MAX_TRIGGER_BATCH_AGE_MS`

**LLM1 (Router):**
`LLM1_ENDPOINT`, `LLM1_MODEL`, `LLM1_API_KEY`, `LLM1_FALLBACK_ENDPOINT/MODEL/API_KEY`,
`LLM1_TEMPERATURE`, `LLM1_TIMEOUT`

**LLM2 (Responder):**
`LLM2_ENDPOINT`, `LLM2_MODEL`, `LLM2_API_KEY`, `LLM2_FALLBACK_ENDPOINT/MODEL/API_KEY`,
`LLM2_TEMPERATURE`, `LLM2_TIMEOUT`

### Docker

The project doesn't currently include a Dockerfile. To containerize:

- **Node gateway**: `docker build` with Node 18+, copy source, run `pnpm install && pnpm dev`
- **Python bridge**: `docker build` with Python 3.10+, install requirements, run `python -m python.bridge.main`
- Mount `data/` as a volume for auth state persistence across restarts.

---

## Known Gotchas & Footguns

### Baileys session state

- **Auth corruption**: If `data/auth/` is partially written during a crash, delete
  the entire directory and re-pair via QR. Never try to fix it manually.
- **Logged out**: If WhatsApp logs out the session (multi-device limit), the
  gateway logs `"Logged out from WhatsApp"` and stops reconnecting. Delete
  `data/auth/` and restart.
- **Pairing phone**: First run prints a QR code. Scan it quickly — it expires
  in ~20 seconds. If missed, restart the gateway.

### Token usage normalization

LLM1 and LLM2 token counts come from different providers with different
tokenizers. The `usage_metadata` from LangChain may report `input_tokens` and
`output_tokens` that don't match OpenAI's billing tokenizer. Don't rely on
these for exact cost calculation.

### Group chat vs DM behavior differences

- **LLM1 is skipped in private chats** — all DMs get a response (confidence 100).
- **Group chats** use prefix/hybrid/auto modes controlled by `/mode` and
  `/trigger` commands.
- **Permission tools** (`delete_messages`, `mute_member`, `kick_members`) are
  only available if the bot is an admin in the group.
- **Interactive messages** (`sendRichMessage`, `sendCarousel`, etc.) don't render
  on WhatsApp Web — only mobile clients support `viewOnceMessage` interactive
  content.
- **Mentions** in outbound text use the format `@Name (senderRef)`. The
  `renderOutboundMentions()` function resolves these to actual JIDs. Invalid
  senderRef tokens are silently stripped. Use `@all (all)` to tag everyone in a
  group — this sets `nonJidMentions` in the WhatsApp `contextInfo` instead of
  listing every participant JID individually.

### WebSocket reconnection

- If the Python bridge restarts, Node's `wsClient` reconnects with exponential
  backoff + symmetric jitter (`WS_RECONNECT_MS` base, `WS_RECONNECT_MAX_MS` cap,
  `WS_RECONNECT_JITTER_RATIO` +/- spread; the jittered delay is also clamped to
  the cap) and flushes queued `sendReliable()` messages after reconnect. The
  `attempt` counter is reset only after the socket has stayed OPEN for a short
  grace period, so a server that accepts the handshake and kicks immediately
  still sees exponential backoff. A per-connection heartbeat uses the canonical
  `ws`-docs `isAlive` pattern: the interval at `WS_HEARTBEAT_INTERVAL_MS` is
  both pinger and reaper, so the interval itself is the detection granularity
  and there is no second timer to race. `WS_HEARTBEAT_TIMEOUT_MS` is kept for
  backwards compatibility with existing `.env` files but is effectively a
  no-op under the new scheme. This mirrors the Python server's symmetrical
  `ping_interval=20, ping_timeout=20` in `python/bridge/main.py`.
- If Node restarts, Python must reconnect. There's no persistent queue on the
  Python side — in-flight batches are lost.

### Message dedup and ordering

- The Python bridge uses a reply dedup window (`BRIDGE_REPLY_DEDUP_WINDOW_MS`,
  default 2 min) with a minimum character threshold (`BRIDGE_REPLY_DEDUP_MIN_CHARS`,
  default 24) to avoid sending duplicate or near-duplicate LLM2 responses.
- `contextMsgId` wraps at `999999`. The system handles this correctly, but
  don't assume it's globally unique — it's only unique within a chat.

### Sticker creation gotchas

- Node and Python have **separate** sticker pipelines. Node handles the
  `/sticker` slash command (using `sharp` and `ffmpeg`). Python's
  `tools/sticker.py` handles LLM-initiated sticker creation (using Pillow).
- Both converge on the same output format (512×512 WebP with WhatsApp EXIF
  metadata), but they're independent implementations.
- Animated stickers from video have three fallback quality levels to stay under
  the size limit. If all levels fail, the sticker command returns an error.

### Interactive message rendering

- WhatsApp requires the `viewOnceMessage` wrapper AND binary XML
  `additionalNodes` to render interactive messages. Without either, the message
  silently fails to render or appears as plain text.
- The `badge` parameter in `buildInteractiveNodes()` adds the AI indicator in
  private chats. In groups, it's omitted because only business accounts can
  show badges in group chats.

---

## Build, Test, and Development Commands

- **Install Node deps**: `pnpm install` (Node 18+; project is ESM)
- **Install Python deps**: `pip install -r requirements.txt` (Python 3.10+)
- **Run gateway**: `pnpm dev` (same as `pnpm start`) — starts WA socket + WS client
- **Run Python bridge**: `python -m python.bridge.main`
- **Run echo server** (for testing): `pip install websockets==12.* pydantic && python examples/llm_ws_echo.py`
- **Lint**: `pnpm lint` (currently placeholder)
- **Tests**: No test framework wired yet. If adding: `vitest` as dev dependency,
  test files named `*.test.ts|js`, mock all network services.

## Coding Style & Naming Conventions

- **Language**: Modern JavaScript (ESM, Node ≥18). Prefer async/await, top-level imports.
- **Formatting**: 2-space indentation, single quotes, no trailing commas (in JS).
  Python follows PEP 8 with the existing project style.
- **Logging**: Use `logger` from `src/logger.js` (Node) or `bridge/log.py` (Python).
  Prefer structured context objects over string interpolation.
- **Paths in payloads**: Stay workspace-relative (`data/media/...`) as shown in README.
- **Naming**: camelCase in JS, snake_case in Python. Don't mix within the same file.
- **Error handling**: Async functions must propagate errors explicitly or catch and
  log. Never silently swallow errors.

## Commit & Pull Request Guidelines

- **Commit messages**: Imperative mood, short prefix (`add`, `fix`, `refactor`).
  If changing protocol, mention `protocol:` in subject.
- **PRs**: Include summary of changes, testing performed (`pnpm dev` smoke test),
  and notes on protocol/schema changes (e.g., new payload fields).
- **Screenshots/logs**: Only when QR flow or UI is affected.

## Security

- Never commit `data/auth/`. `.env` contains secrets.
- Rotate `LLM_WS_TOKEN`, LLM API keys, and Baileys auth if leaked.
- Media handler enforces size limits (`DOWNLOAD_TIMEOUT_MS`, validation in
  `mediaHandler.js`) to prevent OOM from large WhatsApp media.