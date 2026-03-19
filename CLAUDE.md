# CLAUDE.md

## Project Overview

WazzapAgents is a WhatsApp-to-LLM gateway. It connects a WhatsApp account via Baileys v7 (Node.js) and forwards messages to an LLM service over WebSocket. A Python bridge handles LLM decision-routing and response generation.

## Architecture

Two runtime components communicate over WebSocket:

1. **Node.js Gateway** (`src/`) — Connects to WhatsApp via Baileys, forwards messages to LLM bridge, executes moderation actions (send, react, delete, kick, mark read, typing indicator).
2. **Python LLM Bridge** (`python/bridge/`) — Receives messages, runs two-stage LLM pipeline (LLM1 for gating/decision, LLM2 for response generation), handles slash commands, manages chat history and settings via SQLite.

Additional directories:
- `examples/` — Example LLM WebSocket echo server (`llm_ws_echo.py`).
- `website/` — Docusaurus documentation site (Indonesian primary, English i18n). Deployed via GitHub Actions to GitHub Pages.
- `data/` — Runtime artifacts (created automatically): `auth/` (Baileys session), `media/` (downloaded media).

## Key Source Files

### Node.js Gateway (`src/`)
| File | Purpose |
|------|---------|
| `index.js` | Bootstrap, routes LLM commands to WhatsApp actions |
| `waClient.js` | WhatsApp connection via Baileys v7, message send/receive/moderation |
| `wsClient.js` | Outbound WebSocket to LLM bridge |
| `config.js` | Environment variable loading |
| `logger.js` | Pino-based structured logging |
| `utils.js` | Shared utilities |

### Python Bridge (`python/bridge/`)
| File | Purpose |
|------|---------|
| `main.py` | WebSocket handler, message batching, burst debounce |
| `llm1.py` | LLM1 decision/gating stage |
| `llm2.py` | LLM2 response generation |
| `commands.py` | Slash command parsing and handling |
| `db.py` | SQLite settings storage (prompts, permissions) |
| `history.py` | Message dataclass, history formatting |
| `media.py` | Media/attachment processing for LLM input |
| `log.py` | Structured logging setup |

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
- Dependencies: `websockets`, `httpx`, `pydantic`, `langchain`, `langchain-openai`, `python-dotenv`.

### General
- Paths in payloads stay workspace-relative (`data/media/...`).
- Use `senderRef` (short deterministic ID) for user references, never raw JIDs in LLM-facing code.
- `contextMsgId` is a 6-digit per-chat sequence number used to reference messages.

## Environment Configuration

Copy `.env.example` to `.env`. Required variable:
- `LLM_WS_ENDPOINT` — WebSocket URL for the LLM bridge (e.g., `ws://localhost:8080/ws`).

Key optional variables:
- `INSTANCE_ID` — Gateway instance identifier.
- `BOT_OWNER_JIDS` — Comma-separated owner JIDs for `/broadcast`.
- `LLM1_*` — LLM1 (gating) provider config: endpoint, model, API key, temperature, timeout.
- `LLM2_*` — LLM2 (responder) provider config: endpoint, model, API key, temperature, timeout.
- `HISTORY_LIMIT`, `INCOMING_DEBOUNCE_SECONDS`, `INCOMING_BURST_MAX_SECONDS` — Bridge batching tuning.
- `BRIDGE_LOG_*` — Python bridge logging configuration.
- `BOT_DB_PATH` — SQLite database path (default: `data/bot.db`).

See `.env.example` for the full list with descriptions.

## Security Rules

- **Never commit** `data/auth/`, `.env`, or any API keys.
- Treat `LLM_WS_TOKEN`, LLM API keys, and Baileys auth as secrets.
- Media size limits should be respected to avoid OOM.
- Moderation actions (DELETE/KICK) are gated by `<prompt_override>` flags in group descriptions.

## WebSocket Protocol

The gateway and LLM bridge communicate via JSON messages over WebSocket. Key message types:

**Gateway → LLM:** `incoming_message` (with full chat context, attachments, mentions, quoted messages)

**LLM → Gateway:** `send_message`, `react_message`, `delete_message`, `kick_member`, `mark_read`, `send_presence`

**Gateway → LLM:** `action_ack` (success/failure acknowledgement for actions)

See `README.md` for full protocol schema and payload examples.

## CI/CD

- GitHub Actions workflow (`.github/workflows/deploy-docs.yml`) deploys the Docusaurus site to GitHub Pages on push to `main`/`master` when `website/` changes.
- No CI for tests or linting is currently configured.

## Commit Guidelines

- Imperative mood, short prefix: `add`, `fix`, `refactor`.
- If changing the WebSocket protocol, prefix with `protocol:`.
- PRs should include: summary, testing performed, notes on protocol/schema changes.
