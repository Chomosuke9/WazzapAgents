---
sidebar_position: 2
---

# Development Setup

Guide for setting up the WazzapAgents development environment.

## Prerequisites

| Software | Version | Notes |
|----------|---------|-------|
| Node.js | 18+ | Tested with Node 25 |
| pnpm | 9+ | `npm i -g pnpm` or `corepack enable pnpm` |
| Python | 3.10+ | For the bridge |
| SQLite | 3.x | Usually pre-installed on most OS |

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Chomosuke9/WazzapAgents.git
cd WazzapAgents
```

### 2. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

```bash
# Required — WebSocket URL to the Python bridge
LLM_WS_ENDPOINT=ws://localhost:8080/ws

# Optional — API keys for LLM providers
LLM1_API_KEY=sk-...
LLM2_API_KEY=sk-...
```

### 3. Install Dependencies — Node.js Gateway

```bash
pnpm install
```

### 4. Install Dependencies — Python Bridge

```bash
pip install -r requirements.txt
```

Or with a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Running

Both components must run simultaneously:

**Terminal 1 — Python Bridge:**
```bash
python -m python.bridge.main
```

**Terminal 2 — Node.js Gateway:**
```bash
pnpm dev
```

On first run, the gateway will display a QR code in the terminal. Scan it with WhatsApp to pair.

:::tip
If you only want to test the gateway without a real LLM, use the echo server:
```bash
pip install websockets pydantic
python examples/llm_ws_echo.py
```
:::

## Environment Variables

### Gateway (Node.js)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_WS_ENDPOINT` | *(required)* | WebSocket URL to bridge |
| `INSTANCE_ID` | `default` | Gateway instance identifier |
| `LLM_WS_TOKEN` | *(empty)* | Bearer token for WS authentication |
| `DATA_DIR` | `./data` | Runtime data directory |
| `MEDIA_DIR` | `./data/media` | Media storage directory |
| `LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `WS_RECONNECT_MS` | `5000` | WS reconnect interval in ms |
| `GROUP_METADATA_TIMEOUT_MS` | `8000` | Group metadata fetch timeout |
| `DOWNLOAD_TIMEOUT_MS` | `60000` | Media download timeout |
| `SEND_TIMEOUT_MS` | `60000` | Message send timeout |
| `UPSERT_CONCURRENCY` | `5` | Message processing concurrency |
| `BOT_OWNER_JIDS` | *(empty)* | Owner JIDs, comma-separated |

### Bridge (Python)

| Variable | Default | Description |
|----------|---------|-------------|
| `HISTORY_LIMIT` | `20` | History messages per chat |
| `INCOMING_DEBOUNCE_SECONDS` | `5` | Debounce window for batching |
| `INCOMING_BURST_MAX_SECONDS` | `20` | Maximum burst window duration |
| `BOT_DB_PATH` | `data/bot.db` | SQLite database path |
| `ASSISTANT_NAME` | `LLM` | Bot display name in context |
| `CONTEXT_TIME_UTC_OFFSET_HOURS` | *(auto)* | UTC offset for timestamps |

### LLM1 (Gating)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM1_ENDPOINT` | *(OpenAI default)* | LLM1 API endpoint |
| `LLM1_MODEL` | `openai/gpt-oss-20b` | Model for gating |
| `LLM1_API_KEY` | *(empty)* | LLM1 API key |
| `LLM1_TEMPERATURE` | `0` | LLM1 temperature |
| `LLM1_TIMEOUT` | `8` | Timeout in seconds |
| `LLM1_HISTORY_LIMIT` | `20` | History limit for LLM1 context |
| `LLM1_MESSAGE_MAX_CHARS` | `500` | Max chars per message for LLM1 |
| `LLM1_ENABLE_MEDIA_INPUT` | `0` | Enable multimodal LLM1 input |
| `LLM1_FALLBACK_ENDPOINT` | *(reuse LLM1)* | Fallback endpoint |
| `LLM1_FALLBACK_MODEL` | *(empty)* | Fallback model |

### LLM2 (Responder)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM2_ENDPOINT` | *(OpenAI default)* | LLM2 API endpoint |
| `LLM2_MODEL` | `gpt-4.1` | Model for responder |
| `LLM2_API_KEY` | *(empty)* | LLM2 API key |
| `LLM2_TEMPERATURE` | `0.5` | LLM2 temperature |
| `LLM2_TIMEOUT` | `20` | Timeout in seconds |
| `LLM2_RETRY_MAX` | `0` | Max retries on timeout |
| `LLM2_RETRY_BACKOFF_SECONDS` | `0.8` | Backoff between retries |
| `LLM2_ENABLE_MEDIA_INPUT` | `1` | Enable multimodal LLM2 input |
| `LLM2_FALLBACK_ENDPOINT` | *(reuse LLM2)* | Fallback endpoint |
| `LLM2_FALLBACK_MODEL` | *(empty)* | Fallback model |

### Bridge Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIDGE_LOG_LEVEL` | `info` | Bridge log level |
| `BRIDGE_LOG_PROMPT_FULL` | `0` | Log full LLM2 prompt |
| `BRIDGE_LOG_EXTRAS_LIMIT` | `4000` | Extras character limit in logs |
| `BRIDGE_LOG_CHAT_LABEL_WIDTH` | `24` | Chat label width in logs |
| `BRIDGE_SLOW_BATCH_LOG_MS` | `2000` | Slow batch log threshold |

## Running Tests

```bash
# All Python tests
python -m pytest python/tests/

# Specific test
python -m unittest python/tests/test_llm_context_serialization.py
```

:::info
No Node.js test framework is configured yet. If adding tests for the gateway, use **vitest**.
:::

## Building Documentation

```bash
cd website
npm ci
npm run build    # Production build
npm start        # Local dev server
```

## Directory Structure

```
WazzapAgents/
├── src/                    # Node.js Gateway
│   ├── index.js            # Entry point
│   ├── waClient.js         # WhatsApp client
│   ├── wsClient.js         # WebSocket client
│   ├── config.js           # Configuration
│   ├── messageParser.js    # Message parser
│   ├── mediaHandler.js     # Media handler
│   ├── identifiers.js      # contextMsgId & senderRef
│   ├── participants.js     # Participant data
│   ├── groupContext.js     # Group context
│   ├── caches.js           # In-memory caches
│   ├── logger.js           # Logging
│   └── utils.js            # Utilities
├── python/
│   ├── bridge/             # Python LLM Bridge
│   │   ├── main.py         # Entry point
│   │   ├── llm1.py         # LLM1 gating
│   │   ├── llm2.py         # LLM2 responder
│   │   ├── commands.py     # Slash commands
│   │   ├── config.py       # Configuration
│   │   ├── db.py           # Database
│   │   ├── history.py      # History management
│   │   ├── media.py        # Media processing
│   │   └── log.py          # Logging
│   ├── systemprompt.txt    # LLM2 system prompt template
│   └── tests/              # Unit tests
├── examples/
│   └── llm_ws_echo.py      # Example echo server
├── website/                # Docusaurus docs
├── data/                   # Runtime data (auto-created)
│   ├── auth/               # WhatsApp session
│   ├── media/              # Media files
│   └── bot.db              # SQLite database
├── .env.example            # Env template
├── package.json            # Node.js deps
└── requirements.txt        # Python deps
```
