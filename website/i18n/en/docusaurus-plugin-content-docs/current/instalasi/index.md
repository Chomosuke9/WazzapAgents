---
sidebar_position: 1
---

# Installation Guide

This guide walks you through installing and running WazzapAgents on a server or local machine.

## Prerequisites

| Software | Version | Notes |
|----------|---------|-------|
| Node.js | 20+ | Required by the Docusaurus website; the gateway also runs on Node 18+ |
| pnpm | 9+ | `npm i -g pnpm` or `corepack enable pnpm` |
| Python | 3.10+ | For the Python bridge |
| SQLite | 3.x | Usually already installed by the OS |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Chomosuke9/WazzapAgents.git && cd WazzapAgents
```

![Clone Repository](/img/2026-05-05-143910_hyprcap.png)

Check that the repository was cloned:

![Check Repository](/img/check_repo.png)

If it worked, you will see the project files:

![Check Repository Success](/img/check_success.png)

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Then edit the file with a text editor, for example:

```bash
nano .env
```

![Nano .env](/img/nano_env.png)

Fill in at least:

```bash
# Required — WebSocket URL to the Python bridge
LLM_WS_ENDPOINT=ws://localhost:8080/ws

# Assistant display name and aliases
ASSISTANT_NAME=LLM

# OpenAI-compatible model endpoint and API key
LLM2_ENDPOINT=
LLM2_API_KEY=
```

Then add your owner JID or LID:

```bash
# Examples: 628123456789@s.whatsapp.net, 193058310034@lid
BOT_OWNER_JIDS=
```

:::note
Each owner can use either a phone JID or an LID. If the phone JID does not work, use `/info` to get the LID.

[See the full LID guide here](/instalasi/cara-mendapatkan-lid).
:::

### 3. Install Dependencies

**Node.js gateway:**

```bash
pnpm install
```

**Python bridge:**

```bash
pip install -r requirements.txt
```

Or with a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run WazzapAgents

Two components must run at the same time:

:::tip
Use a terminal multiplexer such as Tmux, Zellij, or Byobu so each service can keep running in the background.
:::

**Terminal 1 — Python bridge:**

```bash
python -m python.bridge.main
```

**Terminal 2 — Node.js gateway:**

```bash
pnpm dev
```

On first run, the gateway prints a QR code in the terminal. Scan it with WhatsApp to pair the account.

### 5. Add a Model

Add at least one AI model before using the bot. Without a model, the bot cannot do much.

Steps:

1. Make sure your number is registered as an owner. If not, [configure it first](/instalasi/cara-mendapatkan-lid).
2. Send `/modelcfg add` to the bot with this format:

![model add](/img/slash_model_add.jpg)

```txt
/model add <model id>|<model name>|<model description>|<vision support>
```

:::warning
1. Use `|` as the separator.
2. `<model id>` must be accurate; the bot does not automatically fix it.
3. `<vision support>` must be correct. If the model does not support vision, do not set it to `true`, or the bot may stop responding.
:::

### Verify the Model

Send `/setting` to open this menu:

![slash setting](/img/slash_setting.jpg)

Tap `Change model`. If the previous step was correct, you should see:

![slash setting model](/img/gpt_model.jpg)

## Sub-Agent

**Sub-Agent** is a separate helper service for WazzapAgents. It can run longer, tool-heavy tasks such as file processing, data extraction, web scraping, and code execution without blocking the main WhatsApp conversation.

### When to Enable It

Use Sub-Agent when you want the assistant to handle tasks such as:

- Reading or transforming uploaded files
- Extracting tables or summaries from documents
- Running small scripts or command-line tools
- Researching and returning a structured report

Keep it disabled if you only need normal chat replies or moderation.

### 1. Run WazzapSubAgents

Clone and configure the Sub-Agent service:

```bash
git clone https://github.com/Chomosuke9/WazzapSubAgents.git && cd WazzapSubAgents
cp .env.example .env
```

Edit `.env` and set at least:

```bash
LLM_API_KEY=<your API key>
AGENT_MODEL=<model for the sub-agent>
```

Run it with Docker Compose:

```bash
docker-compose up -d
```

The service exposes:

- Main API: `http://localhost:5000`
- Executor sidecar: `http://localhost:5001`

### 2. Connect WazzapAgents to the Sub-Agent

In the WazzapAgents `.env`, configure:

```bash
SUBAGENT_URL=http://localhost:5000
SUBAGENT_WEBHOOK_PORT=8081
SUBAGENT_WEBHOOK_URL=http://localhost:8081/subagent/callback
```

If WazzapSubAgents runs in Docker and needs to call back into WazzapAgents on the host, use:

```bash
SUBAGENT_WEBHOOK_URL=http://host.docker.internal:8081/subagent/callback
```

:::tip
The Docker Compose file in WazzapSubAgents already maps `host.docker.internal` on Linux with `host-gateway`.
:::

### 3. Share Files Between Services

For file tasks, both services must read and write the same host directory. WazzapSubAgents defaults to `/storage`.

```bash
SUBAGENT_STORAGE_DIR=/storage
WORKDIR_BASE=/storage/subagent_work
```

Make sure WazzapAgents can read the files returned by the Sub-Agent, otherwise output attachments cannot be sent back to WhatsApp.

### 4. Enable Sub-Agent in WhatsApp

Only the bot owner can toggle Sub-Agent:

```txt
/subagent on
/subagent off
/subagent global on
/subagent global off
```

Check the current chat setting:

```txt
/subagent
```

### 5. Test the Flow

After both services are running:

1. Send `/subagent on` in a chat.
2. Ask for a task that needs tooling, for example: "Read this document and extract the tables."
3. The main agent should acknowledge the task.
4. The Sub-Agent sends progress callbacks to the webhook.
5. When done, WazzapAgents summarizes the report and sends any output files.

:::warning
Sub-Agent runs tool and file operations. Use a locked-down server, keep API keys private, and only enable it for trusted chats.
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
| `SEND_TIMEOUT_MS` | `60000` | Send message timeout |
| `UPSERT_CONCURRENCY` | `2` | Message processing concurrency |
| `BOT_OWNER_JIDS` | *(empty)* | Owner JIDs/LIDs, comma-separated |

### Bridge (Python)

| Variable | Default | Description |
|----------|---------|-------------|
| `HISTORY_LIMIT` | `20` | History messages per chat |
| `INCOMING_DEBOUNCE_SECONDS` | `5` | Debounce window for batching |
| `INCOMING_BURST_MAX_SECONDS` | `20` | Maximum burst window duration |
| `ASSISTANT_NAME` | `LLM` | Bot display name in context |
| `CONTEXT_TIME_UTC_OFFSET_HOURS` | *(auto)* | UTC offset for timestamps |
| `SUBAGENT_URL` | `http://localhost:5000` | WazzapSubAgents API URL |
| `SUBAGENT_WEBHOOK_PORT` | `8081` | Local callback webhook port |
| `SUBAGENT_WEBHOOK_URL` | `http://localhost:8081/subagent/callback` | Callback URL sent to WazzapSubAgents |
| `SUBAGENT_ENABLED_DEFAULT` | `false` | Enable Sub-Agent by default for new chats |
| `SUBAGENT_WAIT_TIMEOUT_S` | `300` | Maximum wait for Sub-Agent completion callback |

### LLM1 (Routing)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM1_ENDPOINT` | *(OpenAI default)* | LLM1 API endpoint |
| `LLM1_MODEL` | `openai/gpt-oss-20b` | Model for routing |
| `LLM1_API_KEY` | *(empty)* | LLM1 API key |
| `LLM1_TEMPERATURE` | `0` | LLM1 temperature |
| `LLM1_TIMEOUT` | `8` | Timeout in seconds |
| `LLM1_HISTORY_LIMIT` | `20` | History limit for LLM1 context |
| `LLM1_MESSAGE_MAX_CHARS` | `500` | Max chars per message for LLM1 |

### LLM2 (Response)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM2_ENDPOINT` | *(required)* | LLM2 API endpoint |
| `LLM2_MODEL` | *(required)* | Model for responses |
| `LLM2_API_KEY` | *(required)* | LLM2 API key |
| `LLM2_TEMPERATURE` | `0.7` | LLM2 temperature |
| `LLM2_TIMEOUT` | `60` | Timeout in seconds |
| `LLM2_RETRY_MAX` | `1` | Retry count |
