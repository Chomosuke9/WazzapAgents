---
sidebar_position: 2
---

# Setup Development

Panduan untuk menyiapkan environment development WazzapAgents.

## Prasyarat

| Software | Versi | Catatan |
|----------|-------|---------|
| Node.js | 18+ | Tested dengan Node 25 |
| pnpm | 9+ | `npm i -g pnpm` atau `corepack enable pnpm` |
| Python | 3.10+ | Untuk bridge |
| SQLite | 3.x | Biasanya sudah terinstall di OS |

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Chomosuke9/WazzapAgents.git
cd WazzapAgents
```

### 2. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` dan isi minimal:

```bash
# Wajib — URL WebSocket ke Python bridge
LLM_WS_ENDPOINT=ws://localhost:8080/ws

# Opsional — API key untuk LLM provider
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

Atau dengan virtual environment (direkomendasikan):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Menjalankan

Dua komponen harus berjalan bersamaan:

**Terminal 1 — Python Bridge:**
```bash
python -m python.bridge.main
```

**Terminal 2 — Node.js Gateway:**
```bash
pnpm dev
```

Saat pertama kali jalan, gateway akan menampilkan QR code di terminal. Scan dengan WhatsApp untuk pairing.

:::tip
Jika hanya ingin menguji gateway tanpa LLM sungguhan, gunakan echo server:
```bash
pip install websockets pydantic
python examples/llm_ws_echo.py
```
:::

## Variabel Environment

### Gateway (Node.js)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `LLM_WS_ENDPOINT` | *(wajib)* | URL WebSocket ke bridge |
| `INSTANCE_ID` | `default` | Identifier instance gateway |
| `LLM_WS_TOKEN` | *(kosong)* | Bearer token untuk autentikasi WS |
| `DATA_DIR` | `./data` | Direktori data runtime |
| `MEDIA_DIR` | `./data/media` | Direktori penyimpanan media |
| `LOG_LEVEL` | `info` | Level log (debug, info, warn, error) |
| `WS_RECONNECT_MS` | `5000` | Interval reconnect WS dalam ms |
| `GROUP_METADATA_TIMEOUT_MS` | `8000` | Timeout fetch metadata grup |
| `DOWNLOAD_TIMEOUT_MS` | `60000` | Timeout download media |
| `SEND_TIMEOUT_MS` | `60000` | Timeout kirim pesan |
| `UPSERT_CONCURRENCY` | `2` | Concurrency pemrosesan pesan |
| `BOT_OWNER_JIDS` | *(kosong)* | JID owner, pisahkan koma |

### Bridge (Python)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `HISTORY_LIMIT` | `20` | Jumlah pesan history per chat |
| `INCOMING_DEBOUNCE_SECONDS` | `5` | Debounce window untuk batching |
| `INCOMING_BURST_MAX_SECONDS` | `20` | Maksimum durasi burst window |
| `HISTORY_LIMIT` | `20` | Jumlah pesan history per chat |
| `INCOMING_DEBOUNCE_SECONDS` | `5` | Debounce window untuk batching |
| `INCOMING_BURST_MAX_SECONDS` | `20` | Maksimum durasi burst window |
| `ASSISTANT_NAME` | `LLM` | Nama tampilan bot di konteks |
| `CONTEXT_TIME_UTC_OFFSET_HOURS` | *(auto)* | UTC offset untuk timestamp |
| `ASSISTANT_NAME` | `LLM` | Nama tampilan bot di konteks |
| `CONTEXT_TIME_UTC_OFFSET_HOURS` | *(auto)* | UTC offset untuk timestamp |

### LLM1 (Gating)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `LLM1_ENDPOINT` | *(OpenAI default)* | Endpoint API LLM1 |
| `LLM1_MODEL` | `openai/gpt-oss-20b` | Model untuk gating |
| `LLM1_API_KEY` | *(kosong)* | API key LLM1 |
| `LLM1_TEMPERATURE` | `0` | Temperature untuk LLM1 |
| `LLM1_TIMEOUT` | `8` | Timeout dalam detik |
| `LLM1_HISTORY_LIMIT` | `20` | Limit history untuk konteks LLM1 |
| `LLM1_MESSAGE_MAX_CHARS` | `500` | Maks karakter per pesan untuk LLM1 |
| `LLM1_ENABLE_MEDIA_INPUT` | `0` | Aktifkan input multimodal LLM1 |
| `LLM1_FALLBACK_ENDPOINT` | *(reuse LLM1)* | Endpoint fallback |
| `LLM1_FALLBACK_MODEL` | *(kosong)* | Model fallback |

### LLM2 (Responder)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `LLM2_ENDPOINT` | *(OpenAI default)* | Endpoint API LLM2 |
| `LLM2_MODEL` | `gpt-5.3` | Model untuk responder |
| `LLM2_API_KEY` | *(kosong)* | API key LLM2 |
| `LLM2_TEMPERATURE` | `0.5` | Temperature untuk LLM2 |
| `LLM2_TIMEOUT` | `20` | Timeout dalam detik |
| `LLM2_RETRY_MAX` | `0` | Maks retry saat timeout |
| `LLM2_RETRY_BACKOFF_SECONDS` | `0.8` | Backoff antar retry |
| `LLM2_ENABLE_MEDIA_INPUT` | `1` | Aktifkan input multimodal LLM2 |
| `LLM2_FALLBACK_ENDPOINT` | *(reuse LLM2)* | Endpoint fallback |
| `LLM2_FALLBACK_MODEL` | *(kosong)* | Model fallback |

### Logging Bridge

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `BRIDGE_LOG_LEVEL` | `info` | Level log bridge |
| `BRIDGE_LOG_PROMPT_FULL` | `0` | Log prompt LLM2 lengkap |
| `BRIDGE_LOG_EXTRAS_LIMIT` | `4000` | Limit karakter extras di log |
| `BRIDGE_LOG_CHAT_LABEL_WIDTH` | `24` | Lebar label chat di log |
| `BRIDGE_SLOW_BATCH_LOG_MS` | `2000` | Threshold log batch lambat |

## Menjalankan Tests

```bash
# Semua tests Python
python -m pytest python/tests/

# Test spesifik
python -m unittest python/tests/test_llm_context_serialization.py
```

:::info
Belum ada test framework Node.js yang dikonfigurasi. Jika menambahkan tests untuk gateway, gunakan **vitest**.
:::

## Build Dokumentasi

```bash
cd website
npm ci
npm run build    # Build production
npm start        # Dev server lokal
```

## Struktur Direktori

```
WazzapAgents/
├── src/                        # Node.js Gateway
│   ├── index.js                # Entry point
│   ├── wsClient.js             # WebSocket client (auto-reconnect)
│   ├── config.js               # Konfigurasi
│   ├── logger.js                # Logging
│   ├── messageParser.js         # Parser pesan Baileys
│   ├── mediaHandler.js          # Handler media
│   ├── identifiers.js           # contextMsgId & senderRef
│   ├── participants.js          # Data partisipan
│   ├── groupContext.js           # Konteks grup
│   ├── caches.js                # In-memory caches
│   ├── db.js                    # SQLite (settings, models, stats)
│   ├── utils.js                 # Utilitas
│   └── wa/                      # WhatsApp modules
│       ├── connection.js        # Socket lifecycle
│       ├── inbound.js           # Pesan masuk → payload
│       ├── outbound.js          # Kirim pesan/media
│       ├── actions.js           # React & delete
│       ├── moderation.js       # Kick members
│       ├── presence.js          # Mark read & typing
│       ├── commandHandler.js    # Command dispatcher
│       ├── commands.js          # Alias normalization
│       ├── events.js            # Synthetic context events
│       ├── utils.js              # Concurrency helpers
│       ├── command/             # Per-command handlers
│       │   ├── help.js, prompt.js, reset.js, permission.js
│       │   ├── mode.js, trigger.js, dashboard.js, model.js
│       │   ├── broadcast.js, info.js, debug.js, join.js
│       │   ├── sticker.js, modelcfg.js, setting.js
│       │   └── groupStatus.js, catch.js
│       └── interactive/        # NativeFlow messages
│           ├── sendInteractive.js
│           ├── sendButtons.js
│           └── sendCarousel.js
├── python/
│   ├── bridge/                  # Python LLM Bridge
│   │   ├── main.py              # Entry point + WS server
│   │   ├── config.py            # Konfigurasi
│   │   ├── db.py                # Database (3 SQLite files)
│   │   ├── history.py           # History management
│   │   ├── media.py             # Media processing
│   │   ├── stickers.py          # Sticker catalog
│   │   ├── commands.py           # Slash commands
│   │   ├── dashboard.py          # Stats buffer + flush
│   │   ├── log.py                # Logging
│   │   ├── llm/                  # LLM pipeline
│   │   │   ├── llm1.py          # Gating decision
│   │   │   ├── llm2.py          # Response generation
│   │   │   ├── schemas.py        # Tool schemas
│   │   │   ├── prompt.py         # Prompt assembly
│   │   │   ├── client.py         # Client factory
│   │   │   ├── metadata.py       # Context metadata
│   │   │   └── tool_utils.py     # Tool extraction
│   │   ├── messaging/            # Message pipeline
│   │   │   ├── processing.py    # Burst building
│   │   │   ├── filtering.py     # Trigger logic
│   │   │   ├── actions.py        # Action parsing
│   │   │   ├── gateway.py       # WS actions
│   │   │   └── moderation.py    # Permission checks
│   │   └── tools/
│   │       └── sticker.py        # PIL sticker creation
│   └── systemprompt.txt          # Template system prompt LLM2
├── examples/
│   └── llm_ws_echo.py          # Echo server contoh
├── docs/llm-architecture/       # Architecture docs
├── website/                     # Docusaurus docs (Indonesian + English)
├── data/                        # Runtime data (auto-created, git-ignored)
│   ├── auth/                    # Session WhatsApp
│   ├── media/                   # Media files
│   ├── stickers/                # Sticker catalog
│   ├── settings.db              # Chat settings & model configs
│   ├── stats.db                 # Dashboard statistics
│   └── moderation.db            # Mute state
├── .env.example                  # Template env
├── AGENTS.md                     # Developer context for AI agents
├── llms.txt                      # LLM-friendly project summary
├── README.md                     # Protocol contract & setup guide
├── package.json                  # Node.js deps
└── requirements.txt              # Python deps
```
