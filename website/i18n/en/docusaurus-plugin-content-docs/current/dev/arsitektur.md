---
sidebar_position: 1
---

# Architecture

WazzapAgents consists of two runtime components that communicate over WebSocket:

```
WhatsApp <──Baileys──> Node.js Gateway <──WebSocket──> Python LLM Bridge <──HTTP──> LLM API
```

## Main Components

### 1. Node.js Gateway (`src/`)

The gateway is responsible for:

- **WhatsApp connection** — Uses Baileys v7 to connect to WhatsApp via multi-device protocol.
- **Message parsing** — Extracts text, media, mentions, quoted messages, locations, and vCards from raw Baileys messages.
- **Forwarding to bridge** — Sends `incoming_message` payloads to the Python bridge via WebSocket.
- **Action execution** — Receives commands from the bridge (send, react, delete, kick, mark read, typing) and executes them on WhatsApp.
- **Caching** — Stores message cache, group metadata (60s TTL), participant names, and sender ref registry in memory.

### 2. Python LLM Bridge (`python/bridge/`)

The bridge is responsible for:

- **WebSocket server** — Receives messages from the gateway and sends commands back.
- **Message batching** — Groups incoming messages in burst windows with debounce logic.
- **Two-stage LLM pipeline:**
  - **LLM1 (Gating)** — Decides whether the bot should respond. Lightweight and fast.
  - **LLM2 (Responder)** — Generates complete responses with conversation context and system prompt.
- **Slash commands** — Handles `/prompt`, `/reset`, `/permission` directly.
- **Storage** — SQLite for per-chat settings (custom prompts, permission levels).
- **History management** — Stores conversation history per chat in memory with configurable limits.

## Data Flow

### Incoming Message (User → Bot)

```
1. User sends message on WhatsApp
2. Baileys receives `messages.upsert` event
3. Gateway parses message (messageParser.js)
4. Gateway assigns contextMsgId & senderRef (identifiers.js)
5. Gateway sends `incoming_message` to bridge via WebSocket
6. Bridge batches messages (5s debounce, 20s max burst)
7. Bridge runs LLM1 (gating decision)
8. If LLM1 decides to respond → run LLM2
9. LLM2 generates response
10. Bridge parses actions from response (REPLY_TO, DELETE, KICK, REACT_TO)
11. Bridge sends commands to gateway via WebSocket
12. Gateway executes actions on WhatsApp
```

### Context Messages (Bot → Bridge)

Messages sent by the bot itself are also forwarded to the bridge as `contextOnly: true` and `triggerLlm1: false`. This enriches conversation context without causing loops.

## Message Identification

### contextMsgId

A 6-digit per-chat counter (`000000`–`999999`, wraps after `999999`). Used to reference messages in conversations — for example when the bot needs to reply to a specific message or delete a message.

### senderRef

A short deterministic ID per sender per chat, generated from SHA-1 hash of `chatId|senderId`. Used in all LLM interactions — **never** exposes real JIDs to the LLM.

## Data Storage

| Data | Location | Type |
|------|----------|------|
| WhatsApp session | `data/auth/` | Files (Baileys auth state) |
| Downloaded media | `data/media/` | Files (images, videos, etc.) |
| Chat settings | `data/bot.db` | SQLite |
| Conversation history | Memory (RAM) | In-memory deque |
| Message cache | Memory (RAM) | In-memory Map |
| Group metadata | Memory (RAM) | TTL cache (60 seconds) |

## Module Diagram

### Node.js Gateway

```
index.js          ← Bootstrap, routes WS commands to WhatsApp actions
├── waClient.js   ← WhatsApp connection, send/receive/moderation
├── wsClient.js   ← WebSocket client to bridge (auto-reconnect)
├── config.js     ← Environment variable loading
├── logger.js     ← Pino structured logging
├── messageParser.js  ← Baileys message parsing → structured payload
├── mediaHandler.js   ← Media download & validation, MIME inference
├── identifiers.js    ← contextMsgId counter, senderRef registry
├── participants.js   ← Participant role mapping, name cache
├── groupContext.js   ← Group metadata cache
├── caches.js         ← In-memory caches (message, metadata, names)
└── utils.js          ← Stream utilities (streamToBuffer, streamToFile)
```

### Python Bridge

```
main.py       ← WebSocket handler, batching, pipeline orchestration
├── llm1.py   ← LLM1 gating/decision (LangChain + OpenAI SDK)
├── llm2.py   ← LLM2 response generation
├── commands.py ← Slash command parser (/prompt, /reset, /permission)
├── config.py   ← Environment variable parsing, config constants
├── db.py       ← SQLite storage with in-memory cache
├── history.py  ← WhatsAppMessage dataclass, history formatting
├── media.py    ← Visual attachment processing for multimodal
└── log.py      ← Structured logging with contextvars
```
