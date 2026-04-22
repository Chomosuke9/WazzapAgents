# 05 - State, Data, and DB

## Runtime state (in-memory)

### Node side
- **Group metadata cache** — LRU-cached group names, descriptions, participant lists, and admin roles. Invalidated on `groups.update` and `group-participants.update` events.
- **Message ID / contextMsgId index** — Maps `contextMsgId` → message key, and `messageId` → `contextMsgId`. Used for reply targeting and action resolution (react, delete, kick).
- **senderRef registry** — Per-chat bidirectional mapping between JIDs and short senderRef tokens. Rebuilt from incoming messages on reconnect.
- **Reliable WS queue** — Outbound control events queued for delivery when WS reconnects.

### Python side
- **Per-chat history** — Rolling deque of `WhatsAppMessage` objects (capped by `HISTORY_LIMIT`, default 20).
- **Pending burst buffers** — Per-chat message accumulation during debounce window.
- **Dashboard counters** — In-memory stats buffer, flushed to SQLite every 60 seconds.
- **DB read caches** — TTL-based caches for prompt, permission, mode, triggers, and model lookups.

## SQLite databases

The system uses three separate SQLite databases (WAL mode) to avoid locking contention:

| Database | Tables | Primary writer | Primary reader |
|----------|--------|---------------|---------------|
| `settings.db` | `chat_settings`, `llm_models` | Node | Both |
| `stats.db` | `chat_stats`, `chat_user_stats` | Python | Node |
| `moderation.db` | `chat_mutes` | Python | Python |

### Table details

#### `chat_settings` (in `settings.db`)
Stores per-chat configuration:
- `prompt` — Custom system prompt override
- `permission` — Moderation level (0–3)
- `mode` — Trigger mode (`auto`, `prefix`, `hybrid`)
- `triggers` — Comma-separated trigger prefixes
- `llm2_model` — Per-chat model override (NULL = use default)

#### `llm_models` (in `settings.db`)
Model catalog:
- `model_id` — Unique identifier
- `display_name` — Human-friendly name
- `description` — Optional description
- `is_active` — Whether the model is available for selection
- `sort_order` — Determines default model (lowest = default)
- `vision_support` — Whether the model supports image input

#### `chat_stats` (in `stats.db`)
Periodic aggregation of chat activity metrics.

#### `chat_user_stats` (in `stats.db`)
Per-user invocation statistics for dashboard display.

#### `chat_mutes` (in `moderation.db`)
Active mutes per user per chat, with expiration timestamps.

## Environment variable paths
- `DATA_DIR` — Runtime data directory (default: `./data`)
- `MEDIA_DIR` — Downloaded media directory (default: `./data/media`)
- `STICKERS_DIR` — Sticker catalog directory (default: `./data/stickers`)
- `SETTINGS_DB_PATH` — Path to `settings.db` (default: `data/settings.db`)
- `STATS_DB_PATH` — Path to `stats.db` (default: `data/stats.db`)
- `MODERATION_DB_PATH` — Path to `moderation.db` (default: `data/moderation.db`)

## Dashboard notes
- Counters are recorded in RAM first, then flushed to DB in batches.
- If a flush fails, data is requeued so it isn't lost.
- `/dashboard` reads from `stats.db` via Node and formats the response text.