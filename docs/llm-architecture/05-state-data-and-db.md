# 05 - State, Data, and DB

## Current storage overview

### Runtime state (memory)
- Node:
  - cached group metadata
  - message ID/context mapping
  - reliable WS queue (outbound control events)
- Python:
  - per-chat history in RAM
  - pending burst buffers
  - dashboard counters buffer
  - DB read caches (prompt/permission/mode/triggers/model)

## SQLite tables (current)
Saat ini utama ada di `data/bot.db`:
- `chat_settings` – prompt/permission/mode/triggers/llm2_model per chat.
- `llm_models` – daftar model + default via `sort_order`.
- `chat_stats` – agregat stats periodik.
- `chat_user_stats` – top user invoke stats.
- `chat_mutes` – mute state user per chat.

## Read/Write responsibility
- Node:
  - write: `chat_settings`, `llm_models`
  - read: `chat_stats`, `chat_user_stats` (untuk `/dashboard`)
- Python:
  - write/read semua tabel di atas
  - stats ditulis dari `dashboard.py` via flush periodik

## Dashboard notes
- Counter direkam di RAM terlebih dulu, lalu flush batch ke DB.
- Jika flush gagal, data harus kembali ke buffer (requeue) agar tidak hilang.

## Target split DB (next phase)
Disarankan memecah DB menjadi domain:
- `settings.db`: `chat_settings`, `llm_models`
- `stats.db`: `chat_stats`, `chat_user_stats`
- `moderation.db`: `chat_mutes`

Alasan:
- mengurangi contention antar writer,
- memudahkan observability dan backup,
- siap untuk menambah DB domain baru.

## Env/path penting
- `DATA_DIR`, `MEDIA_DIR`, `BOT_DB_PATH` (Python)
- `DATA_DIR` (Node config) untuk path DB/runtime folder.
