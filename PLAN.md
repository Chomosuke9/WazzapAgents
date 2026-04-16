# Plan: Fitur Ganti Model Per Chat dengan Interactive Message

## Status: ✅ COMPLETED (Database migration to sql.js in progress)

---

## Ringkasan
Implementasi fitur untuk mengganti model LLM2 per chat menggunakan interactive message (list). Termasuk command `/model` untuk user dan `/modelcfg` untuk developer. Semua command sudah dimigrasikan ke JavaScript (Node.js).

---

## 1. Environment Variables (LLM2 Multi-Model)

### Format Baru
```
LLM2_MODEL_1=gpt-4o-mini
LLM2_MODEL_2=gpt-4o
LLM2_MODEL_3=claude-3-5-sonnet
```
- Nomor terkecil = default model
- Bisa sebanyak apapun
- Hanya untuk LLM2 (LLM1 tidak perlu)

### Backward Compatibility
- Jika tidak ada `LLM2_MODEL_x`, fallback ke `LLM2_MODEL` yang sudah ada

---

## 2. Database Schema (SQLite)

### Tabel: `llm_models`
```sql
CREATE TABLE IF NOT EXISTS llm_models (
  model_id    TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  description TEXT,
  is_active    INTEGER NOT NULL DEFAULT 1,
  sort_order   INTEGER NOT NULL DEFAULT 0
);
```

### Kolom di `chat_settings`
```sql
ALTER TABLE chat_settings ADD COLUMN llm2_model TEXT DEFAULT NULL;
```

### Fungsi DB (src/db.js + python/bridge/db.py)
- `get_llm2_model(chat_id)` → return model_id atau default
- `set_llm2_model(chat_id, model_id)` → set model untuk chat
- `get_all_active_models()` → list semua model aktif
- `get_all_models()` → list semua model (aktif + nonaktif)
- `get_default_llm2_model()` → model dengan sort_order terkecil
- `add_model(model_id, display_name, description, sort_order)` → owner only
- `update_model(model_id, ...)` → owner only
- `delete_model(model_id)` → owner only

---

## 3. Commands

### 3.1 `/model` - Untuk User (Admin/Owner)

**Flow:**
1. User ketik `/model`
2. Bot kirim interactive list dengan model yang tersedia
3. Model aktif ditandai dengan emoji (contoh: `GPT-4 Mini ✓`)
4. User pilih model
5. Bot kirim konfirmasi: `Model diubah ke: GPT-4 Mini`

**Permission:**
- Di group: Admin/Owner bot
- Di private chat: Owner bot saja

**UI Interactive List:**
```
┌─────────────────────────────┐
│ Pilih Model LLM             │
├─────────────────────────────┤
│ Model saat ini: GPT-4 Mini ✓ │
├─────────────────────────────┤
│ Models:                     │
│  ○ GPT-4 Mini ✓             │
│    Fast & cheap             │
│  ○ GPT-4o                   │
│    Balanced performance     │
│  ○ Claude 3.5 Sonnet        │
│    Best for complex tasks   │
└─────────────────────────────┘
```

### 3.2 `/modelcfg` - Untuk Developer (Owner Only)

**Subcommands:**
- `/modelcfg list` - Tampilkan semua model dengan detail
- `/modelcfg add <model_id> <display_name>` - Tambah model baru
- `/modelcfg edit <model_id> [name=<name>] [desc=<desc>]` - Edit model
- `/modelcfg remove <model_id>` - Hapus model
- `/modelcfg setdefault <model_id>` - Set sebagai default

**Flow Interactive:**
1. `/modelcfg` tanpa argumen → kirim interactive message dengan opsi
2. Setiap aksi membuka submenu atau konfirmasi

---

## 4. `/settings` - Dashboard Pengaturan Chat

**Flow:**
1. User ketik `/settings`
2. Bot kirim interactive message dengan info

**Interactive Buttons:**
1. **🤖 Ganti Model** → Kirim interactive list model (sama seperti `/model`)
2. **📝 Edit Prompt** → Info untuk gunakan `/prompt`
3. **🔐 Atur Permission** → Info untuk gunakan `/permission`

---

## 5. Command List (All in JavaScript)

| Command | Handler | Status |
|---------|---------|--------|
| `/prompt` | JavaScript | ✅ |
| `/reset` | JavaScript (sends clear_history ke Python) | ✅ |
| `/permission` | JavaScript | ✅ |
| `/mode` | JavaScript | ✅ |
| `/trigger` | JavaScript | ✅ |
| `/dashboard` | JavaScript | ✅ |
| `/help` | JavaScript | ✅ |
| `/broadcast` | JavaScript | ✅ |
| `/info` | JavaScript | ✅ |
| `/debug` | JavaScript | ✅ |
| `/join` | JavaScript | ✅ |
| `/sticker` | JavaScript (sharp untuk image processing) | ✅ |
| `/model` | JavaScript | ✅ |
| `/modelcfg` | JavaScript | ✅ |
| `/settings` | JavaScript | ✅ |

**Python Bridge** sekarang hanya handle LLM processing saja.

---

## 6. Button Response Handling

Button clicks dari interactive messages dihandle di `messages.upsert` listener:

**Row ID Formats:**
- `model_select:<model_id>` - Pilih model
- `settings:model` - Menu settings → model
- `settings:prompt` - Menu settings → prompt
- `settings:permission` - Menu settings → permission
- `modelcfg:list` - modelcfg menu → list
- `modelcfg_remove:<model_id>` - Hapus model
- `modelcfg_confirm_remove:<model_id>` - Konfirmasi hapus
- `modelcfg_cancel_remove` - Batal hapus

---

## 7. Backend Flow

### Python Bridge (LLM Only)
- `python/bridge/db.py` - Schema + model functions
- `python/bridge/llm/llm2.py` - Uses per-chat model via `get_llm2_model_for_chat()`
- `python/bridge/main.py` - Command handling disabled (all in JS)

### Node.js Gateway (Commands)
- `src/db.js` - Node.js SQLite (currently broken - migrating to sql.js)
- `src/wa/commandHandler.js` - All command handlers
- `src/wa/connection.js` - Button response handler + event listeners
- `src/wa/stickerTool.js` - Sticker creation with sharp

---

## 8. File Changes Summary

### Python (DONE)
- `python/bridge/db.py` - Schema + model functions
- `python/bridge/llm/llm2.py` - Per-chat model integration
- `python/bridge/main.py` - Commands skipped (handled by JS)

### Node.js (DONE)
- `src/db.js` - Schema + model functions
- `src/wa/commands.js` - SLASH_CMD_RE regex
- `src/wa/commandHandler.js` - All command handlers
- `src/wa/connection.js` - Button response + listeners
- `src/wa/stickerTool.js` - Sticker creation

### Configuration (DONE)
- `.env.example` - No changes needed (sync via /modelcfg)

---

## 9. Pending Issues

### Critical: Database Migration
- `better-sqlite3` native binding fails on server
- **Solution:** Migrate to `sql.js` (pure JavaScript, no native compilation)
- Status: In progress

### Completed Fixes
- ✅ `senderIsOwner` not defined - Fixed
- ✅ Button click not handled - Fixed (moved to messages.upsert)
- ✅ `/sticker` migration - Completed

---

## 10. Testing Checklist

- [ ] `/prompt` - set/show/clear prompt
- [ ] `/reset` - clear chat memory
- [ ] `/permission` - set permission level
- [ ] `/mode` - change mode (auto/prefix/hybrid)
- [ ] `/trigger` - toggle triggers
- [ ] `/dashboard` - view stats
- [ ] `/model` - select model
- [ ] `/modelcfg add` - add new model
- [ ] `/modelcfg remove` - delete model
- [ ] `/settings` - view settings menu
- [ ] `/sticker` - create sticker from image
- [ ] Button clicks in interactive messages

---

## 11. Notes

- Interactive message menggunakan `sendNativeFlow` dengan `single_select`
- Row ID format: `prefix:suffix` untuk easy parsing
- `model_id` untuk backend, `display_name` dan `description` untuk frontend
- Default model = model dengan `sort_order` terkecil
