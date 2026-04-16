# Plan: Fitur Ganti Model Per Chat dengan Interactive Message

## Ringkasan
Implementasi fitur untuk mengganti model LLM2 per chat menggunakan interactive message (list). Termasuk command `/model` untuk user dan `/modelcfg` untuk developer.

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

### Tabel Baru: `llm_models`
```sql
CREATE TABLE IF NOT EXISTS llm_models (
  model_id    TEXT PRIMARY KEY,  -- e.g., "gpt-4o-mini"
  display_name TEXT NOT NULL,   -- e.g., "GPT-4 Mini"
  description TEXT,              -- e.g., "Fast & cheap for simple tasks"
  is_active    INTEGER NOT NULL DEFAULT 1,
  sort_order   INTEGER NOT NULL DEFAULT 0
);
```

### Kolom Baru di `chat_settings`
```sql
ALTER TABLE chat_settings ADD COLUMN llm2_model TEXT DEFAULT NULL;
```

### Fungsi DB Baru (python/bridge/db.py)
- `get_llm2_model(chat_id)` → return model_id atau default
- `set_llm2_model(chat_id, model_id)` → set model untuk chat
- `get_all_active_models()` → list semua model aktif dari tabel
- `add_model(model_id, display_name, description, sort_order)` → owner only
- `update_model(model_id, display_name, description, is_active, sort_order)` → owner only
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
- Di group: Admin/Founder/Owner bot
- Di private chat: Owner bot saja (atau semua user jika diizinkan)

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
- `/modelcfg sync` - Sync dari env `LLM2_MODEL_x` ke database

**Flow Interactive:**
1. `/modelcfg` tanpa argumen → kirim interactive message dengan opsi
2. Setiap aksi membuka submenu atau konfirmasi

---

## 4. `/settings` - Dashboard Pengaturan Chat

**Flow:**
1. User ketik `/settings`
2. Bot kirim interactive message dengan info:

```
┌─────────────────────────────────┐
│ ⚙️ Pengaturan Chat               │
├─────────────────────────────────┤
│ Model: GPT-4 Mini               │
│ Permission: Level 2             │
│   (delete & mute allowed)       │
│ Prompt: [custom prompt atau     │
│   "(default)"]                  │
├─────────────────────────────────┤
│ Tombol:                         │
│  [📋 Edit Prompt]                │
│  [🤖 Ganti Model]               │
│  [🔐 Atur Permission]           │
└─────────────────────────────────┘
```

**Interactive Buttons:**
1. **📋 Edit Prompt** → Trigger `/prompt` command (existing)
2. **🤖 Ganti Model** → Kirim interactive list model (sama seperti `/model`)
3. **🔐 Atur Permission** → Kirim interactive list permission levels

---

## 5. Backend Flow (Python Bridge)

### config.py
```python
def parse_llm2_models_from_env() -> list[dict]:
    """Parse LLM2_MODEL_x from env, return [{model_id, sort_order}, ...]"""
    models = []
    for key, value in os.environ.items():
        match = re.match(r'LLM2_MODEL_(\d+)', key)
        if match:
            num = int(match.group(1))
            models.append({'model_id': value, 'sort_order': num})
    models.sort(key=lambda x: x['sort_order'])
    return models
```

### llm2.py
```python
def get_llm2_model_for_chat(chat_id: str) -> str:
    """Get model_id for chat, or default if not set"""
    model_id = db.get_llm2_model(chat_id)
    if model_id:
        return model_id
    # Return default (smallest sort_order)
    models = db.get_all_active_models()
    return models[0]['model_id'] if models else 'gpt-4.1'

# Modify generate_reply() to use get_llm2_model_for_chat()
```

---

## 6. Node.js Gateway (commands.js)

### Tambah ke SLASH_CMD_RE
```javascript
const SLASH_CMD_RE = /^\/(broadcast|prompt|reset|permission|info|mode|trigger|dashboard|help|debug|join|sticker|model|modelcfg|settings)\b\s*([\s\S]*)/i;
```

### Handler Functions
- `handleModelCommand()` - untuk user
- `handleModelcfgCommand()` - untuk developer/owner
- `handleSettingsCommand()` - dashboard pengaturan

### Interactive Message Response
Hook ke `inbound.js` untuk handle button click dari interactive message:
- Row ID format: `model_select:<model_id>`
- Row ID format: `permission_select:<level>`
- Row ID format: `settings_action:<action>`

---

## 7. File yang Perlu Diubah

### Python
1. `python/bridge/config.py` - Parse `LLM2_MODEL_x` dari env
2. `python/bridge/db.py` - Schema baru + fungsi baru
3. `python/bridge/llm/llm2.py` - Gunakan model per chat

### Node.js
1. `src/wa/commands.js` - Commands `/model`, `/modelcfg`, `/settings`
2. `src/wa/inbound.js` - Handle interactive button responses
3. `.env.example` - Dokumentasi format baru

### Database
1. Migration: Buat tabel `llm_models` + kolom `chat_settings.llm2_model`
2. Seed: Sync dari env saat pertama kali run

---

## 8. Urutan Implementasi

1. **Phase 1: Database & Config**
   - Tambah schema ke db.py
   - Parse env di config.py
   - Fungsi get/set model

2. **Phase 2: Backend Integration**
   - Modify llm2.py untuk gunakan model per chat
   - Test dengan hardcoded model dulu

3. **Phase 3: Node.js Commands**
   - Implement `/model` dengan interactive list
   - Implement `/settings` dashboard
   - Handle button responses

4. **Phase 4: Developer Commands**
   - Implement `/modelcfg` untuk manage model list
   - Sync dari env ke database

5. **Phase 5: Testing & Polish**
   - Test semua command
   - Test permission (admin/owner)
   - Test edge cases (model tidak ditemukan, dll)

---

## 9. Contoh Penggunaan

### Developer Setup
```bash
# .env
LLM2_MODEL_1=gpt-4o-mini
LLM2_MODEL_2=gpt-4o
LLM2_MODEL_3=claude-3-5-sonnet
```

```bash
# Sync ke database (owner only)
/modelcfg sync
```

```bash
# Tambah model custom
/modelcfg add gemini-2.0-flash "Gemini 2.0 Flash" "Google's fast model"
```

### User
```bash
# Lihat & pilih model
/model
# atau
/settings → klik "Ganti Model"
```

### Admin Group
```bash
# Ganti model untuk group
/model
# Pilih "Claude 3.5 Sonnet"
# Bot: "Model diubah ke: Claude 3.5 Sonnet"
```

---

## 10. Edge Cases

1. **Model dihapus dari db tapi masih dipakai chat** → fallback ke default
2. **Env diubah tapi db tidak di-sync** → tetap pakai data dari db
3. **User non-admin coba `/model` di group** → reject dengan pesan
4. **Model tidak ditemukan di provider** → error log, fallback ke default
5. **Chat tidak punya model set** → gunakan default (angka terkecil)

---

## 11. Notes

- `/modelcfg sync` akan meng-overwrite data di db dengan env, tapi preserve `is_active` dan `sort_order` jika sudah ada
- Kolom `sort_order` di db bisa diubah manual via `/modelcfg edit`
- Interactive message menggunakan `sendList()` dari `sendInteractive.js`
- Format rowId harus mudah diparse, contoh: `model_select:gpt-4o-mini`
