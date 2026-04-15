# Rencana: Split Command dan Chatbot Listener

## Ringkasan
Memisahkan processing command dan chatbot menjadi dua listener terpisah untuk menghindari blocking.

## Arsitektur Saat Ini

```
WhatsApp (Baileys)
    │
    ▼
sock.ev.on('messages.upsert', single_listener)
    │
    ▼
handleIncomingMessage()
    ├── /broadcast, /info, /debug, /join → handle di Node.js, RETURN (tidak kirim ke Python)
    └── pesan lainnya → wsClient.send() ke Python
    │
    ▼
Python Bridge (main.py)
    ├── process_message_batch()
    │   ├── /reset, /sticker, /prompt, /permission, /mode, /trigger, /dashboard, /help
    │   │   └── handle command, skip LLM
    │   └── pesan lainnya → LLM1 → LLM2 → response
```

**Masalah:** Command yang di-handle Python bisa blocking jika LLM sedang berjalan.

## Arsitektur Baru

```
WhatsApp (Baileys)
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
sock.ev.on('messages.upsert', listener_cmd)   sock.ev.on('messages.upsert', listener_chatbot)
    │                                          │
    ▼                                          ▼
handleCommandListener()                      handleIncomingMessage()
    │                                          │
    ├── SEMUA /command di-handle di sini      └── SELALU kirim ke Python
    ├── Akses DB via better-sqlite3            └── Python log command ke history, skip LLM
    ├── Response instant, non-blocking        └── Python proses LLM untuk non-command
    └── (tidak perlu kirim ke Python)
```

**Keuntungan:**
1. Command eksekusi instant, tidak perlu tunggu LLM
2. LLM processing tidak blocking command
3. Python fokus hanya ke LLM, tidak ada command logic

---

## Task List

### 1. Setup Database di Node.js

**File baru:** `src/db.js`

Buat wrapper SQLite menggunakan `better-sqlite3` dengan fungsi:
- `getPrompt(chatId)` / `setPrompt(chatId, prompt)`
- `getPermission(chatId)` / `setPermission(chatId, level)`
- `getMode(chatId)` / `setMode(chatId, mode)`
- `getTriggers(chatId)` / `setTriggers(chatId, triggers)`
- `getStats(chatId, periodType, periodKey)`
- `getTopUsers(chatId, periodType, periodKey, limit)`
- `clearSettings(chatId)` (untuk /reset)

**Dependencies:**
- `pnpm add better-sqlite3`

**Note:** Database path sama dengan Python: `data/bot.db`. Schema sudah ada, tinggal read/write dari Node.js.

---

### 2. Buat Command Handler di Node.js

**File baru:** `src/wa/commandHandler.js`

Fungsi utama: `handleCommand(msg, context)`

Handle semua command:
- `/help` → kirim help text
- `/info` → kirim info user/group (sudah ada)
- `/debug` → kirim debug messages (sudah ada)
- `/broadcast` → broadcast ke semua grup (sudah ada)
- `/join` → join group via invite link (sudah ada)
- `/prompt [text]` → set/get custom prompt (perlu DB)
- `/reset` → clear chat memory (perlu DB + clear history cache)
- `/permission [0-3]` → set/get permission level (perlu DB)
- `/mode [auto|prefix|hybrid]` → set/get mode (perlu DB)
- `/trigger [tag|reply|name|join|all|none]` → toggle triggers (perlu DB)
- `/dashboard` → show usage stats (perlu DB)
- `/sticker [upper#lower]` → create sticker from image (perlu PIL/canvas di Node.js)

**Untuk /sticker:** Perlu tambahkan dependency:
- `pnpm add sharp` (untuk image manipulation di Node.js)
- Atau biarkan /sticker tetap di Python (karena butuh PIL)

---

### 3. Buat Listener Baru di connection.js

**File:** `src/wa/connection.js`

Ubah dari:
```javascript
sock.ev.on('messages.upsert', async ({ messages, type }) => {
  // ... handle semua
});
```

Menjadi:
```javascript
// Listener 1: Command (non-blocking)
sock.ev.on('messages.upsert', async ({ messages, type }) => {
  if (type !== 'notify') return;
  for (const msg of messages) {
    try {
      await handleCommandListener(msg);
    } catch (err) {
      logger.error({ err }, 'command listener error');
    }
  }
});

// Listener 2: Chatbot (send to Python)
sock.ev.on('messages.upsert', async ({ messages, type }) => {
  // ... existing logic, tapi HAPUS command handling
  // ... tetap kirim SEMUA pesan ke Python (termasuk /command)
});
```

---

### 4. Modifikasi handleIncomingMessage di inbound.js

**File:** `src/wa/inbound.js`

Hapus handling untuk:
- `/broadcast` → pindah ke commandHandler.js
- `/info` → pindah ke commandHandler.js
- `/debug` → pindah ke commandHandler.js
- `/join` → pindah ke commandHandler.js

**PENTING:** Jangan `return` setelah detect slash command. Tetap lanjut kirim ke Python.

Payload tetap include `slashCommand` field untuk context di Python.

---

### 5. Modifikasi Python Bridge

**File:** `python/bridge/main.py`

Di `process_message_batch()`:

Saat detect `slashCommand`:
1. Tambah ke history (tetap, untuk context LLM)
2. **SKIP** handle command (karena sudah di-handle di Node.js)
3. **SKIP** LLM1/LLM2
4. Return early (jangan kirim response, karena Node.js sudah handle)

**Perlu:** Cara untuk signal ke Python bahwa command sudah di-handle.

Opsi:
- Tambah field `commandHandled: true` di payload dari Node.js
- Python check field ini, jika true → skip command handling, skip LLM

---

### 6. Hapus Command Logic dari Python

**File:** `python/bridge/commands.py`

Bisa di-simplify atau hapus, karena semua command sekarang di Node.js.

**File:** `python/bridge/main.py`

Hapus import dan logic untuk `/reset`, `/sticker`, `/prompt`, dll.

---

### 7. Chat History Management

**Issue:** /reset perlu clear chat history yang ada di Python memory.

**Solusi:**
- Opsi A: Node.js kirim WebSocket message ke Python dengan type `clear_history`
- Opsi B: Python check DB saat process batch, jika ada flag `reset_requested` → clear history
- Opsi C: Biarkan Python handle /reset saja (tapi via WebSocket message dari Node.js)

**Rekomendasi:** Opsi A - Node.js kirim `{ type: 'clear_history', chatId }` setelah /reset command.

---

### 8. Dashboard Stats

**Issue:** Dashboard stats di-track di Python, tapi /dashboard sekarang di Node.js.

**Solusi:**
- Opsi A: Node.js baca langsung dari DB (table `chat_stats`, `chat_user_stats`)
- Opsi B: Python kirim stats via WebSocket saat diminta

**Rekomendasi:** Opsi A - Node.js baca langsung dari DB.

---

## File yang Perlu Diubah/Dibuat

### File Baru
1. `src/db.js` - Database wrapper (better-sqlite3)
2. `src/wa/commandHandler.js` - Semua command handlers
3. `src/history.js` - Chat history cache (optional, untuk /reset)

### File Diubah
1. `src/wa/connection.js` - Tambah listener kedua
2. `src/wa/inbound.js` - Hapus command handling, tetap kirim ke Python
3. `src/wa/commands.js` - Pindahkan logic ke commandHandler.js atau biarkan sebagai helper
4. `python/bridge/main.py` - Skip command handling, handle clear_history
5. `package.json` - Tambah dependencies: `better-sqlite3`, `sharp` (untuk sticker)

### File Dihapus/Simplify
1. `python/bridge/commands.py` - Tidak diperlukan lagi (atau keep untuk reference)

---

## Dependencies Baru

```bash
pnpm add better-sqlite3
pnpm add sharp  # untuk /sticker (optional)
```

---

## Flow Detail

### Command Flow (Node.js)

```
User: /help
    │
    ▼
listener_cmd → parseSlashCommand(text) → { command: 'help', args: '' }
    │
    ▼
handleCommand('help', msg, context)
    │
    ▼
sock.sendMessage(chatId, helpText)
    │
    ▼
return (selesai, non-blocking)
```

### Chatbot Flow (Node.js → Python)

```
User: "Halo bot!"
    │
    ▼
listener_chatbot → handleIncomingMessage()
    │
    ▼
wsClient.send({ type: 'incoming_message', payload })
    │
    ▼
Python: process_message_batch()
    │
    ▼
LLM1 → LLM2 → send response
```

### Command + Context Flow

```
User: /prompt jangan gunakan bahasa gaul
    │
    ▼
listener_cmd → handleCommand('prompt', msg, context)
    │   ├── setPrompt(chatId, "jangan gunakan bahasa gaul")
    │   └── sock.sendMessage(chatId, "Prompt updated")
    │
    ▼
listener_chatbot → handleIncomingMessage()
    │   └── wsClient.send({ type: 'incoming_message', payload: { slashCommand: {...} } })
    │
    ▼
Python: process_message_batch()
    │   ├── payload.slashCommand detected
    │   ├── commandHandled: true → skip command handling
    │   ├── append to history (for context)
    │   └── skip LLM, return
```

---

## Special Cases

### /reset

```
listener_cmd:
  1. handleCommand('reset')
  2. clearSettings(chatId) via DB
  3. wsClient.send({ type: 'clear_history', chatId })
  4. sock.sendMessage(chatId, "Memory cleared")

listener_chatbot:
  5. send payload to Python

Python:
  6. receive clear_history → per_chat[chatId].clear()
  7. receive payload with slashCommand → skip
```

### /sticker

**Opsi A:** Implement di Node.js dengan `sharp`
```javascript
import sharp from 'sharp';

async function createSticker(imagePath, upperText, lowerText) {
  // Use sharp to add text overlay
  // Return sticker path
}
```

**Opsi B:** Biarkan di Python, tapi trigger dari Node.js via WebSocket
```javascript
wsClient.send({
  type: 'create_sticker',
  chatId,
  imagePath,
  upperText,
  lowerText,
  replyTo
});
```

**Rekomendasi:** Opsi A untuk consistency.

### /dashboard

```javascript
import { getStats, getTopUsers } from '../db.js';

async function handleDashboard(chatId) {
  const daily = getStats(chatId, 'daily', today);
  const weekly = getStats(chatId, 'weekly', weekKey);
  const monthly = getStats(chatId, 'monthly', monthKey);
  const topUsers = getTopUsers(chatId, 'monthly', monthKey, 5);
  // Format and send
}
```

---

## Testing Plan

1. Test setiap command individual di Node.js
2. Test bahwa command tidak blocking chatbot
3. Test bahwa Python tetap menerima command untuk context
4. Test /reset clears history di Python
5. Test /dashboard shows correct stats

---

## Rollback Plan

Jika ada masalah, rollback dengan:
1. Kembalikan single listener di connection.js
2. Hapus file baru (commandHandler.js, db.js)
3. Restore Python command handling

---

## Timeline Estimasi

- Task 1 (DB wrapper): 1-2 jam
- Task 2 (commandHandler): 2-3 jam
- Task 3-4 (listeners): 1 jam
- Task 5-6 (Python modifikasi): 1-2 jam
- Testing: 1-2 jam

**Total:** 6-10 jam

---

## Keputusan yang Sudah Ditetapkan

1. **Race condition:** Kedua listener berjalan paralel, tidak masalah. Jika pesan dimulai dengan `/`, tidak trigger LLM.

2. **Database ownership (write access):**
   - **Node.js writes:** `chat_settings` (prompt, permission, mode, triggers), `chat_mutes`
   - **Python writes:** `chat_stats`, `chat_user_stats` (untuk dashboard tracking)
   - **Python reads only:** `chat_settings`, `chat_mutes`
   - **Node.js reads only:** `chat_stats`, `chat_user_stats`
   - Prinsip: Satu tabel hanya boleh ada satu writer untuk menghindari conflict.

3. **/sticker:** Implement di Node.js dengan `sharp`.

4. **Mute management:** Di-handle oleh Node.js. Python return instruction mute via response, Node.js yang execute dan write ke DB.

5. **History sync:** Node.js kirim `{ type: 'clear_history', chatId }` ke Python setelah /reset. Tidak perlu local cache di Node.js.

6. **Error handling:** Tidak diperlukan fallback khusus karena Node.js adalah single writer untuk settings.

7. **Task execution:** Semua task bisa dikerjakan secara paralel (Task 1-2 bersamaan dengan Task 5-6).
