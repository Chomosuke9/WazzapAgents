---
sidebar_position: 1
---

# Arsitektur

WazzapAgents terdiri dari dua komponen runtime yang berkomunikasi melalui WebSocket:

```
WhatsApp <──Baileys──> Node.js Gateway <──WebSocket──> Python LLM Bridge <──HTTP──> LLM API
```

## Komponen Utama

### 1. Node.js Gateway (`src/`)

Gateway bertanggung jawab untuk:

- **Koneksi WhatsApp** — Menggunakan Baileys v7 untuk connect ke WhatsApp via multi-device protocol.
- **Parsing pesan** — Mengekstrak teks, media, mentions, quoted messages, lokasi, dan vCard dari pesan Baileys mentah.
- **Forwarding ke bridge** — Mengirim payload `incoming_message` ke Python bridge via WebSocket.
- **Eksekusi aksi** — Menerima command dari bridge (send, react, delete, kick, mark read, typing) dan mengeksekusinya di WhatsApp.
- **Caching** — Menyimpan message cache, metadata grup (TTL 60 detik), nama partisipan, dan sender ref registry di memori.

### 2. Python LLM Bridge (`python/bridge/`)

Bridge bertanggung jawab untuk:

- **WebSocket server** — Menerima pesan dari gateway dan mengirim command balik.
- **Message batching** — Mengelompokkan pesan yang masuk dalam burst window dengan debounce logic.
- **Pipeline LLM dua tahap:**
  - **LLM1 (Gating)** — Memutuskan apakah bot harus merespons pesan. Ringan dan cepat.
  - **LLM2 (Responder)** — Menghasilkan respons lengkap dengan konteks percakapan dan system prompt.
- **Slash commands** — Menangani `/prompt`, `/reset`, `/permission` secara langsung.
- **Penyimpanan** — SQLite untuk pengaturan per-chat (prompt kustom, level permission).
- **History management** — Menyimpan riwayat percakapan per-chat di memori dengan limit yang dapat dikonfigurasi.

## Alur Data

### Pesan Masuk (User -> Bot)

```
1. User mengirim pesan di WhatsApp
2. Baileys menerima event `messages.upsert`
3. Gateway parsing pesan (messageParser.js)
4. Gateway assign contextMsgId & senderRef (identifiers.js)
5. Gateway kirim `incoming_message` ke bridge via WebSocket
6. Bridge batch pesan (debounce 5 detik, max burst 20 detik)
7. Bridge jalankan LLM1 (gating decision)
8. Jika LLM1 memutuskan respond → jalankan LLM2
9. LLM2 generate respons
10. Bridge parse aksi dari respons (REPLY_TO, DELETE, KICK, REACT_TO)
11. Bridge kirim command ke gateway via WebSocket
12. Gateway eksekusi aksi di WhatsApp
```

### Pesan Konteks (Bot -> Bridge)

Pesan yang dikirim oleh bot sendiri juga diteruskan ke bridge sebagai `contextOnly: true` dan `triggerLlm1: false`. Ini memperkaya konteks percakapan tanpa menyebabkan loop.

## Identifikasi Pesan

### contextMsgId

Counter 6 digit per-chat (`000000`–`999999`, wrap setelah `999999`). Digunakan untuk referensi pesan dalam percakapan — misalnya saat bot perlu reply ke pesan tertentu atau menghapus pesan.

### senderRef

ID pendek deterministik per-pengirim per-chat, di-generate dari SHA-1 hash `chatId|senderId`. Digunakan di semua interaksi LLM — **tidak pernah** mengekspos JID asli ke LLM.

## Penyimpanan Data

| Data | Lokasi | Tipe |
|------|--------|------|
| Session WhatsApp | `data/auth/` | File (Baileys auth state) |
| Media yang diunduh | `data/media/` | File (gambar, video, dll.) |
| Pengaturan chat | `data/bot.db` | SQLite |
| Riwayat percakapan | Memori (RAM) | In-memory deque |
| Message cache | Memori (RAM) | In-memory Map |
| Metadata grup | Memori (RAM) | TTL cache (60 detik) |

## Diagram Modul

### Node.js Gateway

```
index.js          ← Bootstrap, routing command dari WS ke aksi WhatsApp
├── waClient.js   ← Koneksi WhatsApp, send/receive/moderasi
├── wsClient.js   ← WebSocket client ke bridge (auto-reconnect)
├── config.js     ← Environment variable loading
├── logger.js     ← Pino structured logging
├── messageParser.js  ← Parsing pesan Baileys → payload terstruktur
├── mediaHandler.js   ← Download & validasi media, MIME inference
├── identifiers.js    ← contextMsgId counter, senderRef registry
├── participants.js   ← Mapping role partisipan, cache nama
├── groupContext.js   ← Cache metadata grup
├── caches.js         ← In-memory caches (message, metadata, nama)
└── utils.js          ← Utilitas stream (streamToBuffer, streamToFile)
```

### Python Bridge

```
main.py       ← WebSocket handler, batching, orkestrasi pipeline
├── llm1.py   ← LLM1 gating/decision (LangChain + OpenAI SDK)
├── llm2.py   ← LLM2 response generation
├── commands.py ← Parser slash command (/prompt, /reset, /permission)
├── config.py   ← Parsing env variable, konstanta konfigurasi
├── db.py       ← SQLite storage dengan in-memory cache
├── history.py  ← WhatsAppMessage dataclass, formatting history
├── media.py    ← Pemrosesan attachment visual untuk multimodal
└── log.py      ← Structured logging dengan contextvars
```
