# 00 - System Overview

## Komponen inti

### 1) Node Gateway (`src/`)
Fungsi utama:
- Koneksi WhatsApp (Baileys v7).
- Parsing incoming WA messages + media normalization.
- Menangani slash command dan tombol interactive.
- Mengirim event `incoming_message` ke Python bridge via WebSocket.
- Mengeksekusi action dari Python (`send_message`, `delete_message`, `kick_member`, dll).

### 2) Python Bridge (`python/bridge/`)
Fungsi utama:
- Menerima `incoming_message` dari Node.
- Batching/debounce per chat, filter trigger, dan routing LLM1.
- Generate reply/tool calls via LLM2.
- Mengirim action balik ke Node via WebSocket.
- Menulis statistik dashboard (buffer + flush periodik).

## Arah data (high level)
1. User kirim pesan di WhatsApp.
2. Node parse pesan lalu kirim `incoming_message` ke Python.
3. Python memutuskan respon (LLM1/LLM2) dan aksi moderation.
4. Python kirim action ke Node.
5. Node eksekusi action ke WhatsApp, lalu kirim ack/error ke Python.

## Desain command
- Slash command di-parse di Node (`src/wa/commands.js`) dengan alias singular/plural.
- Alias di-normalisasi ke command canonical, contoh:
  - `/model` dan `/models` -> `model`
  - `/setting` dan `/settings` -> `setting`
- Handler command utama ada di `src/wa/commandHandler.js`.

## Interaktif UI
- Menu `/setting` dan `/model` menggunakan native flow button/list.
- Button click menghasilkan `selectedId` seperti `model_select:<id>`.

## Reliability penting
- `wsClient.send()` = best effort (drop jika WS belum open).
- `wsClient.sendReliable()` = queued, lalu di-flush saat reconnect.
- Event kontrol penting (clear history / invalidate / set model / status) harus pakai jalur reliable.
