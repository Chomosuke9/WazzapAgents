# 04 - Protocol and Actions

## Node -> Python events

### `incoming_message`
Event utama berisi payload chat/message ter-normalisasi.

Field penting:
- `chatId`, `chatType`, `chatName`
- `senderId`, `senderRef`, `senderName`
- `contextMsgId`, `messageId`, `timestampMs`
- `text`, `quoted`, `attachments`, `mentionedJids`
- `botMentioned`, `repliedToBot`
- `slashCommand`, `commandHandled`

### Control events
- `clear_history` – minta Python reset history per chat.
- `set_llm2_model` – set model chat (authoritative sync).
- `invalidate_llm2_model` – clear cache model chat.
- `invalidate_default_model` – clear cache default model.
- `whatsapp_status` – status open/closed dari socket WA.

## Python -> Node actions
- `send_message`
- `react_message`
- `delete_message`
- `kick_member`
- `mark_read`
- `send_presence`
- `send_buttons`
- `send_carousel`

## Ack/Error balik ke Python
- `action_ack`
- `send_ack` (legacy compatibility)
- `error`

## Reliability contract
- Event kontrol penting dari Node ke Python sebaiknya pakai `sendReliable`.
- Jika WS belum open, event disimpan di queue memory.
- Queue akan diflush saat koneksi kembali open.

## Referensi
- Kontrak payload lengkap ada di root `README.md`.
