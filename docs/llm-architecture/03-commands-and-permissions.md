# 03 - Commands, Aliases, Permissions

## Canonical command list (Node)
- `help`
- `prompt`
- `reset`
- `permission`
- `mode`
- `trigger`
- `dashboard`
- `broadcast`
- `info`
- `debug`
- `join`
- `sticker`
- `model`
- `modelcfg`
- `setting`

## Alias singular/plural
Parser menormalisasi command ke bentuk canonical.

Contoh alias:
- `/setting`, `/settings` -> `setting`
- `/model`, `/models` -> `model`
- `/prompt`, `/prompts` -> `prompt`
- `/dashboard`, `/dashboards` -> `dashboard`
- dst (semua command utama punya pasangan singular/plural).

## Permission model (umum)
- Private chat: sebagian besar command diizinkan.
- Group chat: butuh admin/owner untuk command konfigurasi.

### Moderation level (`/permission`)
- `0`: moderation forbidden
- `1`: delete allowed
- `2`: delete + mute allowed
- `3`: delete + mute + kick allowed

Catatan:
- Untuk level > 0, bot harus punya role admin di grup.

## Command behavior ringkas
- `/prompt [text|clear]` – set/lihat/hapus prompt override chat.
- `/reset` – clear memory/history chat di Python.
- `/mode [auto|prefix|hybrid]` – mode trigger respon.
- `/trigger [...]` – set trigger prefix mode.
- `/model` – pilih model LLM2 per chat.
- `/modelcfg ...` – CRUD daftar model (owner only).
- `/setting` – menu interaktif untuk mode/model/permission/misc.
- `/dashboard` – tampilkan statistik pemakaian.
- `/broadcast` – kirim broadcast (owner only).
- `/info` – info user/chat/grup.
- `/debug` – kirim payload test interactive.
- `/join <invite link>` – join grup via invite.
- `/sticker` – buat sticker dari media.
