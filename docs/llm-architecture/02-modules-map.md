# 02 - Modules Map

## Node side (`src/`)

### Core
- `src/index.js` – bootstrap app + dispatcher action dari Python.
- `src/wsClient.js` – WS client ke Python (`send`, `sendReliable`, queue).
- `src/config.js` – env parsing + path runtime (data/auth/media).
- `src/db.js` – SQLite via sql.js (settings/models/stats read).

### WhatsApp integration (`src/wa/`)
- `connection.js` – socket init, lifecycle, button response handling.
- `inbound.js` – normalisasi incoming WA -> payload `incoming_message`.
- `outbound.js` – kirim teks/media ke WhatsApp.
- `actions.js` – react/delete wrappers.
- `moderation.js` – kick members.
- `presence.js` – mark read + typing presence.
- `commands.js` – parse slash command + helpers (`broadcast/info/debug/join`).
- `commandHandler.js` – handler command utama (`prompt`, `mode`, `setting`, `model`, dll).
- `events.js` – synthetic context events (action log, group join, role change).

### Interactive (`src/wa/interactive/`)
- `sendInteractive.js` – low-level helper native flow.
- `sendButtons.js` – tombol/list native flow.
- `sendCarousel.js` – carousel card.

## Python side (`python/bridge/`)

### Core
- `main.py` – WS handler, batching, loop utama processing.
- `db.py` – SQLite access + caches (settings/models/stats/mutes).
- `dashboard.py` – stats buffer + flush + format dashboard.
- `commands.py` – parser/handler slash command (legacy path di Python).
- `history.py` – dataclass history message.

### LLM pipeline (`python/bridge/llm/`)
- `llm1.py` – routing/decision should reply or not.
- `llm2.py` – reply generation + tools invocation context.
- `schemas.py` – tool schema.
- `prompt.py` – prompt assembly helpers.
- `metadata.py` – metadata injection.

### Messaging pipeline (`python/bridge/messaging/`)
- `processing.py` – burst processing helpers.
- `filtering.py` – prefix/trigger/mode filtering.
- `actions.py` – parse action lines dari model output.
- `gateway.py` – kirim action WS ke Node.
- `moderation.py` – moderation checks.
