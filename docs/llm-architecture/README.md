# LLM Architecture Docs (WazzapAgents)

Dokumentasi ini ditujukan untuk dibaca **LLM / agent developer** agar cepat paham cara kerja project.

## Urutan baca
1. `00-overview.md` – gambaran sistem end-to-end.
2. `01-runtime-flow.md` – alur runtime per event.
3. `02-modules-map.md` – peta modul penting dan tanggung jawabnya.
4. `03-commands-and-permissions.md` – command, role, dan behavior.
5. `04-protocol-and-actions.md` – kontrak WebSocket antar komponen.
6. `05-state-data-and-db.md` – state, cache, dan storage (SQLite).

## Prinsip penting
- Node.js gateway menangani koneksi WhatsApp, interactive UI, parsing slash command, dan relay WS.
- Python bridge menangani batching pesan, LLM routing (LLM1/LLM2), aksi moderation, dan penulisan stats.
- `/dashboard` dibaca dari Node (query DB), sedangkan stats ditulis oleh Python (flush periodik).
- Event WS penting sekarang dikirim lewat jalur reliable (`sendReliable`) agar tidak hilang saat reconnect.
