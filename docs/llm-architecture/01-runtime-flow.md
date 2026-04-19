# 01 - Runtime Flow

## A. Startup flow
1. Node bootstrap (`src/index.js`): init DB, start WhatsApp, connect WS client.
2. Python bridge (`python/bridge/main.py`): start WS server, init in-memory state.
3. Node kirim `hello` ke Python setelah WS open.

## B. Incoming message flow (WA -> Node -> Python)
1. Node menerima event `messages.upsert`.
2. Node command listener memproses slash command/button terlebih dulu.
3. Node inbound pipeline membentuk payload `incoming_message`.
4. Node kirim payload ke Python via WS.
5. Python menaruh payload ke pending buffer per chat.
6. Setelah debounce/burst window, Python proses batch:
   - trigger check
   - LLM1 decision
   - LLM2 generation/tool calls
7. Python kirim action ke Node.
8. Node eksekusi action ke WA, kirim `action_ack` / `error` balik.

## C. Model switching flow
1. User pilih model dari interactive menu (`model_select:<modelId>`).
2. Node set `chat_settings.llm2_model` di DB lokal.
3. Node kirim WS control event reliable:
   - `set_llm2_model` (authoritative sync)
   - `invalidate_llm2_model` (fallback cache clear)
4. Python update DB/cache model chat.
5. Request LLM berikutnya memakai model baru.

## D. Dashboard flow
1. Python record counters (RAM buffer) selama pemrosesan.
2. Python flush ke DB periodik.
3. `/dashboard` di Node membaca tabel stats dan mengirim ringkasan ke chat.

## E. Reset flow
1. User jalankan `/reset`.
2. Node kirim `clear_history` via `sendReliable()`.
3. Python clear history per chat.

## F. Failure/reconnect behavior
- Saat WS drop, event non-reliable bisa hilang.
- Event reliable disimpan queue lalu dikirim ulang saat koneksi open.
