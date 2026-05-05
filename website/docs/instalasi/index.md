---
sidebar_position: 1
---

# Panduan Instalasi

Panduan untuk menginstall dan menjalankan WazzapAgents di server atau komputer lokal Kamu.

## Prasyarat

| Software | Versi | Catatan |
|----------|-------|---------|
| Node.js | 18+ | Tested dengan Node 25 |
| pnpm | 9+ | `npm i -g pnpm` atau `corepack enable pnpm` |
| Python | 3.10+ | Untuk bridge |
| SQLite | 3.x | Biasanya sudah terinstall di OS |

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Chomosuke9/WazzapAgents.git && cd WazzapAgents
```
![Clone Repository](/img/2026-05-05-143910_hyprcap.png)

Cek apakah berhasil:

![Check Repository](/img/check_repo.png)

Jika berhasil tandanya akan muncul isi project ini:

![Check Repository Success](/img/check_success.png)


### 2. Setup Environment Variables

```bash
cp .env.example .env
```
Kemudian edit dengan text editor (disini menggunakan `nano`)
```bash
nano .env
```

![Nano .env](/img/nano_env.png)

Edit `.env` dan isi minimal:
```bash
# Wajib — URL WebSocket ke Python bridge
LLM_WS_ENDPOINT=ws://localhost:8080/ws
                                /\
                                ||
                                ||
                                ——
# Port(angka) ini bisa di ganti jadi apa saja, yang penting tidak terpakai port-nya.

# Contoh: vivy,ivy,vivi,ivi,vy  (display name: vivy, aliases: ivy, vivi, etc.)
ASSISTANT_NAME=LLM

# Ini endpoint dan kunci API dari AI yang kamu pakai, WAJIB OpenAi compatible. 
LLM2_ENDPOINT=
LLM2_API_KEY=
```

Kemudian tambahkan JID/LID nomor kamu.
```bash
# Contoh: 628123456789@s.whatsapp.net, 193058310034@lid
BOT_OWNER_JIDS=
```
:::note
Setiap nomor bisa memakai JID(nomor handphone) atau LID(ID yang diberikan WhatsApp), jika JID(nomor handphone) tidak bekerja, gunakan LID dengan mengetik `/info` untuk mendapatkan LID-nya.

[**Cara lengkapnya ada disini**](/instalasi/cara-mendapatkan-lid.md)
:::


### 3. Install Dependensi Node.js dan Python.
**Node.js**:
```bash
pnpm install
```
**Python**:
```bash
pip install -r requirements.txt
```

Atau dengan virtual environment (direkomendasikan):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 4. Menjalankan

Dua komponen harus berjalan bersamaan:
:::tip
Gunakan **Terminal multiplexer** agar bisa menjalankan _service_-nya di background. Kamu bisa memakai _Tmux_, _Zellij_, _Byobu_, atau apapun yang kamu suka. 
:::
**Terminal 1 — Python Bridge:**
```bash
python -m python.bridge.main
```

**Terminal 2 — Node.js Gateway:**
```bash
pnpm dev
```

Saat pertama kali jalan, gateway akan menampilkan QR code di terminal. Scan dengan WhatsApp untuk pairing.

### 5. Menambahkan model
Kamu harus menambahkan model Ai yang ingin kamu gunakan terlebih dahulu, tanpa modelnya, bot ini hampir tidak bisa melakukan apapun.

Langkah-langkah:
1. Pastikan nomor kamu sudah terdaftar sebagai nomor owner, [**kalau belum atur itu sekarang juga**](/instalasi/cara-mendapatkan-lid.md).
2. Kirim command `/modelcfg add` ke bot nya, formatnya seperti ini

![model add](/img/slash_model_add.jpg)

Perhatikan, format nya adalah seperti ini
```txt
/model add <id model>|<nama model>|<deskripsi model>|<vision support modelnya>
```

:::warning
1. Gunakan `|` sebagai pemisah. 
2. `<id model>` WAJIB akurat, tidak ada kode yang memperbaiki id nya secara otomatis.
3. `<vision support modelnya>` WAJIB benar, jika modelnya tidak support _vision_, JANGAN di atur ke `true`, itu akan menyebabkan error dan bot nya jadi tidak menjawab.
:::

### Verifikasi model.
Ketik `/setting` untuk membuka menu seperti ini

![slash setting](/img/slash_setting.jpg)

kemudian pencet bagian `Change model`, jika kamu melakukan [langkah tadi](/instalasi#5-Menambahkan-model) dengan benar, maka akan keluar hasil seperti ini

![slash setting model](/img/gpt_model.jpg)


## Sub-Agent
**_Sub-Agent_** adalah agent pembantu dari WazzapAgents, ini adalah kekuatan asli dari asisten Ai yang sesungguhnya. Dengan ini, asisten Ai-mu bisa benar benar menjadi `asisten` yang benar benar bekerja dan berguna.

### Kapan Sub-Agent Dipakai?

Aktifkan Sub-Agent jika kamu ingin bot mengerjakan tugas yang lebih berat dari sekadar membalas chat, misalnya:

- membaca dan memproses file yang dikirim pengguna;
- mengekstrak tabel atau ringkasan dari dokumen;
- menjalankan script kecil atau command-line tool;
- melakukan riset lalu mengembalikan laporan terstruktur;
- membuat file hasil kerja dan mengirimkannya kembali ke WhatsApp.

Jika bot hanya dipakai untuk ngobrol biasa atau moderasi grup, Sub-Agent boleh tetap dimatikan.

### 1. Jalankan WazzapSubAgents

Clone service Sub-Agent dan siapkan environment-nya:

```bash
git clone https://github.com/Chomosuke9/WazzapSubAgents.git && cd WazzapSubAgents
cp .env.example .env
```

Edit `.env`, lalu isi minimal:

```bash
LLM_API_KEY=<api key kamu>
AGENT_MODEL=<model untuk sub-agent>
```

Jalankan dengan Docker Compose:

```bash
docker-compose up -d
```

Service ini akan membuka:

- API utama: `http://localhost:5000`
- executor sidecar: `http://localhost:5001`

### 2. Hubungkan WazzapAgents ke Sub-Agent

Di `.env` WazzapAgents, tambahkan:

```bash
SUBAGENT_URL=http://localhost:5000
SUBAGENT_WEBHOOK_PORT=8081
SUBAGENT_WEBHOOK_URL=http://localhost:8081/subagent/callback
```

Jika WazzapSubAgents berjalan di Docker dan harus memanggil balik WazzapAgents di host, gunakan:

```bash
SUBAGENT_WEBHOOK_URL=http://host.docker.internal:8081/subagent/callback
```

:::tip
`docker-compose.yml` di WazzapSubAgents sudah menambahkan `host.docker.internal` untuk Linux dengan `host-gateway`.
:::

### 3. Pastikan Folder File Dibagi Bersama

Untuk tugas yang memakai file, WazzapAgents dan WazzapSubAgents harus bisa membaca folder host yang sama. Default WazzapSubAgents adalah `/storage`.

```bash
SUBAGENT_STORAGE_DIR=/storage
WORKDIR_BASE=/storage/subagent_work
```

Pastikan WazzapAgents bisa membaca file yang dikembalikan Sub-Agent. Jika tidak, hasil file tidak bisa dikirim balik sebagai media WhatsApp.

### 4. Aktifkan dari WhatsApp

Hanya owner bot yang bisa menyalakan Sub-Agent:

```txt
/subagent on
/subagent off
/subagent global on
/subagent global off
```

Cek status untuk chat saat ini:

```txt
/subagent
```

### 5. Uji Alurnya

Setelah kedua service berjalan:

1. Kirim `/subagent on` di chat.
2. Minta tugas yang membutuhkan tool, misalnya: "Baca dokumen ini dan ekstrak tabelnya."
3. Main agent akan memberi acknowledgement.
4. Sub-Agent mengirim progress lewat webhook.
5. Setelah selesai, WazzapAgents merangkum hasil dan mengirim file output jika ada.

:::warning
Sub-Agent bisa menjalankan tool dan operasi file. Jalankan hanya di server yang kamu kontrol, jaga API key tetap privat, dan aktifkan hanya untuk chat yang kamu percaya.
:::

## Variabel Environment

### Gateway (Node.js)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `LLM_WS_ENDPOINT` | *(wajib)* | URL WebSocket ke bridge |
| `INSTANCE_ID` | `default` | Identifier instance gateway |
| `LLM_WS_TOKEN` | *(kosong)* | Bearer token untuk autentikasi WS |
| `DATA_DIR` | `./data` | Direktori data runtime |
| `MEDIA_DIR` | `./data/media` | Direktori penyimpanan media |
| `LOG_LEVEL` | `info` | Level log (debug, info, warn, error) |
| `WS_RECONNECT_MS` | `5000` | Interval reconnect WS dalam ms |
| `GROUP_METADATA_TIMEOUT_MS` | `8000` | Timeout fetch metadata grup |
| `DOWNLOAD_TIMEOUT_MS` | `60000` | Timeout download media |
| `SEND_TIMEOUT_MS` | `60000` | Timeout kirim pesan |
| `UPSERT_CONCURRENCY` | `2` | Concurrency pemrosesan pesan |
| `BOT_OWNER_JIDS` | *(kosong)* | JID owner, pisahkan koma |

### Bridge (Python)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `HISTORY_LIMIT` | `20` | Jumlah pesan history per chat |
| `INCOMING_DEBOUNCE_SECONDS` | `5` | Debounce window untuk batching |
| `INCOMING_BURST_MAX_SECONDS` | `20` | Maksimum durasi burst window |
| `HISTORY_LIMIT` | `20` | Jumlah pesan history per chat |
| `INCOMING_DEBOUNCE_SECONDS` | `5` | Debounce window untuk batching |
| `INCOMING_BURST_MAX_SECONDS` | `20` | Maksimum durasi burst window |
| `ASSISTANT_NAME` | `LLM` | Nama tampilan bot di konteks |
| `CONTEXT_TIME_UTC_OFFSET_HOURS` | *(auto)* | UTC offset untuk timestamp |
| `ASSISTANT_NAME` | `LLM` | Nama tampilan bot di konteks |
| `CONTEXT_TIME_UTC_OFFSET_HOURS` | *(auto)* | UTC offset untuk timestamp |

### LLM1 (Gating)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `LLM1_ENDPOINT` | *(OpenAI default)* | Endpoint API LLM1 |
| `LLM1_MODEL` | `openai/gpt-oss-20b` | Model untuk gating |
| `LLM1_API_KEY` | *(kosong)* | API key LLM1 |
| `LLM1_TEMPERATURE` | `0` | Temperature untuk LLM1 |
| `LLM1_TIMEOUT` | `8` | Timeout dalam detik |
| `LLM1_HISTORY_LIMIT` | `20` | Limit history untuk konteks LLM1 |
| `LLM1_MESSAGE_MAX_CHARS` | `500` | Maks karakter per pesan untuk LLM1 |
| `LLM1_ENABLE_MEDIA_INPUT` | `0` | Aktifkan input multimodal LLM1 |
| `LLM1_FALLBACK_ENDPOINT` | *(reuse LLM1)* | Endpoint fallback |
| `LLM1_FALLBACK_MODEL` | *(kosong)* | Model fallback |

### LLM2 (Responder)

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `LLM2_ENDPOINT` | *(OpenAI default)* | Endpoint API LLM2 |
| `LLM2_MODEL` | `gpt-5.3` | Model untuk responder |
| `LLM2_API_KEY` | *(kosong)* | API key LLM2 |
| `LLM2_TEMPERATURE` | `0.5` | Temperature untuk LLM2 |
| `LLM2_TIMEOUT` | `20` | Timeout dalam detik |
| `LLM2_RETRY_MAX` | `0` | Maks retry saat timeout |
| `LLM2_RETRY_BACKOFF_SECONDS` | `0.8` | Backoff antar retry |
| `LLM2_ENABLE_MEDIA_INPUT` | `1` | Aktifkan input multimodal LLM2 |
| `LLM2_FALLBACK_ENDPOINT` | *(reuse LLM2)* | Endpoint fallback |
| `LLM2_FALLBACK_MODEL` | *(kosong)* | Model fallback |

### Logging Bridge

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `BRIDGE_LOG_LEVEL` | `info` | Level log bridge |
| `BRIDGE_LOG_PROMPT_FULL` | `0` | Log prompt LLM2 lengkap |
| `BRIDGE_LOG_EXTRAS_LIMIT` | `4000` | Limit karakter extras di log |
| `BRIDGE_LOG_CHAT_LABEL_WIDTH` | `24` | Lebar label chat di log |
| `BRIDGE_SLOW_BATCH_LOG_MS` | `2000` | Threshold log batch lambat |
