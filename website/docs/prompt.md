---
sidebar_position: 5
---

# Mengatur Prompt

Prompt adalah "instruksi rahasia" yang menentukan **siapa bot ini** dan **apa yang harus dilakukannya**. Ini adalah fitur paling powerful dari WazzapAgents.

## Cara Kerja

Ketika kamu mengetik `/prompt <teks>`, instruksi tersebut disimpan untuk chat ini dan dikirim ke AI setiap kali bot mau merespons. Bot akan berperilaku sesuai instruksi yang kamu berikan.

## Struktur Prompt yang Baik

```
[Siapa bot ini / nama dan perannya]
[Bahasa yang digunakan]
[Apa yang harus dilakukan]
[Apa yang tidak boleh dilakukan]
[Aturan khusus]
```

## Tips Menulis Prompt

1. **Spesifik** — Semakin detail instruksinya, semakin konsisten perilaku bot
2. **Gunakan kata kerja imperatif** — "Jawab dengan...", "Jangan pernah...", "Selalu..."
3. **Sebutkan batasan** — Apa yang boleh dan tidak boleh dilakukan
4. **Tentukan gaya bahasa** — Formal, santai, gaul, dll.
5. **Untuk moderasi** — Sebutkan dengan jelas kapan bot boleh bertindak

## Format Teks WhatsApp yang Didukung

Bot bisa menggunakan format teks WhatsApp berikut dalam jawabannya:

| Format | Hasil |
|--------|-------|
| `*teks*` | **tebal** |
| `_teks_` | *miring* |
| `~teks~` | ~~coret~~ |
| `` `teks` `` | `kode` |
| `> teks` | kutipan |

## Alternatif: Prompt via Deskripsi Grup

Selain `/prompt`, admin bisa memasukkan aturan bot langsung di **deskripsi grup** dengan format khusus:

```
Deskripsi grup biasa di sini...

<prompt_override>
Instruksi untuk bot di sini...
</prompt_override>
```

Teks di dalam tag `<prompt_override>` akan menjadi instruksi tambahan untuk bot, tanpa perlu mengetik `/prompt`.
