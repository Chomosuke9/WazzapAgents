---
sidebar_position: 3
---

# Daftar Perintah

Semua perintah diawali dengan `/` (garis miring). Di grup, sebagian besar perintah hanya bisa digunakan oleh **admin**. Di chat pribadi, semua pengguna bisa memakai semua perintah.

## Ringkasan

| Perintah | Fungsi | Siapa Bisa |
|----------|--------|------------|
| `/prompt` | Lihat prompt aktif | Admin (grup), Siapa saja (pribadi) |
| `/prompt <teks>` | Set prompt baru | Admin (grup), Siapa saja (pribadi) |
| `/prompt -` | Hapus prompt | Admin (grup), Siapa saja (pribadi) |
| `/reset` | Reset memori bot | Admin (grup), Siapa saja (pribadi) |
| `/permission` | Cek level izin moderasi | Admin grup |
| `/permission 0-3` | Set level izin moderasi | Admin grup |
| `/info` | Info pengguna & grup | Semua orang |
| `/broadcast <pesan>` | Kirim ke semua grup | Owner bot saja |

---

## `/prompt`

Mengatur **kepribadian, peran, dan aturan** bot di chat ini.

### Melihat prompt saat ini
```
/prompt
```

### Mengatur prompt baru
```
/prompt <teks aturanmu>
```
**Batas:** maksimal 4000 karakter.

### Menghapus prompt (kembali ke default)
```
/prompt -
```
atau `/prompt clear` atau `/prompt reset`

:::info
Prompt berlaku **per chat/grup**. Pengaturan di grup A tidak mempengaruhi grup B.
:::

---

## `/reset`

Menghapus **memori/riwayat percakapan** bot untuk chat ini.

```
/reset
```

Gunakan ketika:
- Bot sudah "keluar jalur" dan jawabannya tidak nyambung
- Ingin memulai percakapan baru dari awal
- Setelah mengganti prompt besar-besaran

---

## `/info`

Menampilkan informasi pengguna dan grup.

```
/info
```

Menampilkan: nama, nomor WhatsApp, peran di grup, nama grup, jumlah anggota, status admin bot, dan deskripsi grup.

**Bisa digunakan oleh semua orang**, tidak perlu jadi admin.

---

## `/broadcast`

Mengirim pesan ke semua grup tempat bot terdaftar.

```
/broadcast <pesan>
```

Atau **reply** ke pesan tertentu dengan `/broadcast` untuk meneruskan pesan itu ke semua grup.

:::warning
Hanya bisa digunakan oleh **pemilik bot (owner)**. Pengguna biasa tidak bisa menggunakan perintah ini.
:::
