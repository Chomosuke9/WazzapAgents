---
sidebar_position: 7
---

# Fitur-Fitur Bot

## Membaca Gambar & Media

Bot bisa **memahami dan mendeskripsikan** gambar, foto, stiker, dan dokumen yang dikirim ke chat. Tinggal kirim gambar dan bot akan memahami konteksnya secara otomatis.

**Batasan:**
- Maksimal **2 file** per pesan yang diproses
- Ukuran total maksimal **5 MB**

## Sinyal Centang Biru (Read)

Setelah bot selesai memproses pesanmu (memutuskan mau jawab atau tidak), bot akan otomatis **mencentang biru** pesanmu. Ini tanda bot sudah "membaca" dan memproses pesanmu.

## Indikator Mengetik

Ketika bot sedang menyusun balasan, kamu akan melihat **"[Nama Bot] sedang mengetik..."** — persis seperti kalau teman sedang menulis pesan.

## Memory / Konteks Percakapan

Bot **mengingat konteks percakapan** beberapa pesan terakhir, sehingga:
- Bot tahu apa yang dibicarakan sebelumnya
- Bot bisa menjawab pertanyaan lanjutan tanpa perlu mengulang konteks

Gunakan `/reset` untuk menghapus memori ini dan mulai dari awal.

## Reply ke Pesan

Bot me-**reply** ke pesan tertentu saat menjawab, sehingga jelas pesan mana yang sedang dibalas — terutama berguna di grup yang ramai.

## Deteksi Anggota Baru

Bot secara otomatis **mendeteksi ketika ada anggota baru** bergabung ke grup dan bisa menyapa mereka jika prompt-nya mengatur ini.

## Prompt via Deskripsi Grup

Admin bisa menyisipkan instruksi bot langsung di **deskripsi grup** tanpa perlu mengetik `/prompt`:

```
Deskripsi grup biasa...

<prompt_override>
Instruksi bot di sini...
</prompt_override>
```
