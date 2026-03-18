---
sidebar_position: 2
---

# Cara Memulai

## Menambahkan Bot ke Grup

1. **Tambahkan nomor bot** ke grup WhatsApp-mu seperti menambahkan anggota biasa.
2. Bot akan otomatis aktif dan siap digunakan.
3. **Opsional tapi penting untuk moderasi:** Jadikan bot sebagai **admin grup** jika ingin bot bisa menghapus pesan atau mengeluarkan anggota.

:::note
Tanpa status admin, bot tetap bisa mengobrol dan menjawab pesan, tapi tidak bisa melakukan tindakan moderasi (hapus/kick).
:::

## Cara Menjadikan Bot Admin

1. Buka **Info Grup** di WhatsApp
2. Ketuk nama bot dalam daftar anggota
3. Pilih **"Jadikan Admin"**

## Langkah Pertama yang Disarankan

Setelah bot masuk grup, lakukan ini secara berurutan:

1. **Cek info bot** dengan mengetik `/info` di chat — pastikan bot terdeteksi sebagai admin jika sudah dijadikan admin.
2. **Atur kepribadian bot** dengan `/prompt <instruksimu>` — ini menentukan bagaimana bot berperilaku di grup ini.
3. **Uji coba** dengan menyapa bot: `@Vivy halo!`
4. Jika ingin moderasi, baca dulu bagian [Sistem Permission](/permission) sebelum mengaktifkannya.

## Cara Bot Merespons di Grup

Di grup yang ramai, bot **tidak merespons setiap pesan**. Bot akan merespons jika:

- Pesan **mention bot** secara eksplisit (contoh: `@Vivy`)
- Pesan adalah **reply** ke pesan bot sebelumnya
- Bot menilai ada konteks penting yang perlu direspons
- Ada **kejadian penting** seperti anggota baru bergabung

Di **chat pribadi**, bot selalu merespons setiap pesan.
