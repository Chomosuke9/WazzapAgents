# Interactive Messages — Implementation Notes

> **Baca ini dulu sebelum menyentuh file apapun di folder ini.**
> Dokumen ini merangkum semua hal penting yang ditemukan saat mengimplementasikan
> pesan interaktif WhatsApp di Baileys v7. Banyak hal di sini **tidak terdokumentasi**
> secara resmi dan hanya ditemukan melalui riset dan trial-error.

---

## Daftar File

| File | Isi |
|------|-----|
| `sendInteractive.js` | Core helper + NativeFlow functions (quick reply, URL, copy, call, list, combined, rich) |
| `sendButtons.js` | Legacy button formats (ButtonsMessage, TemplateMessage) |
| `sendCarousel.js` | Carousel / swipeable cards |
| `index.js` | Barrel re-export semua fungsi publik |

---

## Bagaimana Cara Kerjanya (Wajib Dibaca)

### 1. Jangan pakai `sock.sendMessage` untuk `interactiveMessage`

`sock.sendMessage` melewati `prepareWAMessageMedia` yang tidak mendukung `interactiveMessage`
dan akan throw `"Invalid media type"`. Semua interactive message **harus** menggunakan:

```js
import { generateWAMessageFromContent, proto } from 'baileys';

const msg = generateWAMessageFromContent(jid, { /* content */ }, { userJid: sock.user.id });
await sock.relayMessage(jid, msg.message, { messageId: msg.key.id, additionalNodes: [...] });
```

### 2. Wrapper proto yang benar

Semua `interactiveMessage` harus dibungkus dalam:

```js
{
  viewOnceMessage: {
    message: {
      messageContextInfo: {
        deviceListMetadata: {},
        deviceListMetadataVersion: 2,
      },
      interactiveMessage: proto.Message.InteractiveMessage.create({ ... }),
    },
  },
}
```

**Jangan** langsung taruh `interactiveMessage` di level atas — tidak akan dirender.

### 3. `proto.create()`, bukan `.fromObject()`

Baileys v7 **menghapus** `.fromObject()` dari semua tipe proto. Selalu gunakan `.create()`:

```js
// ✅ Benar
proto.Message.InteractiveMessage.create({ ... })
proto.Message.InteractiveMessage.Header.create({ ... })
proto.Message.InteractiveMessage.Body.create({ text: '...' })
proto.Message.InteractiveMessage.NativeFlowMessage.create({ buttons: [...] })

// ❌ Salah — akan throw di Baileys v7
proto.Message.InteractiveMessage.fromObject({ ... })
```

### 4. Binary nodes (`additionalNodes`) — WAJIB

Tanpa `additionalNodes` yang benar, WhatsApp akan menampilkan:
> *"Anda telah menerima pesan, tetapi versi WhatsApp anda tidak mendukungnya."*

Struktur yang benar (berlaku untuk **semua** tipe interactive message termasuk carousel):

```js
function buildInteractiveNodes(jid, badge = true) {
  const nodes = [
    {
      tag: 'biz',
      attrs: {},
      content: [
        {
          tag: 'interactive',
          attrs: { type: 'native_flow', v: '1' },   // SELALU native_flow
          content: [
            { tag: 'native_flow', attrs: { v: '9', name: 'mixed' } }, // child WAJIB ada
          ],
        },
      ],
    },
  ];
  if (badge && !isJidGroup(jid)) {
    nodes.push({ tag: 'bot', attrs: { biz_bot: '1' } }); // badge AI — private chat only
  }
  return nodes;
}
```

**Hal-hal penting:**
- `type` di `interactive` attrs selalu `'native_flow'` — bahkan untuk carousel
- Child node `native_flow` dengan `v: '9'` dan `name: 'mixed'` **harus ada** di dalam `interactive`
- Node `bot` (`biz_bot: '1'`) hanya untuk private chat (bukan `@g.us`) — ini yang menciptakan badge AI
- Carousel sempat dicoba dengan `type: 'carousel'` → **error 479**, tidak bekerja

### 5. Error codes yang relevan

| Error | Artinya |
|-------|---------|
| `"Invalid media type"` | Pakai `sock.sendMessage` untuk `interactiveMessage` — ganti ke `relayMessage` |
| Pesan "versi tidak didukung" | `additionalNodes` salah/tidak ada |
| Error 479 (ACK) | Struktur binary stanza tidak valid di server — paling sering: tipe node salah, atau field proto hilang |

---

## Fungsi-Fungsi yang Tersedia

### `sendRichMessage(sock, jid, options)` — Fungsi Utama / Universal

Kirim pesan styled dengan footer, header opsional, dan tombol opsional.
Ini adalah fungsi paling fleksibel — gunakan ini sebagai default untuk pesan bot.

```js
// Pesan teks biasa dengan footer AI
await sendRichMessage(sock, jid, {
  text: 'Halo!',
  footer: 'Pesan ini dibuat oleh AI',
});

// Dengan header (title) dan tombol
await sendRichMessage(sock, jid, {
  title: 'Konfirmasi',
  text: 'Lanjutkan pesanan?',
  footer: 'Tap tombol di bawah',
  buttons: [
    { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Ya', id: 'yes' }) },
    { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Tidak', id: 'no' }) },
  ],
});

// Dengan image header
await sendRichMessage(sock, jid, {
  image: { url: 'https://example.com/img.jpg' },
  title: 'Produk A',
  text: 'Deskripsi produk',
  footer: 'Rp 100.000',
});

// Tanpa badge AI (misalnya untuk broadcast)
await sendRichMessage(sock, jid, { text: 'Halo', footer: 'Broadcast 📢', badge: false });

// Dengan @mentions
await sendRichMessage(sock, jid, {
  text: 'Halo @628123456789!',
  footer: 'Bot',
  mentions: ['628123456789@s.whatsapp.net'],
});
```

**Catatan tentang `title` tanpa media:**
`Header.title` di proto mungkin tidak dirender secara visual oleh WhatsApp jika tidak ada
image/video. Jika ingin header yang pasti terlihat tanpa media, pertimbangkan
untuk memasukkan teks judul langsung ke `text` dengan formatting (`*bold*`).

### `sendQuickReply(sock, jid, body, buttons, options)`

```js
await sendQuickReply(sock, jid, 'Pilih menu:', [
  { id: 'menu_1', displayText: 'Produk' },
  { id: 'menu_2', displayText: 'Hubungi CS' },
], { title: 'Menu', footer: 'Bot v1' });
```

### `sendUrlButtons(sock, jid, body, buttons, options)`

```js
await sendUrlButtons(sock, jid, 'Kunjungi kami:', [
  { displayText: 'Website', url: 'https://example.com' },
]);
```

### `sendCopyCode(sock, jid, body, copyCode, displayText, options)`

```js
await sendCopyCode(sock, jid, 'Kode promo:', 'PROMO2024', 'Salin Kode');
```

### `sendCombinedButtons(sock, jid, body, buttons, options)`

Campurkan berbagai tipe tombol dalam satu pesan:

```js
await sendCombinedButtons(sock, jid, 'Pilih aksi:', [
  { type: 'reply', displayText: 'Konfirmasi', id: 'confirm' },
  { type: 'url',   displayText: 'Detail', url: 'https://example.com' },
  { type: 'copy',  displayText: 'Salin', copyCode: 'CODE123' },
  { type: 'call',  displayText: 'Telepon', phoneNumber: '+6281234567890' },
]);
```

### `sendNativeFlow(sock, jid, body, buttons, options)`

Raw NativeFlow — untuk tombol tipe lain (`single_select`, dll.) dengan format pre-built:

```js
await sendNativeFlow(sock, jid, 'Pilih opsi:', [
  {
    name: 'single_select',
    buttonParamsJson: JSON.stringify({
      title: 'Pilih',
      sections: [{ title: 'Kategori', rows: [{ title: 'Item', id: 'item1' }] }],
    }),
  },
]);
```

### `sendList(sock, jid, content, options)`

List/dropdown menggunakan `listMessage` via `sock.sendMessage` biasa (bukan interactive):

```js
await sendList(sock, jid, {
  title: 'Menu',
  buttonText: 'Buka Daftar',
  description: 'Tap untuk melihat pilihan',
  footer: 'Pilih satu item',
  sections: [{
    title: 'Kategori',
    rows: [{ rowId: 'item1', title: 'Item 1', description: 'Deskripsi' }],
  }],
});
```

### `sendCarousel(sock, jid, cards, options)` — ⚠️ Eksperimental

Carousel / swipeable cards. **Status: error 479 saat pengiriman, belum resolved.**

```js
await sendCarousel(sock, jid, [
  {
    image: { url: 'https://example.com/img.jpg' },
    title: 'Kartu 1',
    body: 'Deskripsi kartu 1',
    footer: 'Footer',
    buttons: [{ name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Pilih', id: 'c1' }) }],
  },
], { title: 'Produk Unggulan', text: 'Swipe untuk lihat lebih' });
```

---

## Mentions di Interactive Message

Mentions bekerja melalui `contextInfo.mentionedJid` di proto `InteractiveMessage`.
`_sendInteractive` menerima parameter `mentions` (array JID) dan menyuntikkannya:

```js
// Internal — di _sendInteractive:
if (mentions.length > 0) {
  interactiveContent.contextInfo = proto.ContextInfo.create({ mentionedJid: mentions });
}
```

`sendRichMessage` meneruskan `options.mentions` ke sini secara otomatis.

---

## Badge AI

Badge AI (label "AI" di pojok pesan) muncul dari node `{ tag: 'bot', attrs: { biz_bot: '1' } }`
di `additionalNodes`. **Hanya bekerja di private chat (`@s.whatsapp.net` / `@lid`).**
Di group chat, node ini diabaikan — tidak ada badge.

Untuk mematikan badge: `badge: false` di `sendRichMessage`, atau gunakan `buildInteractiveNodes(jid, false)`.

---

## Integrasi dengan LLM Replies (`outbound.js`)

Semua teks reply LLM dikirim via `sendRichMessage` dengan footer `'Pesan ini dibuat oleh AI'`.
Jika `sendRichMessage` gagal (apapun alasannya), fallback otomatis ke `sock.sendMessage`.

```js
// Di src/wa/outbound.js
try {
  sentMsg = await sendRichMessage(sock, chatId, {
    text: renderedText.text,
    footer: AI_FOOTER,
    quoted: quoted || undefined,
    mentions: renderedText.mentions,
  });
} catch (err) {
  // Fallback — pesan tetap terkirim
  sentMsg = await sock.sendMessage(chatId, { text, mentions }, { quoted });
}
```

---

## Status Setiap Tipe Pesan

| Tipe | Status | Catatan |
|------|--------|---------|
| `sendQuickReply` | ✅ Bekerja | |
| `sendUrlButtons` | ✅ Bekerja | |
| `sendCopyCode` | ✅ Bekerja | |
| `sendCombinedButtons` | ✅ Bekerja | |
| `sendNativeFlow` | ✅ Bekerja | Base function untuk semua NativeFlow |
| `sendRichMessage` | ✅ Bekerja | `title` tanpa media mungkin tidak render |
| `sendList` | ✅ Bekerja | Pakai `sock.sendMessage`, bukan `relayMessage` |
| `sendCarousel` | ⚠️ Error 479 | Ditunda — belum ditemukan solusi |
| `sendLegacyButtons` | ❓ Tidak diuji | Format lama, kemungkinan deprecated |
| `sendTemplate` | ❓ Tidak diuji | Format lama, kemungkinan deprecated |

---

## `/debug` Command

Untuk menguji semua tipe di WhatsApp:

```
/debug buttons      → quick_reply, cta_url, cta_copy, cta_call
/debug menu         → single_select dropdown
/debug list         → sendList
/debug rich         → sendRichMessage (tanpa & dengan tombol)
/debug combined     → semua tipe tombol dalam satu pesan
/debug broadcast    → preview format pesan broadcast
/debug all          → semua 6 tipe di atas sekaligus
/debug carousel     → carousel (eksperimental, mungkin error)
/debug carousel-img → carousel dengan image header (eksperimental)
```
