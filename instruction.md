# Task: Create Interactive Message Utilities for a Baileys-Based Project

## Project Context

Your project uses `@whiskeysockets/baileys` (the official Baileys fork) installed as an
npm dependency. You are NOT allowed to modify the Baileys source code.

Your goal is to create standalone utility files **inside your own project** that wrap
`sock.sendMessage` with clean, reusable functions for every type of interactive message
that Baileys supports.

---

## Structure to Create

Create a directory `src/wa/interactive/` with these files:

```
src/wa/interactive/
├── index.js          ← re-exports everything from the other files
├── sendButtons.js    ← legacy ButtonsMessage and TemplateMessage
├── sendInteractive.js ← NativeFlow-based interactive messages (quick reply, URL, copy, call, list)
└── sendCarousel.js   ← Carousel/Cards messages
```

---

## How Baileys Processes Message Content

When you call `sock.sendMessage(jid, content, options)`, Baileys' internal
`generateWAMessageContent()` checks the shape of `content` and converts it to the
appropriate proto format. The following content shapes are supported:

### A. NativeFlow / InteractiveMessage (modern)
Pass an object with `interactiveMessage` key containing a plain JS object:
```js
await sock.sendMessage(jid, {
    interactiveMessage: {
        body: { text: 'Choose an option:' },
        footer: { text: 'Powered by Bot' },
        header: { title: 'Main Menu', hasMediaAttachment: false },
        nativeFlowMessage: {
            buttons: [
                { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Yes', id: 'yes' }) },
                { name: 'cta_url',     buttonParamsJson: JSON.stringify({ display_text: 'Website', url: 'https://example.com' }) },
                { name: 'cta_copy',    buttonParamsJson: JSON.stringify({ display_text: 'Copy Code', copy_code: 'ABC123' }) },
                { name: 'cta_call',    buttonParamsJson: JSON.stringify({ display_text: 'Call Us', phone_number: '+628xxx' }) }
            ],
            messageParamsJson: ''
        }
    }
}, { quoted: someMessage })
```

### B. ListMessage (single-select menu)
Pass an object with `listMessage` key:
```js
await sock.sendMessage(jid, {
    listMessage: {
        title: 'Menu',
        description: 'Please select:',
        buttonText: 'View Options',
        footerText: 'Footer',
        listType: 1,  // 1 = SINGLE_SELECT
        sections: [
            {
                title: 'Section 1',
                rows: [
                    { rowId: 'opt1', title: 'Option 1', description: 'Description' }
                ]
            }
        ]
    }
})
```

### C. ButtonsMessage (legacy — may not render on newer WhatsApp versions)
Requires the `proto` object from Baileys:
```js
const { proto } = require('@whiskeysockets/baileys')
await sock.sendMessage(jid, {
    buttonsMessage: proto.Message.ButtonsMessage.fromObject({
        contentText: 'Choose:',
        footerText: 'Footer',
        headerType: 1,  // 1 = EMPTY
        buttons: [
            { buttonId: 'id1', buttonText: { displayText: 'Button 1' }, type: 1 }
        ]
    })
})
```

### D. TemplateMessage (HydratedFourRowTemplate)
Requires the `proto` object from Baileys:
```js
const { proto } = require('@whiskeysockets/baileys')
await sock.sendMessage(jid, {
    templateMessage: proto.Message.TemplateMessage.fromObject({
        hydratedTemplate: {
            hydratedContentText: 'Body text',
            hydratedFooterText: 'Footer',
            hydratedTitleText: 'Title',
            hydratedButtons: [
                { index: 1, quickReplyButton: { displayText: 'Reply', id: 'r1' } },
                { index: 2, urlButton: { displayText: 'Website', url: 'https://example.com' } },
                { index: 3, callButton: { displayText: 'Call', phoneNumber: '+628xxx' } }
            ]
        }
    })
})
```

### E. Carousel / Cards
Pass `cards` array directly to `sock.sendMessage`:
```js
await sock.sendMessage(jid, {
    text: 'Check out our products:',
    title: 'Product Catalog',
    cards: [
        {
            image: { url: 'https://example.com/img.jpg' },
            title: 'Product A',
            body: 'Description',
            footer: 'Rp 100.000',
            buttons: [
                { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Buy', id: 'buy_a' }) }
            ]
        }
    ]
})
```

---

## File Specifications

### `src/wa/interactive/sendInteractive.js`

Exports these functions (all async, all accept `sock` as first param):

```
sendQuickReply(sock, jid, body, buttons, options?)
  buttons: Array<{ id: string, displayText: string }>
  options?: { footer?: string, title?: string, quoted?: WAMessage }

sendUrlButtons(sock, jid, body, buttons, options?)
  buttons: Array<{ displayText: string, url: string, merchantUrl?: string }>
  options?: { footer?: string, title?: string, quoted?: WAMessage }

sendCopyCode(sock, jid, body, copyCode, displayText?, options?)
  displayText defaults to 'Copy Code'
  options?: { footer?: string, quoted?: WAMessage }

sendCombinedButtons(sock, jid, body, buttons, options?)
  buttons: Array of one of:
    { type: 'url',   displayText: string, url: string }
    { type: 'reply', displayText: string, id: string }
    { type: 'copy',  displayText: string, copyCode: string }
    { type: 'call',  displayText: string, phoneNumber: string }
  options?: { footer?: string, title?: string, quoted?: WAMessage }

sendList(sock, jid, content, options?)
  content: {
    title: string, buttonText: string,
    sections: Array<{ title: string, rows: Array<{ rowId: string, title: string, description?: string }> }>,
    footer?: string, description?: string
  }
  options?: { quoted?: WAMessage }

sendNativeFlow(sock, jid, body, buttons, options?)
  buttons: Array<{ name: string, buttonParamsJson: string }>
  options?: { footer?: string, header?: { title?: string, subtitle?: string }, quoted?: WAMessage }
```

Implementation pattern — build a plain JS `interactiveMessage` object and pass to
`sock.sendMessage`. Do NOT import `proto`; native flow messages do not need it.

### `src/wa/interactive/sendButtons.js`

Exports these functions:

```
sendLegacyButtons(sock, jid, body, buttons, options?)
  buttons: Array<{ id: string, displayText: string }>
  options?: { footer?: string, title?: string, quoted?: WAMessage }

sendTemplate(sock, jid, body, buttons, options?)
  buttons: Array of one of:
    { index: number, quickReplyButton: { id: string, displayText: string } }
    { index: number, urlButton: { displayText: string, url: string } }
    { index: number, callButton: { displayText: string, phoneNumber: string } }
  options?: { footer?: string, title?: string, quoted?: WAMessage }
```

Both require `const { proto } = require('@whiskeysockets/baileys')` to build the proto
objects. Use `proto.Message.ButtonsMessage.fromObject()` and
`proto.Message.TemplateMessage.fromObject()` respectively.

### `src/wa/interactive/sendCarousel.js`

Exports one function:

```
sendCarousel(sock, jid, cards, options?)
  cards: Array<{
    image?: { url: string } | Buffer,
    video?: { url: string } | Buffer,
    title?: string,
    body?: string,
    footer?: string,
    buttons?: Array<{ name: string, buttonParamsJson: string }>
  }>
  options?: { text?: string, title?: string, subtitle?: string, footer?: string, quoted?: WAMessage }
```

Pass `cards` directly in the content object to `sock.sendMessage`. Do NOT import `proto`.

### `src/wa/interactive/index.js`

Re-export everything:
```js
"use strict"
module.exports = {
    ...require('./sendInteractive'),
    ...require('./sendButtons'),
    ...require('./sendCarousel')
}
```

---

## Code Style

- CommonJS (`"use strict"`, `module.exports = { ... }`, `require`)
- All exported functions must be `async`
- JSDoc on every exported function with `@param` and `@returns` and a short `@example`
- The `quoted` option is always passed as the third argument to `sock.sendMessage`:
  `await sock.sendMessage(jid, content, { quoted: options?.quoted })`
  Never embed `quoted` inside the message content
- Do NOT add try/catch; let errors propagate to the caller
- Do NOT create `.d.ts` type files

---

## Usage Example (for JSDoc reference)

```js
const {
    sendQuickReply,
    sendUrlButtons,
    sendCopyCode,
    sendCombinedButtons,
    sendList,
    sendNativeFlow,
    sendLegacyButtons,
    sendTemplate,
    sendCarousel
} = require('./src/wa/interactive')

// Quick reply
await sendQuickReply(sock, jid, 'Pilih menu:', [
    { id: 'menu_1', displayText: 'Daftar Produk' },
    { id: 'menu_2', displayText: 'Hubungi CS' }
], { title: 'Menu Utama', footer: 'Bot v1' })

// URL buttons
await sendUrlButtons(sock, jid, 'Kunjungi kami:', [
    { displayText: 'Website', url: 'https://example.com' }
], { footer: 'Klik untuk buka' })

// Copy code
await sendCopyCode(sock, jid, 'Kode promo Anda:', 'PROMO2024', 'Salin', {
    footer: 'Berlaku 7 hari'
})

// Combined buttons
await sendCombinedButtons(sock, jid, 'Pilih aksi:', [
    { type: 'reply', displayText: 'Konfirmasi', id: 'confirm' },
    { type: 'url',   displayText: 'Detail',     url: 'https://example.com' },
    { type: 'call',  displayText: 'Telepon',    phoneNumber: '+6281234567890' }
])

// List / menu
await sendList(sock, jid, {
    title: 'Menu Restoran',
    buttonText: 'Lihat Menu',
    sections: [
        {
            title: 'Makanan',
            rows: [
                { rowId: 'nasi_goreng', title: 'Nasi Goreng', description: 'Rp 25.000' },
                { rowId: 'mie_goreng',  title: 'Mie Goreng',  description: 'Rp 23.000' }
            ]
        }
    ],
    footer: 'Order via chat'
})

// Template (with proto — legacy)
await sendTemplate(sock, jid, 'Selamat datang!', [
    { index: 1, quickReplyButton: { id: 'start', displayText: 'Mulai' } },
    { index: 2, urlButton: { displayText: 'Website', url: 'https://example.com' } }
], { title: 'Halo!', footer: 'Tim Support' })

// Carousel
await sendCarousel(sock, jid, [
    {
        image: { url: 'https://example.com/p1.jpg' },
        title: 'Produk A',
        body: 'Deskripsi A',
        footer: 'Rp 100.000',
        buttons: [{ name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Beli', id: 'buy_a' }) }]
    },
    {
        image: { url: 'https://example.com/p2.jpg' },
        title: 'Produk B',
        body: 'Deskripsi B',
        footer: 'Rp 150.000',
        buttons: [{ name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Beli', id: 'buy_b' }) }]
    }
], { text: 'Produk Unggulan', footer: 'Swipe untuk lihat lebih' })

// Reply to a message
sock.ev.on('messages.upsert', async ({ messages }) => {
    const msg = messages[0]
    if (!msg.key.fromMe) {
        await sendQuickReply(sock, msg.key.remoteJid, 'Ada yang bisa kami bantu?', [
            { id: 'order_status', displayText: 'Cek Status Pesanan' },
            { id: 'contact_cs',   displayText: 'Hubungi CS' }
        ], { quoted: msg })
    }
})
```
