import logger from "../../logger.js";
import { getSock } from "../connection.js";

const HELP_TEXT = `*WazzapAgents - Daftar Perintah*

*Umum (Semua Orang)*
• \`/help\` — Tampilkan pesan bantuan ini
• \`/info\` — Informasi profil, peran, dan chat
• \`/dashboard\` — Statistik penggunaan bot
• \`/join\` [link] — Masuk grup via link
• \`/sticker\` — Buat stiker (balas gambar)
• \`/owner-contact\` — Kirim kartu kontak owner

*Pengaturan & Moderasi (Admin)*
• \`/setting\` — Menu pengaturan interaktif
• \`/prompt\` [teks] — Atur kepribadian bot
• \`/reset\` — Hapus memori percakapan
• \`/model\` — Pilih model AI
• \`/mode\` [auto|prefix|hybrid] — Mode respon
• \`/trigger\` [opsi] — Atur pemicu respon
• \`/idle\` [n|n-m|off] — Auto-trigger LLM2 setelah n pesan tanpa bicara
• \`/group-status\` [text/media] — Kirim group status

_Ketik perintah tanpa argumen untuk melihat status saat ini._`;

async function handleHelp({ chatId }) {
  const sock = getSock();
  try {
    await sock.sendMessage(chatId, { text: HELP_TEXT });
  } catch (err) {
    logger.warn({ err, chatId }, "failed sending /help response");
  }
}

export { handleHelp };
