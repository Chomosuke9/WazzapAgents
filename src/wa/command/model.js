import logger from '../../logger.js';
import { getSock } from '../connection.js';
import { sendNativeFlow } from '../interactive/index.js';
import { getAllActiveModels, getLlm2Model, getDefaultLlm2Model } from '../../db.js';

async function handleModel({ chatId, chatType, senderIsAdmin, senderIsOwner, args }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';
  const canUse = isPrivate || senderIsOwner || senderIsAdmin;

  if (!canUse) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins or bot owner can change the model.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const models = getAllActiveModels();
  if (models.length === 0) {
    try {
      await sock.sendMessage(chatId, { text: 'No models available. Ask the bot owner to add models using `/modelcfg`.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const currentModelId = getLlm2Model(chatId);
  const defaultModel = getDefaultLlm2Model();
  const activeModelId = currentModelId || defaultModel?.modelId || null;
  const activeModel = models.find((m) => m.modelId === activeModelId);

  const sections = models.map((m) => ({
    title: m.displayName,
    rows: [{
      title: m.displayName + (m.modelId === activeModelId ? ' ✓' : ''),
      description: m.description || '',
      id: `model_select:${m.modelId}`,
    }],
  }));

  try {
    await sendNativeFlow(sock, chatId, 'Pilih Model LLM', [
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Pilih Model',
          sections,
        }),
      },
    ], { footer: 'Model saat ini: ' + (activeModel?.displayName || 'default') });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /model interactive');
    try {
      await sock.sendMessage(chatId, { text: 'Failed to show model list. Please try again.' });
    } catch (e) { /* ignore */ }
  }
}

export { handleModel };
