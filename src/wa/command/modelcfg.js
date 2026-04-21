import logger from '../../logger.js';
import { getSock } from '../connection.js';
import { sendNativeFlow } from '../interactive/index.js';
import { getAllActiveModels, getAllModels, getDefaultLlm2Model, addModel, updateModel, deleteModel } from '../../db.js';
import wsClient from '../../wsClient.js';

async function handleModelcfg({ chatId, senderId, senderIsOwner, args }) {
  const sock = getSock();
  if (!senderIsOwner) {
    try {
      await sock.sendMessage(chatId, { text: 'Only bot owner can use `/modelcfg`.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const parts = (args || '').trim().split(/\s+/);
  const subcommand = parts[0]?.toLowerCase() || '';
  const subArgs = parts.slice(1);

  if (!subcommand) {
    const modelRows = getAllActiveModels().map((m) => ({
      id: `modelcfg_default:${m.modelId}`,
      title: m.displayName,
      description: m.description || '',
    }));
    
    const allModelRows = getAllModels().map((m) => ({
      id: `modelcfg_remove:${m.modelId}`,
      title: m.displayName + (m.isActive ? '' : ' (inactive)'),
      description: m.description || `ID: ${m.modelId}`,
    }));

    const editModelRows = getAllModels().map((m) => ({
      id: `/modelcfg edit ${m.modelId}`,
      title: m.displayName + (m.isActive ? '' : ' (inactive)'),
      description: m.description || `ID: ${m.modelId}`,
    }));

    const buttons = [
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Default Model',
          sections: [{
            title: 'Select Default',
            rows: modelRows,
          }],
        }),
      },
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Edit Model',
          sections: [{
            title: 'Select Model to Edit',
            rows: editModelRows,
          }],
        }),
      },
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Remove Model',
          sections: [{
            title: 'Select Model to Remove',
            rows: allModelRows,
          }],
        }),
      },
    ];

    try {
      await sendNativeFlow(sock, chatId, 'Model Configuration', buttons, { footer: 'Bot Owner Only' });
    } catch (err) {
      logger.warn({ err, chatId }, 'failed sending /modelcfg menu');
      try {
        await sock.sendMessage(chatId, { text: 'Failed to show modelcfg menu.' });
      } catch (e) { /* ignore */ }
    }
    return;
  }

  if (subcommand === 'remove_menu') {
    const models = getAllModels();
    if (models.length === 0) {
      try {
        await sock.sendMessage(chatId, { text: 'No models to remove.' });
      } catch (err) { /* ignore */ }
      return;
    }
    const sections = [
      {
        title: 'Select Model to Remove',
        rows: models.map((m) => ({
          title: m.displayName + (m.isActive ? '' : ' (inactive)'),
          description: m.description || `ID: ${m.modelId}`,
          id: `modelcfg_remove:${m.modelId}`,
        })),
      },
    ];
    try {
      await sendNativeFlow(sock, chatId, '⚠️ Remove Model', [
        {
          name: 'single_select',
          buttonParamsJson: JSON.stringify({
            title: 'Hapus Model',
            sections,
          }),
        },
      ], { footer: 'Pilih model untuk dihapus' });
    } catch (err) {
      logger.warn({ err, chatId }, 'failed sending /modelcfg remove menu');
      try {
        await sock.sendMessage(chatId, { text: 'Failed to show remove menu.' });
      } catch (e) { /* ignore */ }
    }
    return;
  }

  switch (subcommand) {
    case 'list': {
      const models = getAllModels();
      if (models.length === 0) {
        try {
          await sock.sendMessage(chatId, { text: 'No models configured. Use `/modelcfg` add <model_id> <display_name> [description]' });
        } catch (err) { /* ignore */ }
        return;
      }
      const lines = ['*Daftar Model:*'];
      const defaultModel = getDefaultLlm2Model();
      for (const m of models) {
        const isDefault = defaultModel?.modelId === m.modelId;
        const status = m.isActive ? '✓' : '✗';
        lines.push(`${status} ${m.displayName} (${m.modelId})${isDefault ? ' [DEFAULT]' : ''}`);
        if (m.description) lines.push(`   ${m.description}`);
      }
      try {
        await sock.sendMessage(chatId, { text: lines.join('\n') });
      } catch (err) { /* ignore */ }
      break;
    }

    case 'add': {
      if (subArgs.length < 2) {
        try {
          await sock.sendMessage(chatId, { text: 'Usage: `/modelcfg` add <model_id> <display_name> [description]' });
        } catch (err) { /* ignore */ }
        return;
      }
      const [modelId, displayName, ...descParts] = subArgs;
      const description = descParts.join(' ');
      const success = addModel(modelId, displayName, description);
      if (success) {
        wsClient.sendReliable({ type: 'invalidate_default_model' });
      }
      try {
        await sock.sendMessage(chatId, { text: success ? `Model "${displayName}" added.` : `Model "${modelId}" already exists.` });
      } catch (err) { /* ignore */ }
      break;
    }

    case 'edit': {
      if (subArgs.length < 1) {
        try {
          await sock.sendMessage(chatId, { text: 'Usage: `/modelcfg` edit <model_id> [name=<name>] [desc=<desc>] [active=0|1] [order=<n>]' });
        } catch (err) { /* ignore */ }
        return;
      }
      const [modelId, ...editParts] = subArgs;
      const updates = {};
      for (const part of editParts) {
        const match = part.match(/^(name|desc|active|order)=(.+)$/);
        if (match) {
          const [, key, value] = match;
          if (key === 'name') updates.displayName = value;
          else if (key === 'desc') updates.description = value;
          else if (key === 'active') updates.isActive = value === '1' || value === 'true';
          else if (key === 'order') updates.sortOrder = parseInt(value, 10);
        }
      }
      const success = updateModel(modelId, updates);
      if (success) {
        wsClient.sendReliable({ type: 'invalidate_default_model' });
      }
      try {
        await sock.sendMessage(chatId, { text: success ? `Model "${modelId}" updated.` : `Model "${modelId}" not found.` });
      } catch (err) { /* ignore */ }
      break;
    }

    case 'remove':
    case 'delete': {
      if (subArgs.length < 1) {
        try {
          await sock.sendMessage(chatId, { text: 'Usage: `/modelcfg` remove <model_id>' });
        } catch (err) { /* ignore */ }
        return;
      }
      const [modelId] = subArgs;
      const result = deleteModel(modelId);
      if (result.success) {
        wsClient.sendReliable({ type: 'invalidate_default_model' });
        for (const affectedChatId of result.affectedChatIds) {
          wsClient.sendReliable({ type: 'clear_history', chatId: affectedChatId });
          wsClient.sendReliable({ type: 'invalidate_llm2_model', chatId: affectedChatId });
        }
      }
      try {
        await sock.sendMessage(chatId, { text: result.success ? `Model "${modelId}" deleted.` : `Model "${modelId}" not found.` });
      } catch (err) { /* ignore */ }
      break;
    }

    default:
      try {
        await sock.sendMessage(chatId, { text: 'Unknown subcommand. Use: list, add, edit, remove' });
      } catch (err) { /* ignore */ }
  }
}

export { handleModelcfg };
