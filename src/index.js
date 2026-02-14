import logger from './logger.js';
import wsClient from './wsClient.js';
import {
  startWhatsApp,
  sendOutgoing,
  deleteMessageByContextId,
  kickMembers,
} from './waClient.js';
import config from './config.js';

function actionErrorCode(err) {
  if (!err || typeof err !== 'object') return 'send_failed';
  if (typeof err.code === 'string' && err.code.trim()) return err.code;
  return 'send_failed';
}

function actionErrorDetail(err) {
  if (!err || typeof err !== 'object') return 'unknown error';
  if (typeof err.detail === 'string' && err.detail.trim()) return err.detail;
  if (typeof err.message === 'string' && err.message.trim()) return err.message;
  return 'unknown error';
}

function deriveKickFailure(result) {
  const rows = Array.isArray(result?.results) ? result.results : [];
  const failures = rows.filter((row) => !row?.ok);
  if (failures.length === 0) {
    return { code: 'send_failed', detail: 'no targets were kicked' };
  }

  const codes = failures
    .map((row) => (typeof row?.error === 'string' ? row.error : null))
    .filter(Boolean);
  const priority = ['permission_denied', 'send_failed', 'not_found', 'invalid_target'];
  const code = priority.find((candidate) => codes.includes(candidate)) || codes[0] || 'send_failed';

  const detail = failures.find((row) => typeof row?.detail === 'string' && row.detail.trim())?.detail
    || 'no targets were kicked';
  return { code, detail };
}

function emitActionAck({
  requestId,
  action,
  ok,
  detail,
  result = null,
  code = null,
}) {
  const payload = {
    requestId,
    action,
    ok: Boolean(ok),
    detail: detail || (ok ? 'ok' : 'failed'),
  };
  if (result && typeof result === 'object') payload.result = result;
  if (code) payload.code = code;
  wsClient.send({ type: 'action_ack', payload });
  if (action === 'send_message' && ok) {
    wsClient.send({ type: 'send_ack', payload: { requestId } });
  }
}

function emitActionError({
  requestId,
  action,
  err,
}) {
  const code = actionErrorCode(err);
  const detail = actionErrorDetail(err);
  emitActionAck({ requestId, action, ok: false, detail, code });
  wsClient.send({
    type: 'error',
    payload: {
      message: `${action} failed`,
      detail,
      code,
      requestId,
      action,
    },
  });
}

async function dispatchCommand(msg) {
  const payload = msg?.payload || {};
  const requestId = payload.requestId;
  const type = msg?.type;

  if (type === 'send_message') {
    const result = await sendOutgoing(payload);
    emitActionAck({ requestId, action: 'send_message', ok: true, detail: 'sent', result });
    return;
  }

  if (type === 'delete_message') {
    const result = await deleteMessageByContextId(payload);
    emitActionAck({ requestId, action: 'delete_message', ok: true, detail: 'deleted', result });
    return;
  }

  if (type === 'kick_member') {
    const result = await kickMembers(payload);
    const ok = Boolean(result?.ok);
    if (ok) {
      emitActionAck({ requestId, action: 'kick_member', ok: true, detail: 'kick applied', result });
      return;
    }

    const failure = deriveKickFailure(result);
    emitActionAck({
      requestId,
      action: 'kick_member',
      ok: false,
      detail: failure.detail,
      result,
      code: failure.code,
    });
    wsClient.send({
      type: 'error',
      payload: {
        message: 'kick_member failed',
        detail: failure.detail,
        code: failure.code,
        requestId,
        action: 'kick_member',
      },
    });
    return;
  }

  if (type && type !== 'hello') {
    wsClient.send({
      type: 'error',
      payload: {
        message: `unsupported command: ${type}`,
        detail: 'command not implemented by gateway',
        code: 'invalid_target',
        requestId,
        action: type,
      },
    });
  }
}

async function bootstrap() {
  if (!config.wsEndpoint) {
    logger.error('Set LLM_WS_ENDPOINT in .env before running.');
    process.exit(1);
  }

  await startWhatsApp();

  wsClient.on('message', async (msg) => {
    if (!msg || !msg.type) return;
    try {
      await dispatchCommand(msg);
    } catch (err) {
      const action = msg.type;
      logger.error({ err, action }, 'failed handling ws command');
      emitActionError({
        requestId: msg?.payload?.requestId,
        action,
        err,
      });
    }
  });

  wsClient.connect();
}

bootstrap().catch((err) => {
  logger.error({ err }, 'bootstrap failed');
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, 'unhandledRejection');
});

process.on('uncaughtException', (err) => {
  logger.error({ err }, 'uncaughtException');
});
