import logger from './logger.js';
import wsClient from './wsClient.js';
import { startWhatsApp, sendOutgoing } from './waClient.js';
import config from './config.js';

async function bootstrap() {
  if (!config.wsEndpoint) {
    logger.error('Set LLM_WS_ENDPOINT in .env before running.');
    process.exit(1);
  }

  await startWhatsApp();

  wsClient.on('message', async (msg) => {
    if (!msg || !msg.type) return;
    if (msg.type === 'send_message') {
      try {
        await sendOutgoing(msg.payload || {});
        wsClient.send({ type: 'send_ack', payload: { requestId: msg.payload?.requestId } });
      } catch (err) {
        logger.error({ err }, 'failed sending outgoing message');
        wsClient.send({
          type: 'error',
          payload: { message: 'send_message failed', detail: err.message, requestId: msg.payload?.requestId },
        });
      }
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
