/**
 * wsClient.js — WebSocket client for Node ↔ Python communication.
 *
 * Two send modes:
 *   - send(message)     — best-effort, drops if disconnected. Used for
 *                         transient events like `incoming_message` (the next
 *                         burst will re-send newer state anyway).
 *   - sendReliable(msg)  — queues if disconnected, flushes on reconnect. Used
 *                         for state-sync events that must not be lost:
 *                         `whatsapp_status`, `clear_history`, `set_llm2_model`,
 *                         `invalidate_llm2_model`, `invalidate_default_model`.
 *
 * See ADR-4 in AGENTS.md for the rationale behind this split.
 */
import WebSocket from 'ws';
import { EventEmitter } from 'events';
import logger from './logger.js';
import config from './config.js';

class LLMWebSocket extends EventEmitter {
  /** Maximum number of queued reliable messages before dropping oldest. */
  static MAX_RELIABLE_QUEUE = 1000;

  constructor() {
    super();
    this.ws = null;
    this.reconnectTimer = null;
    this.reliableQueue = [];
  }

  connect() {
    if (!config.wsEndpoint) {
      logger.error('LLM_WS_ENDPOINT is required');
      return;
    }

    if (this.ws) {
      this.ws.removeAllListeners();
      try {
        this.ws.close();
      } catch (err) {
        logger.warn({ err }, 'close previous ws failed');
      }
    }

    const headers = {};
    if (config.wsToken) {
      headers.Authorization = `Bearer ${config.wsToken}`;
    }

    logger.info({ endpoint: config.wsEndpoint }, 'connecting to LLM websocket');
    this.ws = new WebSocket(config.wsEndpoint, { headers });

    this.ws.on('open', () => {
      logger.info('LLM websocket connected');
      this.send({
        type: 'hello',
        payload: {
          instanceId: config.instanceId,
          role: 'whatsapp-gateway',
        },
      });
      this.flushReliableQueue();
      this.emit('connected');
    });

    this.ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        this.emit('message', msg);
      } catch (err) {
        logger.warn({ err }, 'failed parsing ws message');
      }
    });

    this.ws.on('close', (code, reason) => {
      logger.warn({ code, reason: reason?.toString() }, 'LLM websocket closed, scheduling reconnect');
      this.emit('disconnected', { code, reason });
      this.scheduleReconnect();
    });

    this.ws.on('error', (err) => {
      logger.error({ err }, 'LLM websocket error');
    });
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, config.reconnectIntervalMs);
  }

  send(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      logger.debug('ws not ready, drop message');
      return;
    }
    this.sendRaw(message);
  }

  sendReliable(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.sendRaw(message);
      return;
    }
    this.reliableQueue.push(message);
    if (this.reliableQueue.length > LLMWebSocket.MAX_RELIABLE_QUEUE) {
      this.reliableQueue.shift();
      logger.warn({ queueSize: this.reliableQueue.length }, 'reliable ws queue overflow; oldest message dropped');
    }
    logger.debug({ queueSize: this.reliableQueue.length, type: message?.type }, 'ws not ready, queued reliable message');
  }

  flushReliableQueue() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (this.reliableQueue.length === 0) return;
    const queued = [...this.reliableQueue];
    this.reliableQueue.length = 0;
    for (const message of queued) {
      this.sendRaw(message);
    }
    logger.info({ count: queued.length }, 'flushed queued reliable ws messages');
  }

  sendRaw(message) {
    try {
      this.ws.send(JSON.stringify(message));
    } catch (err) {
      logger.error({ err }, 'failed sending ws message');
    }
  }
}

const wsClient = new LLMWebSocket();

export default wsClient;
