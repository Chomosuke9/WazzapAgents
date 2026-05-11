/**
 * wsClient.js — WebSocket client for Node <-> Python communication.
 *
 * Two send modes:
 *   - send(message)     — best-effort, drops if disconnected. Used for
 *                         transient events like `incoming_message` (the next
 *                         burst will re-send newer state anyway).
 *   - sendReliable(msg)  — queues if disconnected, flushes on reconnect. Used
 *                         for state-sync events that must not be lost:
 *                         `whatsapp_status`, `clear_history`, `set_llm2_model`,
 *                         `invalidate_llm2_model`, `invalidate_default_model`,
 *                         `invalidate_chat_settings`, `set_subagent_enabled`.
 *
 * Reconnect strategy:
 *   Exponential backoff with full-jitter. Base delay is `WS_RECONNECT_MS`
 *   (default 5000), capped at `WS_RECONNECT_MAX_MS` (default 60000), with
 *   +/- `WS_RECONNECT_JITTER_RATIO` (default 0.2 = 20%) applied symmetrically
 *   so retries desynchronise across instances and avoid hammering a
 *   struggling server. The pure function `computeReconnectDelay` is exported
 *   for deterministic unit testing.
 *
 * Heartbeat:
 *   Once connected, Node sends a WS ping every `WS_HEARTBEAT_INTERVAL_MS`
 *   (default 20000). If the matching pong does not arrive within
 *   `WS_HEARTBEAT_TIMEOUT_MS` (default 20000) the socket is deemed half-open
 *   and `ws.terminate()` is invoked, which triggers the normal `close`
 *   reconnect path. This is symmetrical with the Python side, which runs
 *   `websockets.serve(..., ping_interval=20, ping_timeout=20)` in
 *   `python/bridge/main.py` and would detect a dead Node the same way.
 *
 * Graceful shutdown:
 *   Call `close()` on the instance to cancel the reconnect timer, clear
 *   heartbeat timers, empty the reliable queue, remove listeners, and close
 *   the socket. `src/index.js` wires this into its SIGINT/SIGTERM handler.
 *
 * The class is exported as a named export so tests can construct fresh
 * instances; the module's default export is a process-wide singleton used
 * by `src/index.js` and every `src/wa/command/*.js` caller.
 */
import WebSocket from 'ws';
import { EventEmitter } from 'events';
import logger from './logger.js';
import config from './config.js';

/**
 * Pure function computing the next reconnect delay (in ms) for a given
 * attempt number using exponential backoff with full-jitter.
 *
 * Intended to be called for attempt >= 1 (attempt 0 is the initial connect
 * and should not be scheduled). For attempt <= 0 this returns 0.
 *
 * @param {object} opts
 * @param {number} opts.attempt      Reconnect attempt number (1-indexed).
 * @param {number} opts.baseMs       Base delay for attempt 1 (e.g. 5000).
 * @param {number} opts.maxMs        Cap for the exponential growth.
 * @param {number} opts.jitterRatio  [0,1] fraction of delay used for symmetric jitter.
 * @param {() => number} [opts.rand] Random source in [0,1). Injectable for tests.
 * @returns {number} Delay in ms, rounded and floored to 0.
 */
export function computeReconnectDelay({ attempt, baseMs, maxMs, jitterRatio, rand = Math.random }) {
  if (!Number.isFinite(attempt) || attempt < 1) return 0;
  const exp = baseMs * Math.pow(2, attempt - 1);
  const delay = Math.min(maxMs, exp);
  const jitter = delay * jitterRatio * (rand() * 2 - 1);
  return Math.max(0, Math.round(delay + jitter));
}

export class LLMWebSocket extends EventEmitter {
  /** Maximum number of queued reliable messages before dropping oldest. */
  static MAX_RELIABLE_QUEUE = 1000;

  constructor() {
    super();
    this.ws = null;
    this.reconnectTimer = null;
    this.reliableQueue = [];
    this.attempt = 0;
    this.heartbeatInterval = null;
    this.pendingPongTimeout = null;
  }

  connect() {
    if (!config.wsEndpoint) {
      logger.error('LLM_WS_ENDPOINT is required');
      return;
    }

    // Clear any prior heartbeat timers before opening a new socket so we
    // never leak timers across socket lifetimes.
    this._clearHeartbeat();

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

    logger.info({ endpoint: config.wsEndpoint, attempt: this.attempt }, 'connecting to LLM websocket');
    this.ws = new WebSocket(config.wsEndpoint, { headers });

    this.ws.on('open', () => {
      logger.info('LLM websocket connected');
      this.attempt = 0;
      this._startHeartbeat();
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

    this.ws.on('pong', () => {
      if (this.pendingPongTimeout) {
        clearTimeout(this.pendingPongTimeout);
        this.pendingPongTimeout = null;
      }
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
      this._clearHeartbeat();
      this.emit('disconnected', { code, reason });
      this.scheduleReconnect();
    });

    this.ws.on('error', (err) => {
      logger.error({ err }, 'LLM websocket error');
    });
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.attempt += 1;
    const delay = computeReconnectDelay({
      attempt: this.attempt,
      baseMs: config.reconnectIntervalMs,
      maxMs: config.wsReconnectMaxMs,
      jitterRatio: config.wsReconnectJitterRatio,
    });
    logger.info({ attempt: this.attempt, delayMs: delay }, 'scheduling ws reconnect');
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  _startHeartbeat() {
    this._clearHeartbeat();
    this.heartbeatInterval = setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      try {
        this.ws.ping();
      } catch (err) {
        logger.warn({ err }, 'ws ping failed');
        return;
      }
      if (this.pendingPongTimeout) {
        clearTimeout(this.pendingPongTimeout);
      }
      this.pendingPongTimeout = setTimeout(() => {
        this.pendingPongTimeout = null;
        logger.warn({ timeoutMs: config.wsHeartbeatTimeoutMs }, 'ws pong timeout, terminating socket');
        try {
          this.ws?.terminate();
        } catch (err) {
          logger.warn({ err }, 'ws terminate failed');
        }
      }, config.wsHeartbeatTimeoutMs);
    }, config.wsHeartbeatIntervalMs);
  }

  _clearHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.pendingPongTimeout) {
      clearTimeout(this.pendingPongTimeout);
      this.pendingPongTimeout = null;
    }
  }

  close() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this._clearHeartbeat();
    if (this.reliableQueue.length > 0) {
      logger.info({ dropped: this.reliableQueue.length }, 'ws close dropping queued reliable messages');
      this.reliableQueue.length = 0;
    }
    this.removeAllListeners();
    if (this.ws) {
      this.ws.removeAllListeners();
      try {
        this.ws.close();
      } catch (err) {
        logger.warn({ err }, 'ws close failed');
      }
      this.ws = null;
    }
  }

  isConnected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getAttempt() {
    return this.attempt;
  }

  getReliableQueueSize() {
    return this.reliableQueue.length;
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
