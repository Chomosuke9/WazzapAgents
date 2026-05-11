// Set env BEFORE any import of src/* (config.js reads env at import time,
// and src/logger.js also reads LOG_LEVEL at import time; LOG_LEVEL must be
// lowercase for pino to accept it). The static imports below (node builtins,
// `ws`) do not read these env vars, so static-importing them first is safe.
// `src/wsClient.js` is imported dynamically after env is set.
process.env.LOG_LEVEL = 'info';
process.env.LLM_WS_ENDPOINT = 'ws://127.0.0.1:1/ws';
process.env.WS_HEARTBEAT_INTERVAL_MS = '200';
process.env.WS_HEARTBEAT_TIMEOUT_MS = '80';
process.env.WS_RECONNECT_MS = '50';
process.env.WS_RECONNECT_MAX_MS = '500';
process.env.WS_RECONNECT_JITTER_RATIO = '0';

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { WebSocketServer } from 'ws';

const { LLMWebSocket, computeReconnectDelay } = await import('../../src/wsClient.js');
const { default: config } = await import('../../src/config.js');

async function startServer() {
  const server = new WebSocketServer({ port: 0, host: '127.0.0.1' });
  await once(server, 'listening');
  const port = server.address().port;
  config.wsEndpoint = `ws://127.0.0.1:${port}/ws`;
  return { server, port };
}

async function closeServer(server) {
  server.close();
  await once(server, 'close');
}

test('computeReconnectDelay grows exponentially and caps', () => {
  const opts = { baseMs: 1000, maxMs: 60000, jitterRatio: 0, rand: () => 0.5 };
  assert.equal(computeReconnectDelay({ ...opts, attempt: 1 }), 1000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 2 }), 2000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 3 }), 4000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 4 }), 8000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 5 }), 16000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 6 }), 32000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 7 }), 60000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 8 }), 60000);
  assert.equal(computeReconnectDelay({ ...opts, attempt: 0 }), 0);
  assert.equal(computeReconnectDelay({ ...opts, attempt: -1 }), 0);
  assert.equal(computeReconnectDelay({ ...opts, attempt: NaN }), 0);
});

test('computeReconnectDelay applies bounded jitter', () => {
  const base = { baseMs: 1000, maxMs: 60000, jitterRatio: 0.2, attempt: 1 };
  assert.equal(computeReconnectDelay({ ...base, rand: () => 0 }), 800);
  assert.equal(computeReconnectDelay({ ...base, rand: () => 1 }), 1200);
  assert.equal(computeReconnectDelay({ ...base, rand: () => 0.5 }), 1000);
  const extreme = computeReconnectDelay({
    baseMs: 10,
    maxMs: 60000,
    jitterRatio: 1,
    attempt: 1,
    rand: () => 0,
  });
  assert.equal(extreme, 0);
  for (let i = 0; i < 100; i++) {
    const r = Math.random();
    const d = computeReconnectDelay({
      baseMs: 1000,
      maxMs: 60000,
      jitterRatio: 0.5,
      attempt: 3,
      rand: () => r,
    });
    assert.ok(d >= 0, `delay must be >= 0, got ${d}`);
  }
});

test('connect sends hello handshake to server', { timeout: 5000 }, async () => {
  const { server } = await startServer();
  const client = new LLMWebSocket();
  const serverConn = new Promise((resolve) => {
    server.once('connection', (sock) => {
      sock.once('message', (data) => resolve({ sock, msg: JSON.parse(data.toString()) }));
    });
  });
  client.connect();
  const [result] = await Promise.all([serverConn, once(client, 'connected')]);
  assert.deepEqual(result.msg, {
    type: 'hello',
    payload: { instanceId: 'default', role: 'whatsapp-gateway' },
  });
  assert.equal(client.isConnected(), true);
  assert.equal(client.getAttempt(), 0);
  client.close();
  result.sock.close();
  await closeServer(server);
});

test('attempt counter resets on successful reconnect', { timeout: 10000 }, async () => {
  const { server, port } = await startServer();
  const client = new LLMWebSocket();
  let firstServerSock = null;
  server.on('connection', (sock) => {
    if (!firstServerSock) firstServerSock = sock;
  });
  client.connect();
  await once(client, 'connected');
  assert.equal(client.getAttempt(), 0);

  firstServerSock.close();
  await once(client, 'disconnected');
  assert.ok(
    client.getAttempt() >= 1,
    `expected attempt >= 1 after disconnect, got ${client.getAttempt()}`
  );

  await closeServer(server);
  await new Promise((r) => setTimeout(r, 30));

  const server2 = new WebSocketServer({ port, host: '127.0.0.1' });
  await once(server2, 'listening');

  await once(client, 'connected');
  assert.equal(client.getAttempt(), 0);

  client.close();
  await closeServer(server2);
});

test('sendReliable queues while disconnected and flushes in order on reconnect',
  { timeout: 5000 },
  async () => {
    const client = new LLMWebSocket();
    client.sendReliable({ type: 'a' });
    client.sendReliable({ type: 'b' });
    assert.equal(client.getReliableQueueSize(), 2);

    const { server } = await startServer();
    const received = [];
    let serverSock = null;
    server.on('connection', (sock) => {
      serverSock = sock;
      sock.on('message', (data) => received.push(JSON.parse(data.toString())));
    });

    client.connect();
    await once(client, 'connected');
    await new Promise((r) => setTimeout(r, 50));

    assert.equal(received.length, 3);
    assert.equal(received[0].type, 'hello');
    assert.equal(received[1].type, 'a');
    assert.equal(received[2].type, 'b');
    assert.equal(client.getReliableQueueSize(), 0);

    client.close();
    if (serverSock) serverSock.close();
    await closeServer(server);
  }
);

test('heartbeat terminates socket when server swallows pings', { timeout: 5000 }, async () => {
  const { server } = await startServer();
  const connSocks = [];
  server.on('connection', (sock) => {
    // Swallow the automatic pong so the client never observes a reply
    // to its ping. The heartbeat timeout should then fire and ws.terminate()
    // will force a close event on the client side.
    sock.pong = () => {};
    connSocks.push(sock);
  });

  const client = new LLMWebSocket();
  client.connect();
  await once(client, 'connected');

  const disconnected = once(client, 'disconnected');
  const safety = new Promise((_, reject) =>
    setTimeout(() => reject(new Error('heartbeat disconnect did not fire within 2s')), 2000)
  );
  await Promise.race([disconnected, safety]);

  client.close();
  for (const s of connSocks) {
    try { s.terminate(); } catch { /* ignore */ }
  }
  await closeServer(server);
});

test('close() cancels timers and drops queued messages', { timeout: 2000 }, async () => {
  const client = new LLMWebSocket();
  client.sendReliable({ type: 'x' });
  client.sendReliable({ type: 'y' });
  assert.equal(client.getReliableQueueSize(), 2);

  let disconnectedFired = false;
  client.on('disconnected', () => {
    disconnectedFired = true;
  });

  client.close();
  assert.equal(client.getReliableQueueSize(), 0);

  await new Promise((r) => setTimeout(r, 300));
  assert.equal(disconnectedFired, false);
  assert.equal(client.isConnected(), false);
});
