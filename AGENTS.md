# Repository Guidelines

## Project Structure & Module Organization
- `src/` – runtime code. Key files: `index.js` (bootstrap), `waClient.js` (WhatsApp via Baileys v7), `wsClient.js` (LLM WebSocket), `config.js`, `logger.js`, `utils.js`.
- `inbound/` – richer WhatsApp extraction helpers (mentions, interactive replies, location) used as reference patterns.
- `data/` – runtime artifacts: `auth/` (Baileys multi-file auth), `media/` (downloaded inbound media). Created automatically.
- `examples/` – example LLM WebSocket server (`llm_ws_echo.py`).
- `README.md` – protocol contract with the LLM (`incoming_message`, `send_message` payloads).

## Build, Test, and Development Commands
- Install deps: `pnpm install` (Node 18+; project is ESM).
- Run gateway: `pnpm dev` (same as `pnpm start`) – starts WhatsApp socket and connects to the LLM WebSocket defined by `LLM_WS_ENDPOINT`.
- Lint: `pnpm lint` (currently placeholder; exits with message).
- Tests: none wired in package.json; see “Testing Guidelines” for suggested approach.

## Coding Style & Naming Conventions
- Language: modern JavaScript (ESM, Node ≥18). Prefer async/await, top-level imports.
- Formatting: align with existing code (2-space indentation, single quotes, trailing commas avoided). No formatter configured; keep diffs minimal.
- Logging: use `logger` from `src/logger.js`; prefer structured context objects.
- Paths in payloads should stay workspace-relative (`data/media/...`) as shown in README.

## Testing Guidelines
- Framework not configured; Vitest is used in `inbound/access-control.pairing-history.test.ts` as a reference pattern. If adding tests, add `vitest` as a dev dependency and wire `pnpm test`.
- Name tests `*.test.ts|js`; group by module. Mock networked services (LLM WS, Baileys socket) to keep tests hermetic.
- For media handling, use small fixtures (<100 KB) to keep repo light.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, short prefix (e.g., `add`, `fix`, `refactor`). If changing protocol, mention `protocol:` in the subject.
- PRs should include: summary of changes, testing performed (`pnpm dev` smoke, unit tests if added), and notes on protocol or schema changes (e.g., new payload fields like `mentionedJids`).
- Include screenshots or logs only when UI/QR flow is affected.

## Security & Configuration Tips
- Never commit `data/auth`. `.env` should define `LLM_WS_ENDPOINT`, optional `LLM_WS_TOKEN`, `INSTANCE_ID`, and custom `DATA_DIR`/`MEDIA_DIR` if needed.
- Treat outbound WebSocket token and Baileys auth as secrets; rotate if leaked.
- When handling media, respect size limits to avoid OOM; defaults save directly to `data/media`.
