# LLM Architecture Docs (WazzapAgents)

Architecture documentation for **LLM / agent developers** who need to understand the runtime flow, module responsibilities, and data contracts.

> **Start with [AGENTS.md](../../AGENTS.md)** for full project context, terminology, and ADRs.
> These docs dive deeper into specific subsystems.

## Reading order
1. `00-overview.md` — End-to-end system overview
2. `01-runtime-flow.md` — Runtime flow per event type
3. `02-modules-map.md` — Module map and responsibilities
4. `03-commands-and-permissions.md` — Commands, roles, and permission model
5. `04-protocol-and-actions.md` — WebSocket contract between Node and Python
6. `05-state-data-and-db.md` — State, caching, and SQLite storage

## Key principles
- **Node.js gateway** handles WhatsApp connection, interactive UI, slash command parsing, and WS relay.
- **Python bridge** handles message batching, LLM routing (LLM1/LLM2), moderation actions, and stats writing.
- **Dashboard** reads from Node (DB query), while stats are written by Python (periodic flush).
- **Critical WS events** are sent via `sendReliable()` to survive reconnects (see ADR-4 in AGENTS.md).