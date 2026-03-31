# WazzapAgents Refactoring Plan

## Context

WazzapAgents has two "god files" that have grown too large and handle too many responsibilities:

- **python/bridge/main.py** (2,401 lines, 61 functions) — WebSocket handler, message processing, batching, action parsing, outbound messaging, all in one file
- **src/waClient.js** (1,352 lines, 26 functions) — WhatsApp connection, inbound/outbound messaging, moderation, commands, events, all in one file

Secondary issues: python/bridge/llm1.py (1,276 lines) is also large, duplicate requirements.txt, and missing tooling (linter/formatter).

The goal is to split these into focused, cohesive modules while preserving all existing behavior and keeping tests passing.

---

## Phase 1: Split python/bridge/main.py (2,401 -> ~800 lines)

Create 6 new modules. main.py keeps orchestration (~800 lines).

### Step 1.1: Create python/bridge/message_processing.py (~450 lines)

Move these functions (current line numbers in main.py):

- `_append_history()` (114), `_normalize_preview_text()` (121)
- `_quoted_from_payload()` (130), `_infer_quoted_media()` (137), `_quoted_sender()` (154), `_quoted_preview()` (161)
- `_mentioned_participant_rows()` (192), `_mention_label()` (232), `_mention_labels()` (248), `_mention_number_candidates()` (269), `_replace_mentions_in_text()` (283), `_bot_jid_from_rows()` (308), `_ensure_bot_token_in_text()` (317)
- `_payload_text_with_mentions()` (333), `_normalize_context_msg_id()` (360), `_is_system_payload()` (373), `_display_context_msg_id_from_payload()` (387), `_payload_to_message()` (393)
- `_build_burst_current()` (426), `_collect_context_ids()` (476)
- `_reply_signature()` (501), `_merge_fromme_echo_into_provisional()` (507), `_append_or_merge_history_payload()` (555)
- `_extract_send_ack_context_msg_id()` (565), `_hydrate_provisional_context_id_from_ack()` (590)
- `_clean_text()` (614), `_make_request_id()` (610), `_is_context_only_payload()` (620)

**Constants to move:**
- `CONTEXT_MSG_ID_RE`, `SENDER_REF_RE`, `EMPTY_TARGET_TOKENS`, `REQUEST_COUNTER`, `SYSTEM_CONTEXT_TOKEN`, `MENTION_SUMMARY_MAX_ITEMS`

**Imports needed:**
- `re`, `time`, `itertools.count`
- `bridge.history.WhatsAppMessage`, `bridge.history.assistant_name`, `bridge.history.assistant_sender_ref`
- `bridge.config` constants
- `bridge.log.setup_logging`

### Step 1.2: Create python/bridge/payload_filtering.py (~150 lines)

Move these functions:

- `_chat_state_from_payload()` (485), `_payload_timestamp_ms()` (492)
- `_payload_triggers_llm1()` (624), `_message_matches_prefix()` (633)
- `_payload_has_meaningful_content()` (652), `_payload_is_human()` (677)
- `_is_provisional_assistant_echo()` (683), `_payload_has_explicit_join_event()` (770)

**Imports needed:**
- `bridge.history.assistant_sender_ref`
- `bridge.db.get_triggers`
- `bridge.log`

### Step 1.3: Create python/bridge/llm1_metadata.py (~200 lines)

Move these functions:

- `_messages_since_last_assistant()` (714), `_assistant_replies_in_recent()` (731)
- `_llm1_history_limit_for_metadata()` (748), `_is_group_join_action()` (761)
- `_bot_name_mentioned_in_text()` (783), `_bot_name_mentioned_in_payloads()` (796)
- `_build_llm1_context_metadata()` (811)
- `_resolve_group_prompt_context()` (889)

**Imports needed:**
- `os`, `re`
- `bridge.history`
- `bridge.db.get_prompt`
- `bridge.config`
- `bridge.log`

**Note:** `_build_llm1_context_metadata()` calls `_is_provisional_assistant_echo()` and `_payload_is_human()` from payload_filtering — cross-module import needed.

### Step 1.4: Create python/bridge/moderation.py (~70 lines)

Move these functions:

- `_moderation_permissions()` (898), `_enforce_moderation_permissions()` (917)
- `_merge_payload_attachments()` (936)

**Imports needed:**
- `bridge.db.get_permission`
- `bridge.db.permission_allows_kick`
- `bridge.db.permission_allows_delete`
- `bridge.log`

### Step 1.5: Create python/bridge/action_parsing.py (~350 lines)

Move these functions:

- `_infer_media()` (2039), `_extract_reply_text()` (2048)
- `_is_empty_target_token()` (2057), `_unwrap_angle_group()` (2063)
- `_resolve_reply_target()` (2068), `_resolve_delete_target()` (2092)
- `_parse_delete_targets()` (2107), `_parse_kick_targets()` (2131), `_parse_react_context_ids()` (2187)
- `_extract_actions()` (2217) — main state machine parser

**Constants to move:**
- `ACTION_LINE_RE`

**Imports needed:**
- `re`
- `bridge.log`
- `bridge.stickers.resolve_sticker`
- Also needs `_normalize_context_msg_id` and `EMPTY_TARGET_TOKENS` from message_processing.

### Step 1.6: Create python/bridge/gateway_actions.py (~230 lines)

Move these async functions:

- `send_message()` (1810), `send_delete_message()` (1844), `send_kick_member()` (1877), `send_react_message()` (1915), `send_sticker()` (1951)
- `send_mark_read()` (1982), `send_typing()` (2000)
- `typing_indicator()` context manager (2012) + nested `_keep_alive()` (2020)

**Imports needed:**
- `asyncio`
- `contextlib`
- `json`
- `time`
- `bridge.log`
- `bridge.config`

### Step 1.7: Update python/bridge/main.py (orchestration, ~800 lines)

What remains:

- `PendingChat` dataclass (103)
- `handle_socket()` (960) + nested: `_track_task()`, `_is_duplicate_reply()`
- `process_message_batch()` (999) + `_log_slow_batch()` (1096)
- `flush_pending()` (1652)
- `main()` (2385)

Update imports to pull from new modules. Add re-exports for backward compatibility:

```python
# Backward compat re-exports (used by tests)
from .message_processing import _quoted_preview, _build_burst_current  # noqa: F401
from .llm1_metadata import _build_llm1_context_metadata  # noqa: F401
```

### Step 1.8: Update tests

**File:** python/tests/test_llm_context_serialization.py

Current imports from bridge.main:

```python
from bridge.main import (
    _build_burst_current,
    _build_llm1_context_metadata,
    _quoted_preview,
)
```

Update to import from new modules:

```python
from bridge.message_processing import _build_burst_current, _quoted_preview
from bridge.llm1_metadata import _build_llm1_context_metadata
```

### Step 1.9: Run tests to verify

```bash
python -m pytest python/tests/
```

---

## Phase 2: Split src/waClient.js (1,352 -> ~30 line barrel)

Create 7 new modules. waClient.js becomes a barrel re-export file.

### Step 2.1: Create src/waUtils.js (~40 lines)

Move from waClient.js:

- `runWithConcurrency()` (68-86)
- `withTimeout()` (89-105)
- `escapeRegex()` (108-110)

No internal dependencies. Pure utility functions.

### Step 2.2: Create src/waConnection.js (~140 lines)

Move:

- Module-level `let sock` (65)
- `printQrInTerminal()` (114-128)
- `startWhatsApp()` (874-992)

Export `getSock()` getter function for other modules to access the socket.

Export `startWhatsApp`.

This module registers event handlers that call functions from waInbound and waEvents. Those will be passed as callbacks or imported.

**Key design decision:** `startWhatsApp()` registers event listeners that call `handleIncomingMessage()`, `handleGroupParticipantsUpdate()`, etc. These will be imported from their new modules.

### Step 2.3: Create src/waEvents.js (~160 lines)

Move:

- `makeEventMessageId()` (178-181)
- `emitGroupJoinContextEvent()` (184-287)
- `emitBotActionContextEvent()` (290-335)

**Imports:** wsClient, groupContext, identifiers, participants, config, logger, getSock from waConnection

### Step 2.4: Create src/waInbound.js (~250 lines)

Move:

- `resolveParticipantLabel()` (132-146)
- `buildMentionedParticipants()` (149-175)
- `handleGroupParticipantsUpdate()` (338-358)
- `handleIncomingMessage()` (361-557)

**Imports:** messageParser, mediaHandler, identifiers, participants, groupContext, wsClient, logger, config, caches, waEvents, waCommands, getSock from waConnection, withTimeout from waUtils

### Step 2.5: Create src/waOutbound.js (~180 lines)

Move:

- `renderOutboundMentions()` (994-1071)
- `sendOutgoing()` (1073-1167)

**Imports:** groupContext, identifiers, mediaHandler, logger, getSock from waConnection, escapeRegex from waUtils

### Step 2.6: Create src/waActions.js (~80 lines)

Move:

- `actionError()` (559-564)
- `reactToMessage()` (566-605)
- `deleteMessageByContextId()` (607-638)

**Imports:** identifiers, logger, getSock from waConnection, emitBotActionContextEvent from waEvents

### Step 2.7: Create src/waModeration.js (~240 lines)

Move:

- `parseParticipantUpdateStatus()` (640-644)
- `maybeEmitKickAnchorReplies()` (646-671)
- `kickMembers()` (673-872)

**Imports:** groupContext, identifiers, participants, logger, getSock from waConnection, emitBotActionContextEvent from waEvents

### Step 2.8: Create src/waCommands.js (~130 lines)

Move:

- `parseSlashCommand()` (1175-1183)
- `handleBroadcastCommand()` (1186-1266)
- `truncateText()` (1268-1274)
- `handleInfoCommand()` (1276-1313)

**Imports:** logger, participants, caches, getSock from waConnection

### Step 2.9: Create src/waPresence.js (~25 lines)

Move:

- `markChatRead()` (1319-1331)
- `sendPresence()` (1333-1341)

**Imports:** logger, getSock from waConnection

### Step 2.10: Convert src/waClient.js to barrel re-export (~30 lines)

```javascript
// Barrel re-export for backward compatibility
export { withTimeout } from './waUtils.js'
export { startWhatsApp } from './waConnection.js'
export { sendOutgoing } from './waOutbound.js'
export { reactToMessage, deleteMessageByContextId } from './waActions.js'
export { kickMembers } from './waModeration.js'
export { markChatRead, sendPresence } from './waPresence.js'
```

src/index.js continues to import from ./waClient.js — no changes needed.

---

## Phase 3: Split python/bridge/llm1.py (1,276 -> ~600 lines)

### Step 3.1: Create python/bridge/llm1_schemas.py (~115 lines)

Move:

- `LLM1_SCHEMA` (128-157), `LLM1_TOOL` (159-167)
- `LLM1_EXPRESS_SCHEMA` (169-213), `LLM1_REACT_TOOL` (215-227), `LLM1_TOOLS` (229)
- `LLM1Decision` pydantic model (232-239)

### Step 3.2: Create python/bridge/llm1_prompt.py (~280 lines)

Move:

- `_render_prompt_override()` (242-254), `_group_description_block()` (257-261)
- `_format_current_window()` (264-269), `build_llm1_prompt()` (272-397)
- `_metadata_block()` (705-818)

### Step 3.3: Create python/bridge/llm1_client.py (~150 lines)

Move:

- Config parsing functions (40-85): `_llm1_history_limit()`, `_llm1_message_max_chars()`, `_llm1_timeout()`, `_llm1_sdk_max_retries()`, `_llm1_temperature()`, `_llm1_max_tokens()`, `_llm1_reasoning_effort()`
- `_clean_env()` (400-404), `_endpoint_base_url()` (407-414), `_chat_base_url()` (417-418), `_llm1_targets()` (421-464)
- `LLM1Target` dataclass (32-37)
- `get_llm1()` (467-495)

### Step 3.4: Update python/bridge/llm1.py (remains, ~600 lines)

Keeps:

- Text truncation functions (88-125)
- Response parsing functions (498-702)
- Error handling functions (513-560)
- Logging/context functions (563-615, 821-829)
- `call_llm1()` main function (832-1276)

Update imports. Add re-exports for test compatibility:

```python
from .llm1_prompt import _metadata_block, build_llm1_prompt  # noqa: F401
```

### Step 3.5: Update tests

Current imports from bridge.llm1:

```python
from bridge.llm1 import _metadata_block, build_llm1_prompt
```

Update to:

```python
from bridge.llm1_prompt import _metadata_block, build_llm1_prompt
```

---

## Phase 4: Cleanup

### Step 4.1: Remove duplicate requirements.txt

Delete python/requirements.txt (identical to root requirements.txt).

### Step 4.2: Move existing utilities to src/waUtils.js

The current src/utils.js has streamToBuffer and streamToFile. Keep it separate — waUtils.js is for waClient-extracted utilities.

### Step 4.3: Final test run

```bash
python -m pytest python/tests/
```

Manually verify Node.js gateway starts: `pnpm dev` (if possible).

---

## Execution Order

1. **Phase 1 first** (Python bridge) — most impactful, has tests to validate
2. **Phase 2 second** (Node.js gateway) — no automated tests, needs manual verification
3. **Phase 3 third** (llm1.py split) — lower priority, has test coverage
4. **Phase 4 last** (cleanup)

Commit after each phase to allow easy rollback.

---

## Files Modified/Created Summary

### New files (13):

- python/bridge/message_processing.py
- python/bridge/payload_filtering.py
- python/bridge/llm1_metadata.py
- python/bridge/moderation.py
- python/bridge/action_parsing.py
- python/bridge/gateway_actions.py
- python/bridge/llm1_schemas.py
- python/bridge/llm1_prompt.py
- python/bridge/llm1_client.py
- src/waUtils.js
- src/waConnection.js
- src/waEvents.js
- src/waInbound.js
- src/waOutbound.js
- src/waActions.js
- src/waModeration.js
- src/waCommands.js
- src/waPresence.js

### Modified files (5):

- python/bridge/main.py — stripped to ~800 lines orchestration + re-exports
- python/bridge/llm1.py — stripped to ~600 lines + re-exports
- python/tests/test_llm_context_serialization.py — update imports
- src/waClient.js — replaced with ~30 line barrel re-export
- (python/requirements.txt — deleted)

### Deleted files (1):

- python/requirements.txt (duplicate)

---

## Verification

```bash
python -m pytest python/tests/
```

```bash
python -c "from python.bridge.main import handle_socket, main"
```

```bash
node -e "import('./src/waClient.js').then(m => console.log(Object.keys(m)))"
```

Smoke test: If environment is available, run `pnpm dev` and `python -m python.bridge.main` to verify startup
