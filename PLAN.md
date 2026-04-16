# Plan: Fitur Ganti Model Per Chat dengan Interactive Message

## Status: IN PROGRESS

### Todo List
- [ ] Migrate `better-sqlite3` → `sql.js` (database broken)
- [ ] Fix interactive menu submenus (menu-in-menu)
- [ ] Fix bugs in commands (some commands still failing)
- [ ] Test all commands after fixes

---

## Ringkasan
Implementasi fitur untuk mengganti model LLM2 per chat menggunakan interactive message (list). Termasuk command `/model` untuk user dan `/modelcfg` untuk developer. Semua command sudah dimigrasikan ke JavaScript (Node.js).

---

## 1. Database (CRITICAL - Broken)

### Current Problem
- `better-sqlite3` native binding fails on server
- Error: `Could not locate the bindings file`

### Solution: Migrate to `sql.js`
- `sql.js` is pure JavaScript (WebAssembly) - no native compilation needed
- Works on all platforms without issues

### Files to Change
- `package.json` - replace `better-sqlite3` with `sql.js`
- `src/db.js` - rewrite to use sql.js API

### sql.js API Differences from better-sqlite3
```javascript
// better-sqlite3 (sync)
const db = new Database(path);
db.prepare('SELECT ...').get();
db.prepare('INSERT ...').run();

// sql.js (async init, then sync)
const SQL = await initSqlJs();
const db = new SQL.Database(data); // data is Uint8Array
db.run('SELECT ...'); // no prepare needed for simple queries
const result = db.exec('SELECT ...'); // returns array of {columns, values}
```

### sql.js Wrapper Pattern (Recommended)
```javascript
// Wrap sql.js to mimic better-sqlite3 sync API
class SyncDatabase {
  constructor(sqlDb) { this._db = sqlDb; }
  prepare(sql) {
    return {
      get: (...params) => { /* run and return first row */ },
      run: (...params) => { /* run and return */ },
      all: (...params) => { /* run and return all rows */ }
    };
  }
}
```

### Database File Location
- Path: `data/bot.db`
- Persist to disk on every write

---

## 2. Commands Status

### Working ✅
- `/model` - select model (interactive works)
- `/modelcfg` - manage models (interactive menu works)
- `/settings` - shows menu

### Broken ❌
- All database-dependent commands (need sql.js migration):
  - `/prompt` - get/set prompt
  - `/reset` - clear chat memory
  - `/permission` - set permission level
  - `/mode` - change mode
  - `/trigger` - toggle triggers
  - `/dashboard` - view stats

### Unknown ❓
- `/sticker` - needs testing
- `/broadcast`, `/info`, `/debug`, `/join` - need testing

---

## 3. Interactive Menu System (INCOMPLETE)

### Current Implementation
- Menu utama (`/modelcfg`) sudah ada dengan tombol:
  - List Models
  - Add Model
  - Edit Model
  - Remove Model
  - Set Default

### Missing: Submenus
Need to implement menu-in-menu pattern:

```
┌─────────────────────────────┐
│ 🛠️ Model Configuration     │
├─────────────────────────────┤
│  📋 List Models            │
│  ➕ Add Model              │
│  ✏️ Edit Model            │
│  🗑️ Remove Model           │
│  ⭐ Set Default            │
└─────────────────────────────┘
        ↓ User clicks "Edit Model"
┌─────────────────────────────┐
│ ✏️ Edit Model              │
├─────────────────────────────┤
│ Select a model to edit:    │
│  ○ GPT-4 Mini              │
│  ○ GPT-4o                  │
│  ○ Claude 3.5 Sonnet       │
└─────────────────────────────┘
        ↓ User selects model
┌─────────────────────────────┐
│ ✏️ Edit: GPT-4 Mini        │
├─────────────────────────────┤
│ Display Name: [GPT-4 Mini]  │
│ Description: [Fast model]  │
│ Sort Order: [1]            │
│ Active: [✓]                │
│                             │
│ [Save] [Cancel]             │
└─────────────────────────────┘
```

### Implementation Plan for Submenus

**Approach: State Machine**
- Store current menu state per chatId
- On button click, look up state and show appropriate submenu
- States: `main_menu`, `select_model_to_edit`, `edit_model`, `confirm_delete`, etc.

**Alternative: Flow-Based**
- Each button click sends user through a sequence
- Use `selectedId` to track progress: `edit_model`, `edit_model:select`, `edit_model:save`

### Button ID Format
```
modelcfg:list                    → show list
modelcfg:add                     → show add form
modelcfg:edit                    → show model selection
modelcfg:edit:gpt-4o-mini       → show edit form for gpt-4o-mini
modelcfg:remove                 → show model selection
modelcfg:remove:gpt-4o-mini     → show confirm delete
modelcfg:default                → show model selection
modelcfg:default:gpt-4o-mini    → set as default
modelcfg:back                   → go back to main menu
```

### For Edit Model Form
Need input capability - options:
1. Use reply-to-message pattern (bot asks user to reply with new value)
2. Use multiple single-select (predefined options)
3. Custom interactive message with input fields (if WA supports)

---

## 4. Command Reference

### SLASH_CMD_RE (src/wa/commands.js line 11)
```javascript
const SLASH_CMD_RE = /^\/(broadcast|prompt|reset|permission|info|mode|trigger|dashboard|help|debug|join|sticker|model|modelcfg|settings)\b\s*([\s\S]*)/i;
```

### Command Handlers (src/wa/commandHandler.js)

| Command | Function | Status |
|---------|----------|--------|
| `/prompt` | `handlePrompt()` | ❌ DB broken |
| `/reset` | `handleReset()` | ⚠️ Sends to Python |
| `/permission` | `handlePermission()` | ❌ DB broken |
| `/mode` | `handleMode()` | ❌ DB broken |
| `/trigger` | `handleTrigger()` | ❌ DB broken |
| `/dashboard` | `handleDashboard()` | ❌ DB broken |
| `/help` | `handleHelp()` | ✅ |
| `/broadcast` | `handleBroadcastCommand()` | ✅ |
| `/info` | `handleInfoCommand()` | ✅ |
| `/debug` | `handleDebugCommand()` | ✅ |
| `/join` | `handleJoinCommand()` | ✅ |
| `/sticker` | `handleSticker()` | ⚠️ Needs testing |
| `/model` | `handleModel()` | ✅ |
| `/modelcfg` | `handleModelcfg()` | ✅ |
| `/settings` | `handleSettings()` | ✅ |

---

## 5. Button Response Handler

**Location:** `src/wa/connection.js` - `handleButtonResponse()` function

**Triggered by:** `messages.upsert` listener (not `messages.update`)

**Supported Button IDs:**
```
model_select:<model_id>         → select a model
settings:model                  → settings → model menu
settings:prompt                 → settings → prompt info
settings:permission             → settings → permission info
modelcfg:list                   → show model list
modelcfg:add                    → show add model form
modelcfg:edit                   → show model selection for edit
modelcfg:edit:<model_id>        → show edit form
modelcfg:remove                 → show model selection for remove
modelcfg:remove:<model_id>      → show confirm delete
modelcfg:confirm_remove:<model_id> → delete model
modelcfg:default                → show model selection for default
modelcfg:default:<model_id>     → set as default
modelcfg:cancel_remove          → cancel delete
```

---

## 6. Interactive Message Functions

**Location:** `src/wa/interactive/`

### sendNativeFlow(sock, chatId, title, buttons, options)
- For native flow buttons (quick_reply, cta_url, etc.)

### sendList(sock, chatId, title, buttonText, sections, options)
- For list messages (WA native list)

### sendCarousel(sock, chatId, cards, options)
- For swipeable carousel cards

---

## 7. File Structure

### Python (LLM Only - No Changes Needed)
```
python/bridge/
├── db.py                    # Schema + model functions (working)
├── llm/llm2.py              # Uses per-chat model (working)
└── main.py                  # Command handling disabled
```

### Node.js (Commands - Needs Fixes)
```
src/
├── db.js                    # ❌ BROKEN - needs sql.js migration
├── config.js                # Configuration
├── logger.js                # Logging
├── wsClient.js              # WebSocket to LLM
├── mediaHandler.js          # Media download
├── messageParser.js         # Message parsing
├── identifiers.js           # JID handling
├── participants.js          # User roles
├── groupContext.js          # Group metadata
├── caches.js                # In-memory caches
└── wa/
    ├── connection.js         # ✅ Button handler + listeners
    ├── commandHandler.js    # ✅ All command handlers
    ├── commands.js          # ✅ Slash command parsing
    ├── stickerTool.js       # ✅ Sticker creation (sharp)
    ├── outbound.js          # Send messages
    ├── inbound.js           # Receive messages
    ├── actions.js           # Moderation actions
    ├── moderation.js        # Moderation logic
    ├── events.js            # Synthetic events
    ├── presence.js          # Read receipts
    ├── interactive/
    │   ├── index.js         # Export barrel
    │   ├── sendInteractive.js
    │   ├── sendButtons.js
    │   ├── sendCarousel.js
    │   └── README.md        # Documentation
    └── utils.js             # Helpers
```

---

## 8. Database Schema

### Tables
```sql
CREATE TABLE chat_settings (
  chat_id      TEXT PRIMARY KEY,
  prompt       TEXT,
  permission   INTEGER DEFAULT 0,
  mode         TEXT DEFAULT 'prefix',
  triggers     TEXT DEFAULT 'tag,reply,name',
  llm2_model   TEXT,
  updated_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE chat_stats (
  chat_id      TEXT,
  period_type  TEXT,
  period_key   TEXT,
  stat_key     TEXT,
  stat_value   INTEGER DEFAULT 0,
  PRIMARY KEY (chat_id, period_type, period_key, stat_key)
);

CREATE TABLE chat_user_stats (
  chat_id      TEXT,
  period_type  TEXT,
  period_key   TEXT,
  sender_ref   TEXT,
  sender_name  TEXT DEFAULT '',
  invoke_count INTEGER DEFAULT 0,
  PRIMARY KEY (chat_id, period_type, period_key, sender_ref)
);

CREATE TABLE llm_models (
  model_id     TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  description  TEXT,
  is_active    INTEGER DEFAULT 1,
  sort_order   INTEGER DEFAULT 0
);
```

---

## 9. Testing Checklist

### After sql.js Migration
- [ ] `/prompt` - set/show/clear prompt
- [ ] `/reset` - clear chat memory
- [ ] `/permission` - set permission level 0-3
- [ ] `/mode` - change mode (auto/prefix/hybrid)
- [ ] `/trigger` - toggle triggers (tag/reply/join/name)
- [ ] `/dashboard` - view stats

### Interactive Menus
- [ ] `/model` - select model from list
- [ ] `/modelcfg list` - show all models
- [ ] `/modelcfg add` - add new model
- [ ] `/modelcfg edit` - edit model (NEEDS SUBMENU)
- [ ] `/modelcfg remove` - delete model (with confirm)
- [ ] `/modelcfg setdefault` - set default model
- [ ] `/settings` - view settings menu

### Other Commands
- [ ] `/sticker` - create sticker from image
- [ ] `/broadcast` - broadcast message
- [ ] `/help` - show help
- [ ] `/info` - show bot info

### Button Clicks
- [ ] Select model from list
- [ ] Navigate submenus
- [ ] Confirm/cancel actions

---

## 10. Known Issues

1. **Database Broken** - `better-sqlite3` native binding fails
   - Fix: Migrate to `sql.js`

2. **Submenus Missing** - Interactive menus don't have submenu navigation
   - Fix: Implement state machine or flow-based navigation

3. **Some Commands Failing** - Database-dependent commands error
   - Fix: After sql.js migration

4. **Button Click Not Working** - Was going to Python chatbot
   - Status: ✅ FIXED - now handled in messages.upsert

5. **senderIsOwner Not Defined** - Context destructuring missing
   - Status: ✅ FIXED

---

## 11. Environment Variables

### For Testing (if needed)
```bash
LLM_WS_ENDPOINT=ws://localhost:8080/ws
BOT_OWNER_JIDS=6281234567890
LOG_LEVEL=debug
```

---

## 12. Notes

- Interactive message uses `sendNativeFlow` with `single_select`
- Row ID format: `prefix:suffix` for easy parsing
- `model_id` for backend, `display_name` and `description` for frontend
- Default model = model with smallest `sort_order`
- Database path: `data/bot.db`

---

## 13. Git History

Recent commits:
```
a600cb1 update PLAN.md to reflect current progress
472a821 fix: button responses handled in messages.upsert listener
e81f29c fix: handle listResponseMessage for single_select
faa2de0 fix: add senderIsOwner to context destructuring
109be66 fix import: withTimeout from wa/utils.js not src/utils.js
cd134bd fix sticker: use saveMedia instead of downloadContentFromMessage
8e7fa21 migrate /sticker command to JavaScript
519b3d5 add per-chat LLM2 model selection with interactive messages
```

---

## 14. Next Session Tasks

1. **Priority 1:** Migrate `src/db.js` to `sql.js`
   - Install sql.js package
   - Rewrite database wrapper
   - Test all DB operations

2. **Priority 2:** Implement submenu navigation
   - Add state tracking for menu navigation
   - Implement edit model flow
   - Test all submenus

3. **Priority 3:** Test and fix remaining issues
   - Test all commands
   - Fix any remaining bugs
