# 03 - Commands, Aliases, Permissions

## Canonical command list (Node)
- `help`
- `prompt`
- `reset`
- `permission`
- `mode`
- `trigger`
- `dashboard`
- `broadcast`
- `info`
- `debug`
- `join`
- `sticker`
- `model`
- `modelcfg`
- `setting`
- `group-status`
- `catch`

## Singular/plural aliases
The command parser normalizes aliases to the canonical form.

Examples:
- `/setting`, `/settings` → `setting`
- `/model`, `/models` → `model`
- `/prompt`, `/prompts` → `prompt`
- `/dashboard`, `/dashboards` → `dashboard`

Every canonical command has a singular/plural pair.

## Permission model

### General
- **Private chat**: Most commands are allowed.
- **Group chat**: Configuration commands require admin or owner role.

### Moderation level (`/permission`)
- `0`: Moderation forbidden (no tools available).
- `1`: Delete allowed.
- `2`: Delete + mute allowed.
- `3`: Delete + mute + kick allowed.

> **Note**: For levels > 0, the bot must have admin role in the group.

### Available LLM2 tools by permission level

| Tool | Level 0 | Level 1 | Level 2 | Level 3 |
|------|---------|---------|---------|---------|
| `reply_message` | ✅ | ✅ | ✅ | ✅ |
| `llm_express` | ✅ | ✅ | ✅ | ✅ |
| `delete_messages` | ❌ | ✅ | ✅ | ✅ |
| `mute_member` | ❌ | ❌ | ✅ | ✅ |
| `kick_members` | ❌ | ❌ | ❌ | ✅ |

## Command summary
- `/prompt [text|clear]` — Set/view/clear per-chat prompt override.
- `/reset` — Clear chat history in Python.
- `/mode [auto|prefix|hybrid]` — Set trigger mode for group chats.
- `/trigger [...]` — Set trigger prefixes for prefix/hybrid mode.
- `/model` — Select LLM2 model per chat (interactive menu).
- `/modelcfg ...` — CRUD model list (owner only).
- `/setting` — Interactive settings menu (mode/model/permission/misc).
- `/dashboard` — Display usage statistics.
- `/broadcast` — Send broadcast to all chats (owner only).
- `/info` — Show user/chat/group info.
- `/debug` — Send test interactive payload.
- `/join <invite link>` — Join a group via invite link.
- `/sticker` — Create sticker from image/video.