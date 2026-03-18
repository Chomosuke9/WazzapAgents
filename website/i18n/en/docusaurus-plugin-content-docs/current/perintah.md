---
sidebar_position: 3
---

# Command List

All commands start with `/` (forward slash). In groups, most commands can only be used by **admins**. In private chats, all users can use all commands.

## Summary

| Command | Function | Who Can Use |
|---------|----------|-------------|
| `/prompt` | View active prompt | Admin (group), Anyone (private) |
| `/prompt <text>` | Set new prompt | Admin (group), Anyone (private) |
| `/prompt -` | Delete prompt | Admin (group), Anyone (private) |
| `/reset` | Reset bot memory | Admin (group), Anyone (private) |
| `/permission` | Check moderation permission level | Group admin |
| `/permission 0-3` | Set moderation permission level | Group admin |
| `/info` | User & group info | Everyone |
| `/broadcast <message>` | Send to all groups | Bot owner only |

---

## `/prompt`

Sets the **personality, role, and rules** for the bot in this chat.

### View current prompt
```
/prompt
```

### Set a new prompt
```
/prompt <your rules text>
```
**Limit:** maximum 4000 characters.

### Delete prompt (return to default)
```
/prompt -
```
or `/prompt clear` or `/prompt reset`

:::info
Prompts apply **per chat/group**. Settings in group A do not affect group B.
:::

---

## `/reset`

Clears the bot's **memory/conversation history** for this chat.

```
/reset
```

Use when:
- The bot has gone "off track" and its answers don't make sense
- You want to start a fresh conversation from scratch
- After making major prompt changes

---

## `/info`

Displays user and group information.

```
/info
```

Shows: name, WhatsApp number, group role, group name, member count, bot admin status, and group description.

**Can be used by everyone**, no admin required.

---

## `/broadcast`

Sends a message to all groups where the bot is registered.

```
/broadcast <message>
```

Or **reply** to a specific message with `/broadcast` to forward that message to all groups.

:::warning
Can only be used by the **bot owner**. Regular users cannot use this command.
:::
