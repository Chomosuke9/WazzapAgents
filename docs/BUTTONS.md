# WhatsApp Interactive Buttons Guide

## Button Types

### 1. quick_reply
Sends a text message when clicked. Used for simple actions.

```javascript
{
  name: 'quick_reply',
  buttonParamsJson: JSON.stringify({
    display_text: 'Get Prompt',
    id: '/prompt'
  })
}
```

**Important:** The `id` field can be:
- A slash command (e.g., `/prompt`, `/model gpt-4o`) - will be parsed as a command
- Any other string - will be returned as `selectedId` in button response

### 2. single_select (Dropdown Menu)
Opens a dropdown with sections and rows.

```javascript
{
  name: 'single_select',
  buttonParamsJson: JSON.stringify({
    title: 'Change Model',
    sections: [
      {
        title: 'Select Model',
        rows: [
          {
            id: 'model_select:gpt-4o',
            title: 'GPT-4o',
            description: 'Fast and capable'
          },
          {
            id: 'model_select:gpt-4o-mini',
            title: 'GPT-4o Mini',
            description: 'Lightweight model'
          }
        ]
      }
    ]
  })
}
```

**CRITICAL:** Each row MUST have `id` field (NOT `rowId`). The `id` value is what gets returned when user clicks that row.

## Sending Multiple Buttons

WhatsApp NativeFlow supports multiple buttons in one message:

```javascript
const buttons = [
  {
    name: 'quick_reply',
    buttonParamsJson: JSON.stringify({
      display_text: 'Get Prompt',
      id: '/prompt'
    })
  },
  {
    name: 'single_select',
    buttonParamsJson: JSON.stringify({
      title: 'Change Model',
      sections: [...]
    })
  },
  {
    name: 'single_select',
    buttonParamsJson: JSON.stringify({
      title: 'Set Permission',
      sections: [...]
    })
  }
];

await sendNativeFlow(sock, chatId, 'Chat Settings', buttons, { footer: 'Click a button' });
```

## Processing Button Clicks

When a user clicks a button, the message contains:

### For quick_reply:
```javascript
msg.message.buttonsResponseMessage.selectedButtonId
// Example: "/prompt" or "my_custom_id"
```

### For single_select:
```javascript
msg.message.listResponseMessage.singleSelectReply.selectedRowId
// Example: "model_select:gpt-4o"
```

## Button ID Naming Convention

Use prefixes to organize button IDs:

| Prefix | Purpose |
|--------|---------|
| `/` | Slash commands (e.g., `/prompt`, `/model gpt-4o`) |
| `model_select:` | Select a model |
| `modelcfg:` | Model configuration actions |
| `settings:` | Settings menu actions |

### Examples:

```
/prompt                    → slash command
/model gpt-4o             → slash command with args
model_select:gpt-4o       → select model
modelcfg:list              → modelcfg list action
modelcfg:add               → modelcfg add action
modelcfg:default:gpt-4o    → set default model
modelcfg_remove:gpt-4o    → remove model confirmation
settings:prompt            → settings prompt action
```

## Handling Button Responses

```javascript
async function handleButtonResponse(msg, chatId, senderId) {
  // Check for button response
  const buttonsResponse = msg?.message?.buttonsResponseMessage;
  const listResponse = msg?.message?.listResponseMessage;
  
  if (!buttonsResponse && !listResponse) return false;

  // Get selected ID
  const selectedId = 
    (buttonsResponse?.selectedButtonId) || 
    (listResponse?.singleSelectReply?.selectedRowId);
    
  if (!selectedId) return false;

  // Process based on ID prefix
  if (selectedId.startsWith('/')) {
    // It's a slash command - parse and execute
    const slashCommand = parseSlashCommand(selectedId);
    // ... handle command
    return true;
  }

  if (selectedId.startsWith('model_select:')) {
    const modelId = selectedId.replace('model_select:', '');
    setLlm2Model(chatId, modelId);
    await sock.sendMessage(chatId, { text: `Model set to: ${modelId}` });
    return true;
  }

  if (selectedId.startsWith('modelcfg:')) {
    const action = selectedId.replace('modelcfg:', '');
    // ... handle modelcfg actions
    return true;
  }

  return false;
}
```

## Common Mistakes

### ❌ WRONG: Using `rowId` instead of `id`
```javascript
// WRONG
rows: [{ rowId: '/prompt', title: 'Get Prompt' }]

// CORRECT
rows: [{ id: '/prompt', title: 'Get Prompt' }]
```

### ❌ WRONG: Mixing button types incorrectly
WhatsApp NativeFlow:
- All buttons in a message are sent together
- Each button is independent
- `quick_reply` buttons send text immediately
- `single_select` opens a dropdown first

### ❌ WRONG: Not handling the response
Button clicks need to be caught in `messages.upsert` and processed.

## Full Example: Settings Menu

```javascript
async function handleSettings({ chatId, chatType, senderIsAdmin, senderIsOwner }) {
  const buttons = [
    // quick_reply - sends command directly
    {
      name: 'quick_reply',
      buttonParamsJson: JSON.stringify({
        display_text: 'Get Prompt',
        id: '/prompt'
      })
    },
    // single_select - dropdown for model selection
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({
        title: 'Change Model',
        sections: [{
          title: 'Select Model',
          rows: [
            { id: 'model_select:gpt-4o', title: 'GPT-4o' },
            { id: 'model_select:gpt-4o-mini', title: 'GPT-4o Mini' }
          ]
        }]
      })
    },
    // single_select - dropdown for permission
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({
        title: 'Set Permission',
        sections: [{
          title: 'Permission Level',
          rows: [
            { id: '/permission 0', title: 'Forbidden' },
            { id: '/permission 1', title: 'Delete only' },
            { id: '/permission 2', title: 'Delete & mute' },
            { id: '/permission 3', title: 'All moderation' }
          ]
        }]
      })
    }
  ];

  await sendNativeFlow(
    sock, 
    chatId, 
    'Chat Settings\n\nSelect an option:', 
    buttons, 
    { footer: 'Click a button' }
  );
}
```

## Testing Buttons

Add debug logging to see button responses:

```javascript
logger.info({
  msgKey: msg?.key?.id,
  msgType: msg?.message ? Object.keys(msg.message).join(',') : 'none',
  hasButtons: !!buttonsResponse,
  hasList: !!listResponse,
  selectedButtonId: buttonsResponse?.selectedButtonId,
  selectedRowId: listResponse?.singleSelectReply?.selectedRowId
}, 'button response received');
```
