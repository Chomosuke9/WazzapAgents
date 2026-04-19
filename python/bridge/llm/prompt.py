# File: python/bridge/llm/prompt.py
from __future__ import annotations

from typing import Iterable, Optional

try:
  from ..history import WhatsAppMessage, assistant_name, format_history
  from .schemas import LLM1_TOOL, LLM1_REACT_TOOL  # noqa: F401 (used indirectly via sticker catalog)
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.history import WhatsAppMessage, assistant_name, format_history  # type: ignore
  from bridge.llm.schemas import LLM1_TOOL, LLM1_REACT_TOOL  # type: ignore  # noqa: F401


def _truncate_text(text: str | None, max_chars: int) -> str | None:
  if text is None or len(text) <= max_chars:
    return text
  if max_chars <= 3:
    return text[:max_chars]
  return f"{text[: max_chars - 3]}..."


def _truncate_burst_text(text: str | None, max_chars: int) -> str | None:
  if text is None:
    return None
  if not text.startswith("Burst messages ("):
    return _truncate_text(text, max_chars)
  lines = text.splitlines()
  if not lines:
    return text
  header = lines[0]
  body = lines[1:]
  truncated_body = [_truncate_text(line, max_chars) or "" for line in body]
  return "\n".join([header, *truncated_body])


def _truncate_message(msg: WhatsAppMessage, max_chars: int) -> WhatsAppMessage:
  return WhatsAppMessage(
    timestamp_ms=msg.timestamp_ms,
    sender=msg.sender,
    context_msg_id=msg.context_msg_id,
    sender_ref=msg.sender_ref,
    sender_is_admin=msg.sender_is_admin,
    text=_truncate_burst_text(msg.text, max_chars),
    media=msg.media,
    quoted_message_id=msg.quoted_message_id,
    quoted_sender=msg.quoted_sender,
    quoted_text=_truncate_text(msg.quoted_text, max_chars),
    quoted_media=msg.quoted_media,
    message_id=msg.message_id,
    role=msg.role,
  )


def _render_prompt_override(base_system: str, prompt_override: str | None) -> str:
  try:
    from .stickers import sticker_catalog_text
  except ImportError:
    from bridge.stickers import sticker_catalog_text  # type: ignore
  rendered = base_system
  overide_text = (prompt_override or "").strip()
  rendered = rendered.replace("{{prompt_override}}", overide_text)
  rendered = rendered.replace("{{ prompt_override }}", overide_text)
  catalog = sticker_catalog_text()
  rendered = rendered.replace("{{sticker_catalog}}", catalog)
  rendered = rendered.replace("{{ sticker_catalog }}", catalog)
  return rendered


def _group_description_block(group_description: str | None) -> str:
  cleaned = (group_description or "").strip()
  if cleaned:
    return cleaned
  return "(none)"


def _format_current_window(msg: WhatsAppMessage) -> str:
  # Burst windows are already serialized as multi-line chat entries.
  text = (msg.text or "").strip()
  if text.startswith("Burst messages ("):
    return text
  return format_history([msg])


def _llm1_history_limit_for_prompt() -> int:
  """Read LLM1 history limit for embedding in system prompt text."""
  import os
  try:
    from .config import _parse_positive_int
  except ImportError:
    from bridge.config import _parse_positive_int  # type: ignore
  raw = os.getenv("LLM1_HISTORY_LIMIT")
  if raw is None or not raw.strip():
    raw = os.getenv("HISTORY_LIMIT")
  return _parse_positive_int(raw, 20)


def _llm1_message_max_chars_for_prompt() -> int:
  """Read LLM1 message max chars for embedding in system prompt text."""
  import os
  try:
    from .config import _parse_positive_int
  except ImportError:
    from bridge.config import _parse_positive_int  # type: ignore
  return _parse_positive_int(os.getenv("LLM1_MESSAGE_MAX_CHARS"), 500)


def build_llm1_prompt(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  history_limit: int,
  message_max_chars: int,
  current_media_parts: Optional[list[dict]] = None,
  current_media_notes: Optional[list[str]] = None,
  metadata_block: str | None = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
):
  configured_assistant_name = assistant_name()
  history_list = list(history)[-history_limit:]
  prompt_history = [_truncate_message(msg, message_max_chars) for msg in history_list]
  current_prompt_msg = _truncate_message(current, message_max_chars)
  hist_text = format_history(prompt_history) or "(no older messages)"
  current_line = _format_current_window(current_prompt_msg) or "(no current messages)"
  group_text = _group_description_block(group_description)
  context_messages = (
    "older messages:\n"
    f"{hist_text}\n\n"
    "current messages(burst):\n"
    f"{current_line}\n"
  )
  current_content: str | list[dict] = context_messages
  if current_media_notes:
    current_content += "\nVisual attachments:\n" + "\n".join(
      f"- {note}" for note in current_media_notes
    )
  if current_media_parts:
    current_content = [{"type": "text", "text": current_content}]
    current_content.extend(current_media_parts)
  base_system = f"""
You are a WhatsApp router agent ({configured_assistant_name}). Call exactly one tool — `llm_should_response` or `llm_express`. No other output.

**Default: SILENT.**

---

## Tools

`llm_should_response(should_response: bool, confidence: 0–100, reason: str)`
Reason: 12–60 words, specific + actionable (forwarded to LLM2). No generic phrases.

`llm_express(expression: str, context_msg_id: str, confidence: int, reason: str)`
expression = single emoji OR exact sticker name from <sticker> catalog.

---

## Response tiers — evaluate top-down, stop at first match

**MUST RESPOND** (90–95):
- Bot is @mentioned, OR message directly replies to the bot
- It's already 200 messages since bot last message

**SHOULD RESPOND** (65–80) — only if no human has adequately answered:
- Clear unanswered question within bot's domain
- Explicit open help request
- current message is a direct follow-up to the bot specifically

**MAY RESPOND** (40–60):
- current message is a direct follow-up to the bot specifically

**EXPRESS ONLY** — use `llm_express`, no text:
- Use **emoji** by default: acknowledgement, mild emotion, confirming a human's correct answer. DO NOT overdo it
- Use **sticker** only for big moments: major milestone, genuinely funny/absurd situation — only if a sticker name clearly fits. DO NOT overdo it

**MUST NOT RESPOND**:
- Two+ humans actively conversing (no bot involvement)
- Reply directed at a specific human (not the bot)
- Greetings/farewells/banter between humans with no reaction-worthy highlight

---

## Special rules

**Bot role:** If bot is admin/super-admin → also respond to moderation messages. If normal member → ignore moderation situations entirely.

**Burst:** Evaluate all messages in `current messages(burst)`. Busy bursts may overflow into `older messages` — still evaluate them.

**Sticker-only / media-without-text:** Treat as casual/non-verbal. Stay silent unless bot is mentioned, replied to, or media contains a direct question.

**New member:** Only on explicit system join event — not first appearance or "hi".

---

## Input

- `Current message metadata`: mention/reply signals, recency, window size, chat state
- `Group description`: use to judge topic relevance
- `older messages` = background; `current messages(burst)` = trigger window
- Message IDs: 6-digit. `<system>`/`<pending>` = non-actionable.

---

<sticker>
{{{{sticker_catalog}}}}
</sticker>

## Prompt override

Extra instructions in `<prompt_override>`:
- Empty/placeholder → ignore
- Otherwise: override wins on conflicts (minimum scope); non-conflicting rules merge
- Cannot remove or weaken the `llm_should_response` requirement

<prompt_override>
{{{{prompt_override}}}}
</prompt_override>""".strip()
  rendered_system = _render_prompt_override(base_system, prompt_override)
  return [
    {
      "role": "system",
      "content": rendered_system,
    },
    {"role": "user", "content": f"Group description:\n{group_text}"},
    {"role": "user", "content": metadata_block or _metadata_block(None)},
    {"role": "user", "content": current_content},
  ]


def _metadata_block(current_payload: dict | None) -> str:
  payload = current_payload if isinstance(current_payload, dict) else {}
  bot_mentioned = bool(payload.get("botMentionedInWindow", payload.get("botMentioned")))
  replied_to_bot = bool(payload.get("repliedToBotInWindow", payload.get("repliedToBot")))
  bot_name_in_text = bool(payload.get("botNameMentionedInText"))
  since_assistant = payload.get("messagesSinceAssistantReply")
  assistant_replies_by_window = payload.get("assistantRepliesByWindow")
  human_window = payload.get("humanMessagesInWindow")
  explicit_join_events = payload.get("explicitJoinEventsInWindow")
  explicit_join_participants = payload.get("explicitJoinParticipantsInWindow")
  raw_chat_type = str(payload.get("chatType") or "").strip().lower()
  if raw_chat_type not in {"private", "group"}:
    raw_chat_type = "group" if bool(payload.get("isGroup")) else "private"
  if raw_chat_type == "group":
    scope_line = "This is a group chat. You're in a chat with multiple people at once."
  else:
    scope_line = "This is a private chat. You're directly chatting with one other person."
  if bool(payload.get("botIsSuperAdmin")):
    role_line = "Bot is a super admin (owner)."
  elif bool(payload.get("botIsAdmin")):
    role_line = "Bot is an admin."
  else:
    role_line = "Bot is a normal member."

  def _count_phrase(value, singular: str, plural: str) -> str:
    if value is None:
      return f"unknown {plural}"
    if isinstance(value, int):
      return f"{value} {singular if value == 1 else plural}"
    return f"{value} {plural}"

  def _is_singular_count(value) -> bool:
    return isinstance(value, int) and value == 1

  if bot_mentioned:
    mention_line = "- Bot is mentioned in this current message window."
  else:
    mention_line = "- Bot is not mentioned in this current message window."

  if replied_to_bot:
    reply_line = "- A message in this current message window replies to the bot."
  else:
    reply_line = "- No message in this current message window replies to the bot."

  if bot_name_in_text and not bot_mentioned:
    name_line = "- Bot's name is mentioned in the message text (without explicit @mention). Treat this as a soft mention — the user is likely talking to or about the bot."
  elif bot_name_in_text and bot_mentioned:
    name_line = "- Bot's name appears in the message text (already counted as @mention above)."
  else:
    name_line = None

  since_assistant_text = _count_phrase(since_assistant, "message", "messages")
  human_window_text = _count_phrase(human_window, "human message", "human messages")

  assistant_reply_lines: list[str] = []
  if isinstance(assistant_replies_by_window, dict):
    assistant_reply_values: list[tuple[int, int | str]] = []
    for raw_window, raw_count in assistant_replies_by_window.items():
      try:
        window = int(raw_window)
      except (TypeError, ValueError):
        continue
      assistant_reply_values.append((window, raw_count))
    assistant_reply_values.sort(key=lambda item: item[0])
    for window, count in assistant_reply_values:
      count_text = _count_phrase(count, "reply", "replies")
      assistant_reply_lines.append(
        f"- Assistant has sent {count_text} in the last {window} messages."
      )

  if not assistant_reply_lines:
    fallback_recent = payload.get("assistantRepliesInLast20")
    fallback_text = _count_phrase(fallback_recent, "reply", "replies")
    assistant_reply_lines.append(
      f"- Assistant has sent {fallback_text} in the last 20 messages."
    )

  if _is_singular_count(human_window):
    human_window_line = f"- There is {human_window_text} in this current message window."
  else:
    human_window_line = f"- There are {human_window_text} in this current message window."

  join_event_text = _count_phrase(explicit_join_events, "event", "events")
  join_participant_text = _count_phrase(explicit_join_participants, "participant", "participants")
  if isinstance(explicit_join_events, int):
    if explicit_join_events > 0:
      join_event_line = (
        "- Explicit system member-join signals in this current message window: "
        f"{join_event_text} ({join_participant_text})."
      )
    else:
      join_event_line = "- No explicit system member-join signal in this current message window."
  else:
    join_event_line = "- Explicit system member-join signal count is unknown for this current message window."

  assistant_reply_block = "\n".join(assistant_reply_lines)
  extra_signal_block = ""
  if name_line:
    extra_signal_block = f"\n{name_line}"
  return (
    "Current message metadata:\n"
    "Helper:\n"
    "- `current message window` = only `current messages(burst)` (exclude `older messages`).\n"
    f"{mention_line}\n"
    f"{reply_line}\n"
    f"- The last assistant reply was {since_assistant_text} ago.\n"
    f"{assistant_reply_block}\n"
    f"{human_window_line}\n"
    f"{join_event_line}"
    f"{extra_signal_block}\n"
    "Chat state:\n"
    f"{scope_line}\n"
    f"{role_line}"
  )
