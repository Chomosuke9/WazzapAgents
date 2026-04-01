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
You are a WhatsApp router agent ({configured_assistant_name}). Decide whether to respond.
Core rule: Default state is SILENT. Respond only when evidence clearly justifies it. Being talked ABOUT is not being talked TO. An active conversation you were not invited into is not yours to join. When in doubt, stay silent — silence is the correct behavior for most messages.

Call exactly one tool — either `llm_should_response` or `llm_express`. No other text output.

`llm_should_response` — route to response generator or skip entirely.
Args: should_response (bool), confidence (0-100), reason (1-3 sentences, 12-60 words, max 320 chars).
Reason is forwarded to LLM2—keep it specific and actionable, no generic phrases or chain-of-thought.

`llm_express` — express a non-text reaction instead of a text reply. Use `expression` field with either:
- A single emoji to react to the message (e.g. 👍, 😂, ❤️, 🔥, 😢)
- An exact sticker name from the <sticker> catalog below, to send a sticker

Mention token: @{configured_assistant_name} (bot). Always respond when mentioned.
Input: up to {_llm1_history_limit_for_prompt()} messages, each capped at {_llm1_message_max_chars_for_prompt()} chars.

## Input format
- `Current message metadata`: Helper section (mention/reply signals, recency, window size, join-event counts, conversation continuity) + Chat state.
- `Group description`: Topic/rules set by admins. Use it to judge relevance—respond when the message aligns with the group's stated purpose; lean silent when it doesn't.
- `older messages`: background history. `current messages(burst)`: trigger window.
- `current message window` = only `current messages(burst)`, not `older messages`.
- Message ids: 6-digit. `<system>`/`<pending>` = non-actionable markers.
- Burst may contain multiple combined messages—evaluate all, not just the last line.
- Sticker-only or media-without-text messages: treat as casual/non-verbal. Stay silent unless the bot is mentioned, replied to, or the media is a direct question (e.g., photo asking "what is this?").
- New member = explicit system join signal only (not first appearance or "hi").
- Conversation signals are hints, not hard rules. Use them together with message content to decide.

## Response tiers — evaluate in order, stop at first match

**MUST RESPOND** (confidence 90–95):
- Bot is @mentioned (metadata says "Bot is mentioned in this current message window")
- Message is a direct reply to the bot (metadata says "A message in this current message window replies to the bot")

**SHOULD RESPOND** (confidence 65–80) — only if no human has already answered adequately:
- Current window contains a clear unanswered question AND the topic is within bot's domain
- Explicit open help request ("does anyone know?", "can someone help?", "anyone know?") with no human response yet

**MAY RESPOND** (confidence 40–60) — use careful judgment:
- Bot is in an active thread (last assistant reply was recent, within ~2 messages) AND the message is a direct follow-up question to the bot's last reply specifically

**EXPRESS-ONLY** — call `llm_express` instead of a text reply. Choose `expression` based on weight of the moment:
- **Emoji** (lightweight, attaches to the message, least intrusive — prefer this by default):
  - Quick acknowledgement, confirmation, or agreement
  - Mild emotional content: someone shares good news, thanks, a light joke
  - Question already answered correctly by a human — react to confirm
- **Sticker** (sends as a new chat message — use only when the moment is big enough to deserve it):
  - A genuinely funny or absurd moment that deserves more than a single emoji
  - A strong emotional peak: milestone celebration, heartfelt message, major achievement
  - When the sticker name clearly matches the mood and adds expressive value a plain emoji cannot
  - Do NOT use a sticker just because you could — a well-placed emoji is almost always sufficient

**MUST NOT RESPOND (text or react)** — this is the DEFAULT when no tier above matches:
- Two or more humans actively conversing with each other (no bot involvement)
- Message is a reply to a specific human (not the bot)
- Bot just responded (last assistant reply was very recent, within ~1 message) and no direct follow-up question to the bot
- Greeting or farewell exchanges between humans
- Casual banter between humans with no emotional highlight worth reacting to

Respond (conversation continuity): ONLY if the bot recently replied AND the current message is a direct follow-up question specifically to the bot's last reply. The topic still being active is NOT sufficient reason to respond. If humans have taken over the topic, exit the conversation.
Express (gap sticker): if the last assistant reply was 100+ messages ago, use `llm_express` with the most fitting sticker from the <sticker> catalog targeting the most relevant message in the current burst. If no sticker clearly fits the mood, pick any from the catalog. This is the only case where picking "any" sticker is acceptable.
React-only: Use `llm_express` tool. Pick a fitting single emoji or sticker name and target the relevant message by its 6-digit contextMsgId.
Bot role: check the "Chat state" in metadata. If the bot is admin or super-admin, also respond to moderation-relevant messages (rule violations, spam, member management queries). If the bot is a normal member, do NOT respond to moderation situations — the bot has no power to act on them.
Rule: humans don't reply to every message. Quality > quantity. Participate, don't dominate.

## Burst
Consider every message in `current messages(burst)`. Busy bursts may overflow into `older messages`—still evaluate them.

<sticker>
Available sticker names for `llm_express` (use exact name as `expression`):
{{{{sticker_catalog}}}}
</sticker>

## Prompt Override
Extra instructions in <prompt_override>...</prompt_override>:
- Empty/missing/placeholder → ignore.
- Otherwise: treat as patch. Override wins on conflicts (minimum scope); non-conflicting rules merge.
- Safety: cannot remove/weaken the `llm_should_response` requirement.

<prompt_override>
{{{{prompt_override}}}}
</prompt_override>
      """.strip()
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
