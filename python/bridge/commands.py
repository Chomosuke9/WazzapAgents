"""Slash-command parser and handler for /prompt, /reset, /permission.

/broadcast is handled entirely on the Node.js gateway side.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
  from .db import (
    get_prompt,
    set_prompt,
    get_permission,
    set_permission,
    permission_description,
  )
  from .log import setup_logging
except ImportError:
  import sys
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.db import get_prompt, set_prompt, get_permission, set_permission, permission_description  # type: ignore
  from bridge.log import setup_logging  # type: ignore

logger = setup_logging()

_PROMPT_MAX_CHARS = 4000

# Match "/command" at start of text, optionally followed by arguments.
_CMD_RE = re.compile(
  r"^/(prompt|reset|permission|broadcast)\b\s*(.*)",
  re.IGNORECASE | re.DOTALL,
)


@dataclass
class CommandResult:
  """Outcome of a slash-command execution."""
  command: str
  success: bool
  reply: str  # confirmation text to send back
  skip_llm: bool = True  # whether to skip LLM1/LLM2 processing


def parse_command(text: str | None) -> tuple[str | None, str | None]:
  """Return ``(command_name, args_text)`` or ``(None, None)``."""
  if not text:
    return None, None
  m = _CMD_RE.match(text.strip())
  if not m:
    return None, None
  return m.group(1).lower(), (m.group(2) or "").strip()


def handle_command(
  command: str,
  args: str,
  *,
  chat_id: str,
  chat_type: str,
  sender_is_admin: bool,
  sender_jid: str | None = None,
) -> Optional[CommandResult]:
  """
  Execute a slash command and return a ``CommandResult``.

  Returns ``None`` if the command is not handled here (e.g. /broadcast).
  """
  if command == "broadcast":
    # Handled by Node.js gateway; Python only records history.
    return None

  if command == "prompt":
    return _handle_prompt(args, chat_id=chat_id, chat_type=chat_type, sender_is_admin=sender_is_admin)

  if command == "reset":
    return _handle_reset(chat_id=chat_id, chat_type=chat_type, sender_is_admin=sender_is_admin)

  if command == "permission":
    return _handle_permission(args, chat_id=chat_id, chat_type=chat_type, sender_is_admin=sender_is_admin)

  return None


# ---------------------------------------------------------------------------
# /prompt
# ---------------------------------------------------------------------------

def _handle_prompt(
  args: str,
  *,
  chat_id: str,
  chat_type: str,
  sender_is_admin: bool,
) -> CommandResult:
  is_private = chat_type == "private"

  if not is_private and not sender_is_admin:
    return CommandResult(
      command="prompt",
      success=False,
      reply="Only group admins can use /prompt.",
    )

  if not args:
    # Show current prompt
    current = get_prompt(chat_id)
    if current:
      return CommandResult(
        command="prompt",
        success=True,
        reply=f"Current prompt:\n{current}",
      )
    return CommandResult(
      command="prompt",
      success=True,
      reply="No custom prompt set for this chat. Use /prompt <text> to set one.",
    )

  # Clear prompt if "-" or "clear"
  if args.strip().lower() in {"-", "clear", "reset"}:
    set_prompt(chat_id, None)
    return CommandResult(
      command="prompt",
      success=True,
      reply="Custom prompt cleared. Bot will use the default.",
    )

  if len(args) > _PROMPT_MAX_CHARS:
    return CommandResult(
      command="prompt",
      success=False,
      reply=f"Prompt too long ({len(args)} chars). Maximum is {_PROMPT_MAX_CHARS} characters.",
    )

  set_prompt(chat_id, args)
  preview = args[:200] + ("..." if len(args) > 200 else "")
  return CommandResult(
    command="prompt",
    success=True,
    reply=f"Prompt updated:\n{preview}",
  )


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

def _handle_reset(
  *,
  chat_id: str,
  chat_type: str,
  sender_is_admin: bool,
) -> CommandResult:
  is_private = chat_type == "private"

  if not is_private and not sender_is_admin:
    return CommandResult(
      command="reset",
      success=False,
      reply="Only group admins can use /reset.",
    )

  # Memory clearing is done by the caller (main.py) based on this result.
  return CommandResult(
    command="reset",
    success=True,
    reply="Bot memory for this chat has been reset.",
  )


# ---------------------------------------------------------------------------
# /permission
# ---------------------------------------------------------------------------

_PERMISSION_LABELS = {
  0: "0 (kick & delete forbidden)",
  1: "1 (delete allowed, kick forbidden)",
  2: "2 (kick allowed, delete forbidden)",
  3: "3 (kick & delete allowed)",
}


def _handle_permission(
  args: str,
  *,
  chat_id: str,
  chat_type: str,
  sender_is_admin: bool,
) -> CommandResult:
  if chat_type == "private":
    return CommandResult(
      command="permission",
      success=False,
      reply="/permission can only be used in group chats.",
    )

  if not sender_is_admin:
    return CommandResult(
      command="permission",
      success=False,
      reply="Only group admins can use /permission.",
    )

  if not args:
    current = get_permission(chat_id)
    label = _PERMISSION_LABELS.get(current, str(current))
    return CommandResult(
      command="permission",
      success=True,
      reply=f"Current permission level: {label}",
    )

  try:
    level = int(args.strip())
  except ValueError:
    return CommandResult(
      command="permission",
      success=False,
      reply="Usage: /permission 0, 1, 2, or 3.",
    )

  if level < 0 or level > 3:
    return CommandResult(
      command="permission",
      success=False,
      reply="Level must be 0-3.\n0: no kick/delete\n1: delete allowed\n2: kick allowed\n3: kick & delete allowed",
    )

  set_permission(chat_id, level)
  label = _PERMISSION_LABELS.get(level, str(level))
  return CommandResult(
    command="permission",
    success=True,
    reply=f"Permission updated: {label}",
  )
