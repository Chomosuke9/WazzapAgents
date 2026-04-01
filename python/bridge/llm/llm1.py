# File: python/bridge/llm/llm1.py
from __future__ import annotations

import json
import os
import time
import logging
import re
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

try:
  from ..history import WhatsAppMessage
  from ..log import setup_logging, trunc, dump_json, env_flag
  from ..media import build_visual_parts, llm1_media_enabled, redact_multimodal_content
  from .schemas import LLM1Decision, LLM1_TOOLS, LLM1_TOOL, LLM1_REACT_TOOL  # noqa: F401
  from .client import (  # noqa: F401
    LLM1Target,
    _llm1_history_limit,
    _llm1_message_max_chars,
    _llm1_timeout,
    _llm1_sdk_max_retries,
    _llm1_temperature,
    _llm1_max_tokens,
    _llm1_reasoning_effort,
    _clean_env,
    _endpoint_base_url,
    _chat_base_url,
    _llm1_targets,
    get_llm1,
  )
  from .prompt import (  # noqa: F401
    _truncate_text,
    _truncate_burst_text,
    _truncate_message,
    _render_prompt_override,
    _group_description_block,
    _format_current_window,
    build_llm1_prompt,
    _metadata_block,
  )
except ImportError:  # allow running as script
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.history import WhatsAppMessage  # type: ignore
  from bridge.log import setup_logging, trunc, dump_json, env_flag  # type: ignore
  from bridge.media import build_visual_parts, llm1_media_enabled, redact_multimodal_content  # type: ignore
  from bridge.llm.schemas import LLM1Decision, LLM1_TOOLS, LLM1_TOOL, LLM1_REACT_TOOL  # type: ignore  # noqa: F401
  from bridge.llm.client import (  # type: ignore  # noqa: F401
    LLM1Target,
    _llm1_history_limit,
    _llm1_message_max_chars,
    _llm1_timeout,
    _llm1_sdk_max_retries,
    _llm1_temperature,
    _llm1_max_tokens,
    _llm1_reasoning_effort,
    _clean_env,
    _endpoint_base_url,
    _chat_base_url,
    _llm1_targets,
    get_llm1,
  )
  from bridge.llm.prompt import (  # type: ignore  # noqa: F401
    _truncate_text,
    _truncate_burst_text,
    _truncate_message,
    _render_prompt_override,
    _group_description_block,
    _format_current_window,
    build_llm1_prompt,
    _metadata_block,
  )

logger = setup_logging()


def _prompt_to_langchain_messages(prompt: list[dict]) -> list[SystemMessage | HumanMessage]:
  messages: list[SystemMessage | HumanMessage] = []
  for item in prompt:
    if not isinstance(item, dict):
      continue
    role = str(item.get("role") or "").strip().lower()
    content = item.get("content", "")
    if role == "system":
      messages.append(SystemMessage(content=content))
    else:
      messages.append(HumanMessage(content=content))
  return messages


def _is_timeout_error(err: Exception) -> bool:
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < 8:
    if "timeout" in type(current).__name__.lower():
      return True
    current = current.__cause__ or current.__context__
    depth += 1
  return False


def _error_chain(err: Exception, limit: int = 8) -> list[str]:
  chain: list[str] = []
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < limit:
    chain.append(type(current).__name__)
    current = current.__cause__ or current.__context__
    depth += 1
  return chain


def _error_text_chain(err: Exception, limit: int = 8) -> str:
  texts: list[str] = []
  current: BaseException | None = err
  depth = 0
  while current is not None and depth < limit:
    text = str(current).strip()
    if text:
      texts.append(text.lower())
    current = current.__cause__ or current.__context__
    depth += 1
  return " | ".join(texts)


def _is_reasoning_unsupported_error(err: Exception) -> bool:
  text = _error_text_chain(err)
  if "reasoning_effort" not in text and "reasoning effort" not in text:
    return False
  unsupported_markers = (
    "unsupported",
    "not supported",
    "unknown",
    "invalid",
    "not allowed",
    "unrecognized",
  )
  return any(marker in text for marker in unsupported_markers)


def _llm1_ctx(
  current: WhatsAppMessage,
  *,
  provider: str,
  model: str,
  url: str | None,
  current_payload: dict | None = None,
) -> dict:
  payload = current_payload if isinstance(current_payload, dict) else {}
  chat_id = payload.get("chatId") or payload.get("chat_id")
  raw_chat_type = str(payload.get("chatType") or payload.get("chat_type") or "").strip().lower()
  if raw_chat_type not in {"group", "private"}:
    if isinstance(chat_id, str) and chat_id.endswith("@g.us"):
      raw_chat_type = "group"
    else:
      raw_chat_type = "group" if bool(payload.get("isGroup")) else "private"
  chat_name = (payload.get("chatName") or payload.get("chat_name")) if raw_chat_type == "group" else None
  return {
    "chat_id": chat_id or getattr(current, "sender", None),
    "chat_name": chat_name,
    "message_id": getattr(current, "message_id", None) or getattr(current, "id", None),
    "provider": provider,
    "model": model,
    "endpoint": url,
  }


def _log_llm1_decision(
  decision: LLM1Decision,
  *,
  ctx: dict,
  elapsed_ms: int,
  source: str,
) -> None:
  status = "respond" if decision.should_response else "skip"
  reason_text = trunc(" ".join((decision.reason or "").split()), 220)
  logger.info(
    'LLM1 decision final (%s): %s conf=%s%% reason="%s" elapsed=%sms',
    source,
    status,
    decision.confidence,
    reason_text,
    elapsed_ms,
    extra={
      **ctx,
      "source": source,
      "should_response": decision.should_response,
      "confidence": decision.confidence,
      "reason": decision.reason,
      "elapsed_ms": elapsed_ms,
      "raw": trunc(dump_json(decision.model_dump()), 400),
    },
  )


def _extract_tool_args(tool_call) -> dict:
  """Best-effort extraction of tool arguments across provider shapes."""
  raw_args = None
  raw_fn = None

  if isinstance(tool_call, dict):
    raw_args = (
      tool_call.get("args")
      or tool_call.get("arguments")
      or tool_call.get("input")
      or tool_call.get("parameters")
    )
    raw_fn = tool_call.get("function")
  else:
    raw_args = (
      getattr(tool_call, "args", None)
      or getattr(tool_call, "arguments", None)
      or getattr(tool_call, "input", None)
      or getattr(tool_call, "parameters", None)
    )
    raw_fn = getattr(tool_call, "function", None)

  # OpenAI-like shape: {"function": {"arguments": "..."}}
  if not raw_args and isinstance(raw_fn, dict):
    raw_args = (
      raw_fn.get("args")
      or raw_fn.get("arguments")
      or raw_fn.get("input")
      or raw_fn.get("parameters")
    )

  if isinstance(raw_args, str):
    try:
      raw_args = json.loads(raw_args)
    except json.JSONDecodeError:
      return {}

  return raw_args or {}


def _content_to_text(content) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: list[str] = []
    for item in content:
      if not isinstance(item, dict):
        parts.append(str(item))
        continue
      if item.get("type") == "text":
        parts.append(str(item.get("text") or ""))
        continue
      if item.get("type") == "image_url":
        parts.append("[image]")
        continue
      parts.append(f"[{item.get('type') or 'part'}]")
    return "\n".join(parts)
  return str(content)


def _extract_decision_from_content(content) -> dict:
  text = _content_to_text(content).strip()
  if not text:
    return {}

  candidates: list[str] = [text]
  fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
  if fenced:
    fenced_text = fenced.group(1).strip()
    if fenced_text:
      candidates.append(fenced_text)

  first_brace = text.find("{")
  last_brace = text.rfind("}")
  if first_brace >= 0 and last_brace > first_brace:
    candidates.append(text[first_brace : last_brace + 1].strip())

  for candidate in candidates:
    try:
      parsed = json.loads(candidate)
    except Exception:
      continue
    if isinstance(parsed, dict):
      return parsed
  return {}


def _redact_messages_for_log(messages: list[dict]) -> list[dict]:
  redacted: list[dict] = []
  for msg in messages:
    if not isinstance(msg, dict):
      continue
    copied = dict(msg)
    copied["content"] = redact_multimodal_content(copied.get("content"))
    redacted.append(copied)
  return redacted


async def call_llm1(
  history: Iterable[WhatsAppMessage],
  current: WhatsAppMessage,
  *,
  timeout: float = 8.0,
  client: Optional[ChatOpenAI] = None,
  current_payload: dict | None = None,
  group_description: str | None = None,
  prompt_override: str | None = None,
) -> LLM1Decision:
  primary_endpoint = _clean_env(os.getenv("LLM1_ENDPOINT"))
  fallback_endpoint = _clean_env(os.getenv("LLM1_FALLBACK_ENDPOINT"))
  # If LLM1 is not configured, allow responding by default.
  if not primary_endpoint and not fallback_endpoint:
    logger.debug("LLM1 disabled (no LLM1_ENDPOINT set); defaulting to respond")
    return LLM1Decision(should_response=True, confidence=50, reason="llm1_disabled")

  targets = _llm1_targets()
  if client is not None and targets:
    targets = targets[:1]
  if not targets:
    logger.debug("LLM1 endpoint missing after normalization; defaulting to skip")
    return LLM1Decision(should_response=False, confidence=10, reason="llm1_missing_url")

  history_limit = _llm1_history_limit()
  message_max_chars = _llm1_message_max_chars()
  history_list = list(history)
  prompt_history = history_list[-history_limit:]
  current_media_parts: list[dict] = []
  current_media_notes: list[str] = []
  if llm1_media_enabled():
    current_media_parts, current_media_notes = build_visual_parts(current_payload)
  prompt = build_llm1_prompt(
    prompt_history,
    current,
    history_limit=history_limit,
    message_max_chars=message_max_chars,
    current_media_parts=current_media_parts,
    current_media_notes=current_media_notes,
    metadata_block=_metadata_block(current_payload),
    group_description=group_description,
    prompt_override=prompt_override,
  )
  prompt_text = "\n".join(
    [_content_to_text(m.get("content", "")) for m in prompt if isinstance(m, dict)]
  )

  last_failure: LLM1Decision | None = None
  total_targets = len(targets)
  llm1_temperature = _llm1_temperature()
  llm1_max_tokens = _llm1_max_tokens()

  for idx, target in enumerate(targets):
    has_next_target = idx < (total_targets - 1)
    reasoning_effort = _llm1_reasoning_effort()
    t0 = time.perf_counter()
    ctx = _llm1_ctx(
      current,
      provider=target.name,
      model=target.model,
      url=target.base_url,
      current_payload=current_payload,
    )
    llm = client if (client is not None and idx == 0) else get_llm1(
      model=target.model,
      base_url=target.base_url,
      api_key=target.api_key,
      timeout=timeout,
      include_reasoning=bool(reasoning_effort),
    )

    if env_flag("BRIDGE_LOG_PROMPT_FULL"):
      logger.info(
        "LLM1 prompt full",
        extra={
          **ctx,
          "history_limit": history_limit,
          "history_used": len(prompt_history),
          "message_max_chars": message_max_chars,
          "base_url": target.base_url,
          "media_parts": len(current_media_parts),
          "reasoning_effort": reasoning_effort,
          "messages": _redact_messages_for_log(prompt),
        },
      )

    logger.info(
      "LLM1 invoke start (model=%s, history=%s)",
      target.model,
      len(prompt_history),
      extra={
        **ctx,
        "history_used": len(prompt_history),
        "media_parts": len(current_media_parts),
        "temperature": llm1_temperature,
        "max_tokens": llm1_max_tokens,
        "reasoning_effort": reasoning_effort,
      },
    )

    logger.debug(
      "LLM1 request start",
      extra={
        **ctx,
        "history_limit": history_limit,
        "history_used": len(prompt_history),
        "message_max_chars": message_max_chars,
        "timeout_s": _llm1_timeout(timeout),
        "prompt_chars": len(prompt_text),
        "prompt_preview": trunc(prompt_text, 300),
        "media_parts": len(current_media_parts),
        "base_url": target.base_url,
        "temperature": llm1_temperature,
        "max_tokens": llm1_max_tokens,
        "reasoning_effort": reasoning_effort,
        "tool_names": [t["function"]["name"] for t in LLM1_TOOLS],
      },
    )

    async def _invoke_once(llm_client: ChatOpenAI):
      try:
        llm_with_tool = llm_client.bind_tools(
          LLM1_TOOLS,
          tool_choice="required",
        )
      except Exception as err:
        logger.warning(
          "LLM1 bind_tools with tool_choice=required failed; retrying default bind_tools",
          exc_info=err,
          extra={
            **ctx,
            "error_type": type(err).__name__,
          },
        )
        llm_with_tool = llm_client.bind_tools(LLM1_TOOLS)
      return await llm_with_tool.ainvoke(_prompt_to_langchain_messages(prompt))

    try:
      response = await _invoke_once(llm)
    except Exception as err:
      # Some providers reject `reasoning_effort`. Retry once without it when using
      # internally created client.
      if (
        client is None
        and reasoning_effort
        and _is_reasoning_unsupported_error(err)
      ):
        logger.warning(
          "LLM1 invoke rejected reasoning_effort; retrying without reasoning",
          exc_info=True,
          extra={
            **ctx,
            "reasoning_effort": reasoning_effort,
            "error_type": type(err).__name__,
            "error_chain": _error_chain(err),
          },
        )
        llm = get_llm1(
          model=target.model,
          base_url=target.base_url,
          api_key=target.api_key,
          timeout=timeout,
          include_reasoning=False,
        )
        reasoning_effort = None
        try:
          response = await _invoke_once(llm)
        except Exception as retry_err:
          elapsed_ms = int((time.perf_counter() - t0) * 1000)
          timeout_error = _is_timeout_error(retry_err)
          logger.error(
            "LLM1 invoke failed after reasoning fallback",
            exc_info=True,
            extra={
              **ctx,
              "elapsed_ms": elapsed_ms,
              "reasoning_effort": reasoning_effort,
              "error_type": type(retry_err).__name__,
              "error_chain": _error_chain(retry_err),
              "will_try_fallback_target": has_next_target,
            },
          )
          last_failure = LLM1Decision(
            should_response=False,
            confidence=10,
            reason="llm1_unreachable" if timeout_error else "llm1_exception",
          )
          if has_next_target:
            logger.warning(
              "LLM1 provider failed; trying fallback target",
              extra={
                **ctx,
                "next_provider": targets[idx + 1].name,
              },
            )
            continue
          return last_failure
      else:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        timeout_error = _is_timeout_error(err)
        logger.error(
          "LLM1 invoke failed",
          exc_info=True,
          extra={
            **ctx,
            "elapsed_ms": elapsed_ms,
            "reasoning_effort": reasoning_effort,
            "error_type": type(err).__name__,
            "error_chain": _error_chain(err),
            "will_try_fallback_target": has_next_target,
          },
        )
        last_failure = LLM1Decision(
          should_response=False,
          confidence=10,
          reason="llm1_unreachable" if timeout_error else "llm1_exception",
        )
        if has_next_target:
          logger.warning(
            "LLM1 provider failed; trying fallback target",
            extra={
              **ctx,
              "next_provider": targets[idx + 1].name,
            },
          )
          continue
        return last_failure

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    response_metadata = getattr(response, "response_metadata", None)
    usage_metadata = getattr(response, "usage_metadata", None)
    _llm1_input_tokens = 0
    _llm1_output_tokens = 0
    if isinstance(usage_metadata, dict):
      _llm1_input_tokens = usage_metadata.get("input_tokens", 0) or 0
      _llm1_output_tokens = usage_metadata.get("output_tokens", 0) or 0
    raw_tool_calls = getattr(response, "tool_calls", None) or []
    content = getattr(response, "content", None)
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    if not raw_tool_calls and isinstance(additional_kwargs, dict):
      maybe_tool_calls = additional_kwargs.get("tool_calls")
      if isinstance(maybe_tool_calls, list):
        raw_tool_calls = maybe_tool_calls

    logger.debug(
      "LLM1 response received",
      extra={
        **ctx,
        "elapsed_ms": elapsed_ms,
        "reasoning_effort": reasoning_effort,
        "response_metadata": response_metadata,
        "usage": usage_metadata,
        "tool_calls_count": len(raw_tool_calls),
        "content_preview": trunc(_content_to_text(content), 600),
      },
    )

    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(
        "LLM1 raw response",
        extra={
          **ctx,
          "raw": dump_json(getattr(response, "model_dump", lambda: str(response))()),
        },
      )

    tool_calls = raw_tool_calls or []
    if not tool_calls:
      parsed_fallback = _extract_decision_from_content(content)
      if parsed_fallback:
        try:
          decision = LLM1Decision.model_validate(parsed_fallback)
          logger.warning(
            "LLM1 response missing tool call; parsed JSON fallback",
            extra={
              **ctx,
              "reasoning_effort": reasoning_effort,
              "response_metadata": response_metadata,
              "fallback_args": parsed_fallback,
            },
          )
          decision.input_tokens = _llm1_input_tokens
          decision.output_tokens = _llm1_output_tokens
          _log_llm1_decision(
            decision,
            ctx=ctx,
            elapsed_ms=elapsed_ms,
            source="json_fallback",
          )
          return decision
        except ValidationError:
          pass
      logger.warning(
        "LLM1 response missing tool call",
        extra={
          **ctx,
          "reasoning_effort": reasoning_effort,
          "response_metadata": response_metadata,
          "will_try_fallback_target": has_next_target,
        },
      )
      last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_no_tool")
      if has_next_target:
        logger.warning(
          "LLM1 invalid response shape; trying fallback target",
          extra={
            **ctx,
            "next_provider": targets[idx + 1].name,
          },
        )
        continue
      return last_failure

    # Detect which tool was called: llm_should_response or llm_express
    respond_tool_name = LLM1_TOOL["function"]["name"]
    react_tool_name = LLM1_REACT_TOOL["function"]["name"]

    def _get_tool_call_name(tc) -> str | None:
      if isinstance(tc, dict):
        name = tc.get("name")
        if name:
          return str(name)
        fn = tc.get("function")
        if isinstance(fn, dict):
          return fn.get("name")
      else:
        name = getattr(tc, "name", None)
        if name:
          return str(name)
        fn = getattr(tc, "function", None)
        if isinstance(fn, dict):
          return fn.get("name")
      return None

    # Find the first recognized tool call
    tool_call = None
    called_tool_name = None
    for tc in tool_calls:
      tc_name = _get_tool_call_name(tc)
      if tc_name in (respond_tool_name, react_tool_name):
        tool_call = tc
        called_tool_name = tc_name
        break
    if tool_call is None:
      tool_call = tool_calls[0]
      called_tool_name = _get_tool_call_name(tool_call)

    args = _extract_tool_args(tool_call)
    if not args:
      logger.warning(
        "LLM1 tool args empty",
        extra={
          **ctx,
          "raw_tool_call": trunc(str(tool_call), 500),
          "will_try_fallback_target": has_next_target,
        },
      )
      last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_empty_tool")
      if has_next_target:
        logger.warning(
          "LLM1 invalid tool args; trying fallback target",
          extra={
            **ctx,
            "next_provider": targets[idx + 1].name,
          },
        )
        continue
      return last_failure

    # Handle llm_express tool call
    if called_tool_name == react_tool_name:
      react_expression = str(args.get("expression") or "").strip()
      react_context_msg_id = str(args.get("context_msg_id") or "").strip()
      react_confidence = args.get("confidence", 50)
      react_reason = str(args.get("reason") or "express-only").strip()
      if not react_expression or not react_context_msg_id:
        logger.warning(
          "LLM1 llm_express missing expression or context_msg_id",
          extra={**ctx, "raw_args": args, "will_try_fallback_target": has_next_target},
        )
        last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_invalid_express_tool")
        if has_next_target:
          continue
        return last_failure
      decision = LLM1Decision(
        should_response=False,
        confidence=react_confidence if isinstance(react_confidence, int) else 50,
        reason=react_reason[:320],
        react_expression=react_expression,
        react_context_msg_id=react_context_msg_id,
        input_tokens=_llm1_input_tokens,
        output_tokens=_llm1_output_tokens,
      )
      logger.info(
        'LLM1 express decision: expression=%s target=%s conf=%s%% reason="%s" elapsed=%sms',
        react_expression,
        react_context_msg_id,
        decision.confidence,
        trunc(" ".join((decision.reason or "").split()), 220),
        elapsed_ms,
        extra={
          **ctx,
          "source": "express_tool_call",
          "should_response": False,
          "react_expression": react_expression,
          "react_context_msg_id": react_context_msg_id,
          "confidence": decision.confidence,
          "reason": decision.reason,
          "elapsed_ms": elapsed_ms,
        },
      )
      return decision

    # Handle llm_should_response tool call (existing behavior)
    try:
      decision = LLM1Decision.model_validate(args)
    except ValidationError as err:
      logger.warning(
        "LLM1 tool args failed validation",
        exc_info=err,
        extra={**ctx, "raw_args": args, "will_try_fallback_target": has_next_target},
      )
      last_failure = LLM1Decision(should_response=False, confidence=10, reason="llm1_invalid_tool")
      if has_next_target:
        logger.warning(
          "LLM1 invalid tool args; trying fallback target",
          extra={
            **ctx,
            "next_provider": targets[idx + 1].name,
          },
        )
        continue
      return last_failure

    decision.input_tokens = _llm1_input_tokens
    decision.output_tokens = _llm1_output_tokens
    _log_llm1_decision(
      decision,
      ctx=ctx,
      elapsed_ms=elapsed_ms,
      source="tool_call",
    )
    return decision

  return last_failure or LLM1Decision(should_response=False, confidence=10, reason="llm1_exception")
