"""
LLM tool-calling runner (Cerebras primary, Groq fallback).

This module provides a generic loop:
- ask model with `tools`
- if model requests tool calls, execute them and append tool results
- stop when model returns a final message without tool_calls

All text outputs are expected to be English for Faz 3.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.core.debug_probe import debug_probe
from app.services.llm.cerebras_service import CerebrasService
from app.services.llm.groq_service import GroqService
from app.services.telegram_operator_alerts import fire_operator_alert, format_exc_brief
from app.tools.executor import ToolExecutor

log = get_logger("llm.tool_calling")
_MAX_TOOL_CONTENT_CHARS = 1400
_MAX_CHAT_MESSAGES = 28


@dataclass(frozen=True)
class ToolRunResult:
    reasoning_text: str
    decisions: list[dict[str, Any]]
    model: str
    iterations: int


def _extract_json_array(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Split `text` into (reasoning, decisions_array) where decisions_array is parsed JSON.
    Tolerates extra prose around the JSON array.
    """
    raw = (text or "").strip()
    if not raw:
        return "", []

    # Find the first JSON array in the output
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return raw, []

    maybe_json = raw[start:end]
    try:
        data = json.loads(maybe_json)
        if isinstance(data, list):
            cleaned: list[dict[str, Any]] = [x for x in data if isinstance(x, dict)]
            reasoning = raw[:start].strip()
            return reasoning or raw, cleaned
    except Exception:
        pass

    # If parse fails, return whole text as reasoning.
    return raw, []


def _coerce_tool_args(args: Any) -> dict:
    """Groq tool call args may be a JSON string or already a dict."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            # try to salvage: remove trailing commas, etc.
            s2 = re.sub(r",\s*}", "}", s)
            s2 = re.sub(r",\s*]", "]", s2)
            try:
                return json.loads(s2)
            except Exception:
                return {}
    return {}


def _truncate_tool_content(text: str) -> str:
    body = (text or "").strip()
    if len(body) <= _MAX_TOOL_CONTENT_CHARS:
        return body
    return body[:_MAX_TOOL_CONTENT_CHARS] + "\n... [truncated]"


def _compact_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep prompt anchors (system + first user) and trim rolling history.
    Prevents context_length_exceeded when tool loops append many entries.
    """
    if len(messages) <= _MAX_CHAT_MESSAGES:
        return messages
    head: list[dict[str, Any]] = []
    idx = 0
    if messages and messages[0].get("role") == "system":
        head.append(messages[0])
        idx = 1
    if idx < len(messages) and messages[idx].get("role") == "user":
        head.append(messages[idx])
        idx += 1
    budget = max(4, _MAX_CHAT_MESSAGES - len(head))
    tail = messages[-budget:]
    return head + tail


def _serialize_assistant_message(msg: Any) -> dict[str, Any]:
    """Build OpenAI-compatible assistant message for Groq follow-up turns."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        return out
    serialized: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        arguments = getattr(fn, "arguments", "") if fn else ""
        if hasattr(arguments, "model_dump_json"):
            arguments = arguments.model_dump_json()
        elif not isinstance(arguments, str):
            arguments = str(arguments or "")
        serialized.append(
            {
                "id": getattr(tc, "id", "") or "",
                "type": getattr(tc, "type", "function") or "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    out["tool_calls"] = serialized
    return out


def _estimate_payload_chars(messages: list[dict[str, Any]], tools: list[dict]) -> tuple[int, int]:
    """Cheap payload size estimator for request diagnostics."""
    try:
        messages_chars = len(json.dumps(messages, ensure_ascii=False, default=str))
    except Exception:
        messages_chars = len(str(messages))
    try:
        tools_chars = len(json.dumps(tools, ensure_ascii=False, default=str))
    except Exception:
        tools_chars = len(str(tools))
    return messages_chars, tools_chars


def _error_detail(exc: Exception) -> str:
    """Best-effort extraction of Groq error payload without leaking secrets."""
    parts: list[str] = []
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    body = getattr(response, "text", None)
    if body:
        parts.append(f"body={str(body)[:700]}")
    # some SDK errors expose parsed payload directly
    data = getattr(exc, "body", None) or getattr(exc, "error", None)
    if data:
        if isinstance(data, Mapping):
            try:
                parts.append(f"payload={json.dumps(data, ensure_ascii=False)[:700]}")
            except Exception:
                parts.append(f"payload={str(data)[:700]}")
        else:
            parts.append(f"payload={str(data)[:700]}")
    if not parts:
        parts.append(str(exc)[:700])
    return " | ".join(parts)


def _is_rate_limited(exc: Exception) -> bool:
    """Return True when provider error is an HTTP 429/rate-limit condition."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 429:
        return True
    txt = str(exc).lower()
    return "rate limit" in txt or "error code: 429" in txt or "status=429" in txt


def _is_bad_request(exc: Exception) -> bool:
    """Return True for provider-side HTTP 400 bad-request failures."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 400:
        return True
    txt = str(exc).lower()
    return "error code: 400" in txt or "status=400" in txt or "badrequest" in txt


async def analyze_with_tools(
    *,
    cerebras: CerebrasService | None,
    groq: GroqService | None,
    tool_executor: ToolExecutor,
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    max_iterations: int = 10,
) -> ToolRunResult:
    """
    Returns reasoning + decisions (JSON array) using Cerebras tool calling when available.
    Falls back to Groq if Cerebras is unavailable/fails.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    # region agent log
    debug_probe(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="app/services/llm/tool_calling.py:196",
        message="analyze_with_tools entry provider flags",
        data={
            "has_cerebras": bool(cerebras),
            "has_groq": bool(groq),
            "tool_count": len(tools),
        },
    )
    # endregion

    cerebras_failed = False
    # ── Cerebras primary ─────────────────────────────────────────────
    if cerebras:
        t0 = time.perf_counter()
        initial_messages_chars, tools_chars = _estimate_payload_chars(messages, tools)
        try:
            for i in range(max_iterations):
                # region agent log
                debug_probe(
                    run_id="pre-fix",
                    hypothesis_id="H1",
                    location="app/services/llm/tool_calling.py:212",
                    message="cerebras request attempt",
                    data={"model": cerebras.model, "iter": i + 1},
                )
                # endregion
                completion = await cerebras.create_chat_completion(
                    messages=_compact_chat_messages(messages),
                    tools=tools,
                    tool_choice="auto",
                )
                msg = ((completion.get("choices") or [{}])[0].get("message") or {})
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                }
                if msg.get("tool_calls"):
                    assistant_msg["tool_calls"] = msg.get("tool_calls")
                messages.append(assistant_msg)

                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        fn_name = fn.get("name") or ""
                        fn_args = _coerce_tool_args(fn.get("arguments"))
                        tool_out = await tool_executor.execute(fn_name, fn_args)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id"),
                                "name": fn_name,
                                "content": _truncate_tool_content(tool_out),
                            }
                        )
                    continue

                # Final
                text = str(msg.get("content") or "").strip()
                reasoning, decisions = _extract_json_array(text)
                # region agent log
                debug_probe(
                    run_id="pre-fix",
                    hypothesis_id="H4",
                    location="app/services/llm/tool_calling.py:248",
                    message="cerebras final parse",
                    data={"decisions_count": len(decisions)},
                )
                # endregion
                elapsed = time.perf_counter() - t0
                log.info("Cerebras tool-run OK | iters=%d | %.1fs", i + 1, elapsed)
                return ToolRunResult(
                    reasoning_text=reasoning,
                    decisions=decisions,
                    model=cerebras.model,
                    iterations=i + 1,
                )

            return ToolRunResult(
                reasoning_text="",
                decisions=[],
                model=cerebras.model,
                iterations=max_iterations,
            )
        except Exception as exc:
            cerebras_failed = True
            # region agent log
            debug_probe(
                run_id="pre-fix",
                hypothesis_id="H1",
                location="app/services/llm/tool_calling.py:268",
                message="cerebras exception",
                data={"error": str(exc)[:220]},
            )
            # endregion
            final_messages_chars, _ = _estimate_payload_chars(messages, tools)
            soft_fail = _is_rate_limited(exc) or _is_bad_request(exc)
            fallback_target = "Groq" if bool(groq) else "local prepass / no-trade guard"
            log.warning(
                "Cerebras analyze_with_tools failed; fallback=%s | model=%s iters=%d "
                "msg_chars_initial=%d msg_chars_final=%d tools_chars=%d tool_count=%d "
                "soft_fail=%s | %s",
                fallback_target,
                cerebras.model,
                len(messages),
                initial_messages_chars,
                final_messages_chars,
                tools_chars,
                len(tools),
                soft_fail,
                _error_detail(exc),
            )
            if not soft_fail:
                await fire_operator_alert(
                    category="LLM · Cerebras",
                    summary=f"analyze_with_tools: Cerebras failed — fallback to {fallback_target}.",
                    detail=format_exc_brief(exc),
                    dedupe_key="llm_cerebras_tool_fail",
                )
            if soft_fail:
                # region agent log
                debug_probe(
                    run_id="pre-fix",
                    hypothesis_id="H2",
                    location="app/services/llm/tool_calling.py:312",
                    message="cerebras safe no-trade return",
                    data={
                        "reason": "rate_limited_or_bad_request",
                        "groq_available": bool(groq),
                    },
                )
                # endregion
                return ToolRunResult(
                    reasoning_text=(
                        "Cerebras is temporarily unavailable (rate-limit or bad-request). "
                        "Returning no-trade output to keep the agent loop alive."
                    ),
                    decisions=[],
                    model=cerebras.model,
                    iterations=len(messages),
                )

    # ── Groq fallback ────────────────────────────────────────────────
    if groq:
        t0 = time.perf_counter()
        initial_messages_chars, tools_chars = _estimate_payload_chars(messages, tools)
        try:
            client = groq._get_client()  # reuse lazy client
            for i in range(max_iterations):
                completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=groq.model,
                    messages=_compact_chat_messages(messages),
                    tools=tools,
                    tool_choice="auto",
                )
                msg = completion.choices[0].message
                messages.append(_serialize_assistant_message(msg))

                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    for tc in tool_calls:
                        fn = getattr(tc, "function", None)
                        fn_name = getattr(fn, "name", "") if fn else ""
                        fn_args = _coerce_tool_args(getattr(fn, "arguments", None) if fn else None)
                        tool_out = await tool_executor.execute(fn_name, fn_args)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": getattr(tc, "id", None),
                                "name": fn_name,
                                "content": _truncate_tool_content(tool_out),
                            }
                        )
                    continue

                # Final
                text = (msg.content or "").strip()
                reasoning, decisions = _extract_json_array(text)
                # region agent log
                debug_probe(
                    run_id="pre-fix",
                    hypothesis_id="H4",
                    location="app/services/llm/tool_calling.py:356",
                    message="groq final parse",
                    data={"decisions_count": len(decisions)},
                )
                # endregion
                elapsed = time.perf_counter() - t0
                log.info("Groq tool-run OK | iters=%d | %.1fs", i + 1, elapsed)
                return ToolRunResult(
                    reasoning_text=reasoning,
                    decisions=decisions,
                    model=groq.model,
                    iterations=i + 1,
                )

            return ToolRunResult(
                reasoning_text="",
                decisions=[],
                model=groq.model,
                iterations=max_iterations,
            )
        except Exception as exc:
            final_messages_chars, _ = _estimate_payload_chars(messages, tools)
            log.warning(
                "Groq analyze_with_tools failed | model=%s iters=%d "
                "msg_chars_initial=%d msg_chars_final=%d tools_chars=%d tool_count=%d | %s",
                groq.model,
                len(messages),
                initial_messages_chars,
                final_messages_chars,
                tools_chars,
                len(tools),
                _error_detail(exc),
            )
            await fire_operator_alert(
                category="LLM · Groq",
                summary="analyze_with_tools: Groq failed after Cerebras miss.",
                detail=format_exc_brief(exc),
                dedupe_key="llm_groq_tool_fail",
            )
            if _is_rate_limited(exc) or _is_bad_request(exc):
                # region agent log
                debug_probe(
                    run_id="pre-fix",
                    hypothesis_id="H2",
                    location="app/services/llm/tool_calling.py:404",
                    message="groq safe no-trade return",
                    data={"reason": "rate_limited_or_bad_request"},
                )
                # endregion
                return ToolRunResult(
                    reasoning_text=(
                        "Groq is temporarily unavailable (rate-limit or bad-request). "
                        "Returning no-trade output to keep the agent loop alive."
                    ),
                    decisions=[],
                    model=groq.model,
                    iterations=len(messages),
                )

    if not cerebras and not groq:
        # region agent log
        debug_probe(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="app/services/llm/tool_calling.py:416",
            message="no providers configured",
            data={},
        )
        # endregion
        await fire_operator_alert(
            category="LLM",
            summary="analyze_with_tools: Cerebras and Groq are both unavailable (not configured).",
            dedupe_key="llm_no_providers",
        )
    elif cerebras_failed and not groq:
        # region agent log
        debug_probe(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="app/services/llm/tool_calling.py:422",
            message="cerebras failed and groq disabled_or_missing",
            data={},
        )
        # endregion
        # Cerebras failure was already alerted above; Groq missing — no second ping.
        pass

    return ToolRunResult(
        reasoning_text=(
            "No LLM available (Cerebras/Groq offline). "
            "Returning empty decisions to keep the agent loop alive."
        ),
        decisions=[],
        model="none",
        iterations=0,
    )
