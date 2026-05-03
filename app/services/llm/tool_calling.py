"""
LLM tool-calling runner (Groq primary, Ollama fallback).

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
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.services.llm.groq_service import GroqService, LLMResponse
from app.services.llm.ollama_service import OllamaService
from app.tools.executor import ToolExecutor

log = get_logger("llm.tool_calling")


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


async def analyze_with_tools(
    *,
    groq: GroqService | None,
    ollama: OllamaService | None,
    tool_executor: ToolExecutor,
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    max_iterations: int = 10,
) -> ToolRunResult:
    """
    Returns reasoning + decisions (JSON array) using Groq tool calling when available.
    Falls back to Ollama (no tools) if Groq is unavailable/fails.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    # ── Groq primary ────────────────────────────────────────────────
    if groq:
        t0 = time.perf_counter()
        try:
            client = groq._get_client()  # reuse lazy client
            for i in range(max_iterations):
                completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=groq.model,
                    messages=messages,
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
                                "content": tool_out,
                            }
                        )
                    continue

                # Final
                text = (msg.content or "").strip()
                reasoning, decisions = _extract_json_array(text)
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
            log.warning("Groq analyze_with_tools failed; falling back to Ollama: %s", exc)

    # ── Ollama fallback (no tool calling in this repo yet) ──────────
    if ollama:
        prompt = user_message
        # Put system prompt on top to preserve instruction ordering.
        system = system_prompt or None
        resp: LLMResponse = await ollama.analyze(prompt, system=system)
        reasoning, decisions = _extract_json_array(resp.text)
        return ToolRunResult(
            reasoning_text=reasoning,
            decisions=decisions,
            model=resp.model,
            iterations=1,
        )

    return ToolRunResult(
        reasoning_text="ERROR: No LLM available (Groq disabled and Ollama not configured).",
        decisions=[],
        model="none",
        iterations=0,
    )

