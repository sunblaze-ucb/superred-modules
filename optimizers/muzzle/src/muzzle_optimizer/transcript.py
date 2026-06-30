"""Reconstruct a MUZZLE-style transcript from a superred trajectory snapshot.

MUZZLE (arXiv 2602.09222) drives its PAIR sub-agent from a *transcript template*:
the recorded sequence of messages the victim agent exchanged with its model, with
a probe marker (``[INSTRUCTION_PLACEHOLDER]``) planted where an injected
instruction landed. Upstream captures this by proxying the victim's vLLM traffic
(``utils/network/vllm_proxy.py``: a list of request/response records whose
``request.json`` holds OpenAI-style messages) and then scans it for the marker
(``agents/pair.py:_load_transcript_template``).

superred has no network proxy; the equivalent record is the run ``Trajectory``.
This module turns ``current_trajectory.snapshot()`` items back into the ordered,
role-tagged message list MUZZLE's Summarizer / Judge / PAIR expect, and ports the
marker-localisation logic to that shape.

The reconstruction is steering input only, so the builders are permissive and
never raise; only :func:`extract_placeholder_template` raises, and only the
dedicated :class:`PlaceholderNotFoundError` when no marker is present.
"""

from __future__ import annotations

import json
import re
from typing import Any

from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ObservableEvent,
    RunEndEvent,
)
from superred.core.types.observable import ObservableValue

# Observable-name substrings that mark a runtime item as model/agent output (an
# assistant turn). ``model_output`` / ``trace_message`` / ``plan`` are the names
# the agentic superred targets emit; the rest are defensive synonyms because the
# optimizer must run against arbitrary, unseen observable names.
_ASSISTANT_NAME_HINTS = (
    "model_output",
    "trace_message",
    "plan",
    "thought",
    "reasoning",
    "completion",
    "response",
    "output",
    "action",
)
# Observable-name substrings that mark the benign user task / request.
_USER_NAME_HINTS = ("user", "task", "query")
# Observable-name substring that marks a navigable action URL.
_URL_NAME_HINT = "url"

# Static observable name (case-insensitive) carrying the victim system prompt.
_SYSTEM_PROMPT_NAME = "system_prompt"

# Empty response-name set for the internal transcript builds in the recovery
# helpers (they only inspect system / user roles, never the response role).
_NO_RESPONSE_NAMES: frozenset[str] = frozenset()

# Matches an ``http(s)://`` URL up to the first whitespace; trailing punctuation
# is trimmed by ``_find_url`` so a URL embedded in prose is recovered cleanly.
_URL_RE = re.compile(r"https?://\S+")
_URL_TRAILING = "\"'<>).,;:]}"


class PlaceholderNotFoundError(Exception):
    """Raised when no placeholder token is present in a transcript template."""


def _stringify(content: Any) -> str:
    """Best-effort string form of arbitrary observable / controllable content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(content)


def _observable_role(name: str, response_observable_names: frozenset[str]) -> str:
    """Map an observable name to an OpenAI-style transcript role.

    ``system`` for any system-named observable, ``assistant`` for model/agent
    output (the configured response names plus model-output synonyms), ``user``
    for the benign task, and ``assistant`` as the default for unclassified
    runtime activity.
    """
    lowered = name.lower()
    if "system" in lowered:
        return "system"
    if name in response_observable_names or lowered in response_observable_names:
        return "assistant"
    if any(hint in lowered for hint in _ASSISTANT_NAME_HINTS):
        return "assistant"
    if any(hint in lowered for hint in _USER_NAME_HINTS):
        return "user"
    return "assistant"


def _message_text(message: dict[str, Any]) -> str:
    """Primary text of a reconstructed transcript message.

    Plain messages carry ``content``; tool records carry ``request`` / ``answer``.
    """
    if "content" in message:
        return _stringify(message["content"])
    parts = [_stringify(message[key]) for key in ("answer", "request") if message.get(key)]
    return "\n".join(part for part in parts if part)


def _find_url(text: str) -> str | None:
    """First ``http(s)://`` URL in ``text`` (trailing punctuation trimmed), else ``None``."""
    match = _URL_RE.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(_URL_TRAILING) or None


def build_transcript(
    items: list[object],
    *,
    response_observable_names: frozenset[str],
) -> list[dict[str, Any]]:
    """Reconstruct an ordered MUZZLE-style transcript from trajectory items.

    ``items`` is ``current_trajectory.snapshot()`` (``Event | EventResponse``).
    Each relevant item becomes one message, in emission order:

    * :class:`ObservableEvent` -> ``{"role": ..., "content": <content as-is>}``;
      the role is classified from the observable name (system / assistant / user)
      and the content is preserved verbatim (str or json) for the Summarizer.
    * :class:`ControllablePostCallEvent` -> a tool record
      ``{"role": "tool", "tool": name, "request": ..., "answer": ...}``.
    * :class:`ControllableInjection` -> the attacker-delivered value as a tool
      record so a planted probe marker is always recoverable, whether it was
      injected at a pre-call or a post-call surface.

    Lifecycle items (:class:`RunEndEvent`, run-start, pre-call events, decline
    responses) are not transcript content and are skipped. Never raises.
    """
    transcript: list[dict[str, Any]] = []
    for item in items:
        record: dict[str, Any] | None = None
        if isinstance(item, ObservableEvent):
            record = {
                "role": _observable_role(item.observable.name, response_observable_names),
                "content": item.content,
            }
        elif isinstance(item, ControllablePostCallEvent):
            record = {
                "role": "tool",
                "tool": item.controllable.name,
                "request": item.request,
                "answer": item.answer,
            }
        elif isinstance(item, ControllableInjection):
            record = {
                "role": "tool",
                "tool": item.controllable.name,
                "request": "",
                "answer": item.value,
            }
        elif isinstance(item, RunEndEvent):
            # Run-end feedback is consumed by the optimizer directly; it is not
            # part of the reconstructed victim transcript.
            record = None
        if record is not None:
            transcript.append(record)
    return transcript


def find_victim_system_prompt(
    items: list[object],
    observables: list[ObservableValue],
) -> str | None:
    """Recover the victim agent's system prompt.

    Prefers an in-scope static observable named ``system_prompt``; otherwise the
    first system-role :class:`ObservableEvent` in ``items``; otherwise ``None``.
    PAIR uses this to align its simulated target's system prompt with the real
    victim (upstream ``pair.py`` overrides ``targetLM.model.system_prompt``).
    """
    for value in observables:
        if value.observable.name.lower() == _SYSTEM_PROMPT_NAME:
            text = _stringify(value.content)
            if text.strip():
                return text
    for item in items:
        if isinstance(item, ObservableEvent) and "system" in item.observable.name.lower():
            text = _stringify(item.content)
            if text.strip():
                return text
    return None


def recover_user_goal(
    items: list[object],
    observables: list[ObservableValue],
) -> str | None:
    """Recover the benign user task the victim agent was given.

    Prefers an in-scope static observable named like ``user`` / ``user_prompt`` /
    ``task`` with string content; otherwise the first user-role message in the
    reconstructed transcript; otherwise ``None``. Drives the injection-template
    family selection (``goal_hijacking`` when a benign goal is known).
    """
    for value in observables:
        lowered = value.observable.name.lower()
        if any(hint in lowered for hint in _USER_NAME_HINTS):
            if isinstance(value.content, str) and value.content.strip():
                return value.content
    transcript = build_transcript(items, response_observable_names=_NO_RESPONSE_NAMES)
    for message in transcript:
        if message.get("role") == "user":
            text = _message_text(message)
            if text.strip():
                return text
    return None


def recover_action_url(
    items: list[object],
    observables: list[ObservableValue],
) -> str | None:
    """Recover a navigable action URL the injection can point the agent to.

    Prefers an in-scope static observable named like ``url`` / ``start_url`` /
    ``target_url`` whose content holds an ``http(s)`` URL; otherwise the first
    ``http(s)`` URL in a system- or user-role transcript message; otherwise
    ``None``. Drives the injection-template form selection (``url_injection``
    only when a URL is available).
    """
    for value in observables:
        if _URL_NAME_HINT in value.observable.name.lower():
            found = _find_url(_stringify(value.content))
            if found is not None:
                return found
    transcript = build_transcript(items, response_observable_names=_NO_RESPONSE_NAMES)
    for message in transcript:
        if message.get("role") in {"system", "user"}:
            found = _find_url(_message_text(message))
            if found is not None:
                return found
    return None


def _extract_texts(content: Any) -> list[str]:
    """Extract text from ``str | list | dict`` content.

    Behavioral port of upstream ``pair.py:_load_transcript_template._extract_texts``
    (gsiros/muzzle SHA ed611c0): a bare string is itself; a list yields its string
    items plus the ``text`` / recursive ``content`` fields of any dict items; a
    dict yields its ``text`` plus its recursive ``content``.
    """
    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    texts.append(item["text"])
                if "content" in item:
                    texts.extend(_extract_texts(item["content"]))
    elif isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            texts.append(content["text"])
        if "content" in content:
            texts.extend(_extract_texts(content["content"]))
    return [text for text in texts if isinstance(text, str)]


def _candidate_texts(message: dict[str, Any]) -> list[str]:
    """All searchable text fragments of a reconstructed transcript message."""
    if "content" in message:
        return _extract_texts(message["content"])
    texts: list[str] = []
    if "answer" in message:
        texts.extend(_extract_texts(message["answer"]))
    if "request" in message:
        texts.extend(_extract_texts(message["request"]))
    return texts


def _system_text(message: dict[str, Any]) -> str | None:
    """Stringified content of a system-role message, or ``None`` when empty."""
    content = message.get("content")
    if content is None:
        return None
    texts = _extract_texts(content)
    candidate = "\n".join(texts) if texts else _stringify(content)
    return candidate if candidate.strip() else None


def extract_placeholder_template(
    transcript: list[dict[str, Any]],
    placeholder_tokens: tuple[str, ...],
) -> tuple[str, str | None]:
    """Locate the planted probe marker and its accompanying victim system prompt.

    Port of upstream ``pair.py:_load_transcript_template`` to the transcript shape
    produced by :func:`build_transcript`. Scans messages in emission order; the
    first text fragment containing any token in ``placeholder_tokens`` becomes
    ``step_text`` -- the template into which PAIR splices each candidate, replacing
    the token. The most recently seen *prior* system-role content is returned as
    the accompanying system prompt (the upstream "system prompt sent in that same
    request"), or ``None`` when no system message preceded the marker.

    Args:
        transcript: A transcript as built by :func:`build_transcript`.
        placeholder_tokens: Marker tokens to search for, e.g. the upstream
            ``[INSTRUCTION_PLACEHOLDER]`` and its ``[PLACEHOLDER]`` alias.

    Returns:
        ``(step_text, accompanying_system_prompt)``.

    Raises:
        PlaceholderNotFoundError: if no token occurs anywhere in the transcript.
    """
    most_recent_system: str | None = None
    for message in transcript:
        for text in _candidate_texts(message):
            if any(token in text for token in placeholder_tokens):
                return text, most_recent_system
        if message.get("role") == "system":
            system_text = _system_text(message)
            if system_text is not None:
                most_recent_system = system_text
    raise PlaceholderNotFoundError(
        "No transcript step containing a placeholder token "
        f"({', '.join(placeholder_tokens)}) found in transcript template."
    )


__all__ = [
    "PlaceholderNotFoundError",
    "build_transcript",
    "extract_placeholder_template",
    "find_victim_system_prompt",
    "recover_action_url",
    "recover_user_goal",
]
