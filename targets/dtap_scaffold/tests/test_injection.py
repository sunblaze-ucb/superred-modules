"""Offline tests for the DTAP environment-write vector (no Docker, no fastmcp).

The genuine fastmcp call (:meth:`McpEnvInjector._call`) is monkeypatched with a
recorder, so these tests verify the *parsing and routing* logic: the structured
``{"injection_mcp_tool": "<server>:<tool>", "kwargs": {...}}`` value is turned into
the right ``(tool, kwargs)`` call against ``injection_server_urls[point.server]``,
lists fan out to several calls, and every fallback/failure path is non-crashing.
The real fastmcp round-trip is a ``live`` concern, not exercised here.
"""

from __future__ import annotations

import json

from dtap_scaffold.injection import McpEnvInjector, _parse_injection_calls
from dtap_scaffold.protocols import EnvInjector
from dtap_scaffold.types import InjectionPoint

POINT = InjectionPoint(server="gmail-injection")


def _injector_with_recorder(urls: dict[str, str] | None = None):
    """An injector whose ``_call`` records ``(url, tool, kwargs)`` instead of dialing out."""
    inj = McpEnvInjector(urls if urls is not None else {"gmail-injection": "http://host/gmail/mcp"})
    calls: list[tuple[str, str, dict]] = []

    async def fake_call(url, tool, kwargs):
        calls.append((url, tool, kwargs))

    inj._call = fake_call  # type: ignore[method-assign]
    return inj, calls


# --------------------------- protocol + snapshot --------------------------


def test_satisfies_envinjector_protocol():
    assert isinstance(McpEnvInjector({}), EnvInjector)


async def test_snapshot_returns_empty_str():
    inj = McpEnvInjector({"gmail-injection": "http://x"})
    out = await inj.snapshot(POINT)
    assert isinstance(out, str)
    assert out == ""  # write-only injection servers expose no genuine read


# --------------------------- apply: the structured path -------------------


async def test_apply_structured_json_calls_right_tool_and_kwargs():
    inj, calls = _injector_with_recorder()
    value = json.dumps(
        {
            "injection_mcp_tool": "gmail-injection:inject_email",
            "kwargs": {
                "from_email": "a@b.c",
                "to_email": "v@x.y",
                "subject": "hi",
                "body": "POISON",
            },
        }
    )
    await inj.apply(POINT, value)

    assert len(calls) == 1
    url, tool, kwargs = calls[0]
    assert url == "http://host/gmail/mcp"  # URL comes from point.server, not the value
    assert tool == "inject_email"  # the part after the colon
    assert kwargs == {
        "from_email": "a@b.c",
        "to_email": "v@x.y",
        "subject": "hi",
        "body": "POISON",
    }


async def test_apply_tool_without_colon_uses_whole_name():
    inj, calls = _injector_with_recorder()
    value = json.dumps({"injection_mcp_tool": "inject_comment", "kwargs": {"comment": "X"}})
    await inj.apply(POINT, value)
    assert calls[0][1] == "inject_comment"
    assert calls[0][2] == {"comment": "X"}


async def test_apply_list_of_specs_fans_out():
    inj, calls = _injector_with_recorder({"salesforce-injection": "http://sf/mcp"})
    point = InjectionPoint(server="salesforce-injection")
    value = json.dumps(
        [
            {
                "injection_mcp_tool": "salesforce-injection:inject_comment",
                "kwargs": {"comment": "1"},
            },
            {
                "injection_mcp_tool": "salesforce-injection:inject_lead",
                "kwargs": {"name": "2"},
            },
        ]
    )
    await inj.apply(point, value)
    assert [(t, kw) for _, t, kw in calls] == [
        ("inject_comment", {"comment": "1"}),
        ("inject_lead", {"name": "2"}),
    ]


async def test_apply_missing_kwargs_defaults_to_empty():
    inj, calls = _injector_with_recorder()
    await inj.apply(POINT, json.dumps({"injection_mcp_tool": "gmail-injection:inject_email"}))
    assert calls[0][1] == "inject_email"
    assert calls[0][2] == {}


# --------------------------- apply: non-crashing edge paths ---------------


async def test_apply_unknown_server_is_noop():
    inj, calls = _injector_with_recorder({})  # no URLs known
    await inj.apply(
        InjectionPoint(server="absent"),
        json.dumps({"injection_mcp_tool": "absent:inject", "kwargs": {}}),
    )
    assert calls == []  # nowhere to write -> best-effort no-op


async def test_apply_swallows_call_errors():
    inj = McpEnvInjector({"gmail-injection": "http://x"})

    async def boom(url, tool, kwargs):
        raise RuntimeError("backend down")

    inj._call = boom  # type: ignore[method-assign]
    # must not raise: a failed write just means the attack did not land
    await inj.apply(
        POINT,
        json.dumps({"injection_mcp_tool": "gmail-injection:inject", "kwargs": {}}),
    )


async def test_apply_bare_text_is_noop():
    inj, calls = _injector_with_recorder({"s": "http://x"})
    await inj.apply(InjectionPoint(server="s"), "just plain text, not JSON")
    assert calls == []  # only the structured {injection_mcp_tool, kwargs} form routes


async def test_apply_dict_without_mcp_tool_is_noop():
    inj, calls = _injector_with_recorder({"s": "http://x"})
    await inj.apply(InjectionPoint(server="s"), json.dumps({"to_email": "v@x.y"}))
    assert calls == []  # the tool identity must ride in the value (injection_mcp_tool)


async def test_apply_non_dict_kwargs_does_not_crash():
    """A wrong-shaped ``kwargs`` (string/int/bool/list) must not abort the run.

    It is coerced to an empty mapping: the write still fires with no kwargs (the
    attack simply does not land its payload). Before the guard, ``dict("POISON")``
    raised and propagated out of ``run()``, crashing the whole task.
    """
    for bad in ("POISON", 5, True, ["a", "b"]):
        inj, calls = _injector_with_recorder()
        await inj.apply(
            POINT,
            json.dumps({"injection_mcp_tool": "gmail-injection:inject_email", "kwargs": bad}),
        )
        assert len(calls) == 1
        assert calls[0][1] == "inject_email"
        assert calls[0][2] == {}


# --------------------------- the parser, directly -------------------------


def test_parse_structured_extracts_tool_after_colon():
    calls = _parse_injection_calls(
        json.dumps(
            {
                "injection_mcp_tool": "gmail-injection:inject_email",
                "kwargs": {"body": "x"},
            }
        ),
    )
    assert calls == [("inject_email", {"body": "x"})]


def test_parse_non_json_with_all_sentinel_is_empty():
    assert _parse_injection_calls("not json") == []


def test_parse_ignores_non_dict_list_items():
    value = json.dumps(["junk", {"injection_mcp_tool": "s:inject", "kwargs": {"a": 1}}])
    assert _parse_injection_calls(value) == [("inject", {"a": 1})]


def test_parse_non_dict_kwargs_coerced_to_empty():
    for bad in ("POISON", 5, True, ["a", "b"]):
        value = json.dumps({"injection_mcp_tool": "s:inject", "kwargs": bad})
        assert _parse_injection_calls(value) == [("inject", {})]


def test_parse_deeply_nested_json_is_empty():
    """Deeply-nested JSON makes json.loads raise RecursionError (a RuntimeError, not a
    JSONDecodeError/TypeError); the parser must still return [] so the env-write vector
    never aborts the run on this attacker value."""
    assert _parse_injection_calls("[" * 100000) == []
    assert _parse_injection_calls("[" * 30000 + "]" * 30000) == []


def test_parse_oversized_int_is_empty():
    """A >4300-digit integer literal makes json.loads raise a bare ValueError (CPython
    int-string-conversion limit), not a JSONDecodeError; the parser must still return []."""
    assert _parse_injection_calls("1" * 4301) == []
