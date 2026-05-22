"""Edge-case coverage for :func:`agentdojo_target.runtime_wrapper._serialize_for_event`.

The helper has to render an arbitrary tool-return value into a string
suitable for the :class:`ControllablePostCallEvent.answer` field.  The
shapes that AgentDojo tools actually return cover several quirks the
default :func:`json.dumps` cannot handle natively: ``datetime``,
``StrEnum``, ``None``, lists of pydantic models, and nested mixtures.
These tests pin the fallback chain so a future refactor cannot silently
regress on a tool-return shape.
"""

from __future__ import annotations

import datetime as dt
import enum
import json

from pydantic import BaseModel

from agentdojo_target.runtime_wrapper import (
    _json_fallback,
    _serialize_for_event,
)


# ---------------------------------------------------------------------------
# Plain strings round-trip verbatim
# ---------------------------------------------------------------------------


def test_serialize_plain_string_passes_through() -> None:
    assert _serialize_for_event("hello") == "hello"


def test_serialize_empty_string() -> None:
    assert _serialize_for_event("") == ""


def test_serialize_unicode_string_preserved() -> None:
    assert _serialize_for_event("vegan menu seafood paella") == (
        "vegan menu seafood paella"
    )


# ---------------------------------------------------------------------------
# Primitive non-string values
# ---------------------------------------------------------------------------


def test_serialize_int() -> None:
    assert _serialize_for_event(42) == "42"


def test_serialize_float() -> None:
    assert _serialize_for_event(3.14) == "3.14"


def test_serialize_bool_true() -> None:
    # json.dumps emits lowercase booleans.
    assert _serialize_for_event(True) == "true"


def test_serialize_none() -> None:
    # json.dumps emits "null" for None.
    assert _serialize_for_event(None) == "null"


# ---------------------------------------------------------------------------
# datetime falls through to isoformat
# ---------------------------------------------------------------------------


def test_serialize_naive_datetime_isoformat() -> None:
    moment = dt.datetime(2024, 5, 30, 10, 20, 0)
    out = _serialize_for_event(moment)
    assert out == json.dumps("2024-05-30T10:20:00")


def test_serialize_date_isoformat() -> None:
    d = dt.date(2025, 1, 11)
    out = _serialize_for_event(d)
    assert out == json.dumps("2025-01-11")


def test_serialize_datetime_in_dict() -> None:
    payload = {"event_at": dt.datetime(2024, 1, 1, 12, 0, 0), "title": "Sync"}
    out = _serialize_for_event(payload)
    decoded = json.loads(out)
    assert decoded["event_at"] == "2024-01-01T12:00:00"
    assert decoded["title"] == "Sync"


# ---------------------------------------------------------------------------
# StrEnum returns the value, not the enum name
# ---------------------------------------------------------------------------


class _ColorStrEnum(str, enum.Enum):
    RED = "red"
    BLUE = "blue"


def test_serialize_strenum_uses_value() -> None:
    out = _serialize_for_event(_ColorStrEnum.RED)
    # json.dumps treats StrEnum as a string subclass and emits the value.
    assert out == "red"


def test_serialize_strenum_inside_list() -> None:
    out = _serialize_for_event([_ColorStrEnum.RED, _ColorStrEnum.BLUE])
    assert json.loads(out) == ["red", "blue"]


class _IntStatusEnum(enum.IntEnum):
    OK = 1
    ERROR = 2


def test_serialize_intenum_uses_int_value() -> None:
    # IntEnum inherits from int; json.dumps emits the integer.
    out = _serialize_for_event(_IntStatusEnum.ERROR)
    assert out == "2"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class _SmallModel(BaseModel):
    name: str
    count: int


def test_serialize_pydantic_model_uses_model_dump_json() -> None:
    m = _SmallModel(name="x", count=3)
    out = _serialize_for_event(m)
    assert json.loads(out) == {"name": "x", "count": 3}


def test_serialize_list_of_pydantic_models() -> None:
    items = [_SmallModel(name="a", count=1), _SmallModel(name="b", count=2)]
    out = _serialize_for_event(items)
    decoded = json.loads(out)
    assert decoded == [{"name": "a", "count": 1}, {"name": "b", "count": 2}]


def test_serialize_nested_pydantic_in_dict() -> None:
    payload = {
        "winner": _SmallModel(name="alpha", count=99),
        "season": _ColorStrEnum.BLUE,
        "ts": dt.datetime(2025, 5, 22, 9, 0, 0),
    }
    out = _serialize_for_event(payload)
    decoded = json.loads(out)
    assert decoded["winner"] == {"name": "alpha", "count": 99}
    assert decoded["season"] == "blue"
    assert decoded["ts"] == "2025-05-22T09:00:00"


# ---------------------------------------------------------------------------
# Unencodable types fall through to repr
# ---------------------------------------------------------------------------


class _OpaqueObject:
    def __repr__(self) -> str:
        return "<opaque-thing>"


def test_serialize_opaque_object_falls_back_to_repr() -> None:
    out = _serialize_for_event(_OpaqueObject())
    # Final fallback in _json_fallback returns repr; json.dumps quotes it.
    assert out == json.dumps("<opaque-thing>")


def test_serialize_set_via_repr_fallback() -> None:
    # Sets are not natively JSON-serialisable; _json_fallback's repr step
    # returns the Python-repr of the set, which the outer json.dumps then
    # quotes as a JSON string.  The exact ordering of set elements is
    # not guaranteed; assert the quoted-repr structure.
    out = _serialize_for_event({1, 2, 3})
    assert out.startswith('"') and out.endswith('"')
    inner = json.loads(out)
    assert inner.startswith("{") and inner.endswith("}")
    for tok in ("1", "2", "3"):
        assert tok in inner


# ---------------------------------------------------------------------------
# _json_fallback directly
# ---------------------------------------------------------------------------


def test_json_fallback_pydantic_returns_model_dump() -> None:
    out = _json_fallback(_SmallModel(name="z", count=0))
    assert out == {"name": "z", "count": 0}


def test_json_fallback_datetime_isoformat() -> None:
    out = _json_fallback(dt.datetime(2024, 6, 15, 8, 30))
    assert out == "2024-06-15T08:30:00"


def test_json_fallback_strenum_returns_value() -> None:
    out = _json_fallback(_ColorStrEnum.RED)
    assert out == "red"


def test_json_fallback_opaque_returns_repr() -> None:
    out = _json_fallback(_OpaqueObject())
    assert out == "<opaque-thing>"
