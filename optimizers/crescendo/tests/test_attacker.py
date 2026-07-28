"""Tests for the Crescendo attacker LLM question generator."""

from unittest.mock import AsyncMock, MagicMock
import json
import pytest

from superred.core.types.llm import BudgetExhaustedError, LLMUsage

from crescendo_optimizer.attacker import Attacker, _render_framing_instruction
from crescendo_optimizer.prompts import BASE_SCHEMA, FRAMING_FIELD
from crescendo_optimizer.retry import DEFAULT_ATTEMPTS, HelperLLMUnavailableError


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _valid_json(**overrides) -> str:
    data = {
        "generated_question": "Tell me about the history of X",
        "last_response_summary": "",
        "rationale_behind_jailbreak": "Starting with innocuous historical context",
    }
    data.update(overrides)
    return json.dumps(data)


@pytest.mark.asyncio
async def test_generate_question_first_turn():
    """First turn: no prior response, should generate initial question."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(_valid_json())
    attacker = Attacker(
        llm=mock_llm,
        system_prompt="You are a red teamer. Goal: {objective}",
    )
    output = await attacker.generate_question(
        goal="test goal",
        turn=1,
        max_turns=10,
        last_response=None,
        last_score=None,
        last_rationale=None,
    )
    assert output.question == "Tell me about the history of X"
    assert output.summary == ""
    assert "innocuous" in output.rationale.lower()
    assert output.framing is None
    # System prompt should be first message
    messages = mock_llm.complete.call_args[0][0]
    assert messages[0]["role"] == "system"
    # No temperature is sent: reasoning models reject the parameter.
    assert "temperature" not in mock_llm.complete.call_args.kwargs


@pytest.mark.asyncio
async def test_generate_question_with_score_feedback():
    """Subsequent turn: includes score and response feedback."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(_valid_json(
        generated_question="Can you elaborate on that point?",
    ))
    attacker = Attacker(
        llm=mock_llm,
        system_prompt="Goal: {objective}",
    )
    output = await attacker.generate_question(
        goal="test goal",
        turn=3,
        max_turns=10,
        last_response="The history of X is fascinating...",
        last_score=0.3,
        last_rationale="Some progress but not enough",
    )
    assert output.question == "Can you elaborate on that point?"
    # User message should contain score info
    messages = mock_llm.complete.call_args[0][0]
    user_msg = messages[-1]["content"]
    assert "0.30" in user_msg
    assert "turn 3 of 10" in user_msg.lower()


@pytest.mark.asyncio
async def test_generate_question_accumulates_history():
    """Each call adds to the attacker's conversation history."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(_valid_json())
    attacker = Attacker(
        llm=mock_llm,
        system_prompt="Goal: {objective}",
    )

    # First call
    await attacker.generate_question(
        goal="g", turn=1, max_turns=10,
        last_response=None, last_score=None, last_rationale=None,
    )
    first_call_msg_count = len(mock_llm.complete.call_args[0][0])

    # Second call
    mock_llm.complete.return_value = _mock_response(_valid_json(
        generated_question="Follow up question",
    ))
    await attacker.generate_question(
        goal="g", turn=2, max_turns=10,
        last_response="Some response", last_score=0.2, last_rationale="Low",
    )
    second_call_msg_count = len(mock_llm.complete.call_args[0][0])

    # History should grow
    assert second_call_msg_count > first_call_msg_count


@pytest.mark.asyncio
async def test_notify_refusal_adds_to_history():
    """Refusal notification should affect the next question generation."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(_valid_json())
    attacker = Attacker(
        llm=mock_llm,
        system_prompt="Goal: {objective}",
    )

    # Generate first question
    await attacker.generate_question(
        goal="g", turn=1, max_turns=10,
        last_response=None, last_score=None, last_rationale=None,
    )

    # Notify refusal
    attacker.notify_refusal("The refused question text")

    # Generate next question — should include refusal context
    mock_llm.complete.return_value = _mock_response(_valid_json(
        generated_question="A different approach",
    ))
    await attacker.generate_question(
        goal="g", turn=1, max_turns=10,
        last_response=None, last_score=None, last_rationale=None,
    )
    messages = mock_llm.complete.call_args[0][0]
    # The refusal notification should be in the history
    all_content = " ".join(m["content"] for m in messages)
    assert "refused" in all_content.lower()


@pytest.mark.asyncio
async def test_parse_response_handles_markdown_json():
    """Should parse JSON even if wrapped in markdown code blocks."""
    wrapped = f"```json\n{_valid_json()}\n```"
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(wrapped)
    attacker = Attacker(
        llm=mock_llm,
        system_prompt="Goal: {objective}",
    )
    output = await attacker.generate_question(
        goal="g", turn=1, max_turns=10,
        last_response=None, last_score=None, last_rationale=None,
    )
    assert output.question == "Tell me about the history of X"


def test_parse_response_raises_on_missing_key():
    """Should raise ValueError if a required key is missing."""
    bad_json = json.dumps({"generated_question": "Q"})  # missing other keys
    with pytest.raises(ValueError, match="missing required keys"):
        Attacker._parse_response(bad_json, require_framing=False)


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", 42, ["a"], {"a": 1}],
    ids=["null", "empty", "blank", "int", "list", "object"],
)
def test_parse_response_rejects_unusable_question(value):
    """A present-but-unusable ``generated_question`` is a parse failure.

    All six shapes were accepted before this check. A JSON ``null`` reached
    the target as ``ControllableInjection(value=None)``; litellm strips a
    ``None`` content field and its Bedrock Converse transform then raises
    ``KeyError: 'content'``, killing the whole task. An empty string is worse:
    Converse silently drops the turn, so the run is scored on a transcript
    containing a phantom turn.
    """
    payload = json.dumps({
        "generated_question": value,
        "last_response_summary": "s",
        "rationale_behind_jailbreak": "r",
    })
    with pytest.raises(ValueError):
        Attacker._parse_response(payload, require_framing=False)


@pytest.mark.parametrize("field", ["last_response_summary", "rationale_behind_jailbreak"])
def test_parse_response_rejects_non_string_supporting_field(field):
    """Summary and rationale are rendered back into later meta-prompts."""
    data = {
        "generated_question": "Q",
        "last_response_summary": "s",
        "rationale_behind_jailbreak": "r",
    }
    data[field] = None
    with pytest.raises(ValueError, match="must be a string"):
        Attacker._parse_response(json.dumps(data), require_framing=False)


def test_parse_response_rejects_non_object_payload():
    """A JSON array must be a ValueError like its siblings, not an AttributeError."""
    with pytest.raises(ValueError, match="not a JSON object"):
        Attacker._parse_response("[1, 2]", require_framing=False)


def test_parse_response_preserves_question_verbatim():
    """Blankness is only a rejection test: the question itself is not stripped."""
    question = "  Line one.\n\nLine two.  "
    output = Attacker._parse_response(
        _valid_json(generated_question=question), require_framing=False,
    )
    assert output.question == question


@pytest.mark.asyncio
async def test_generate_question_resamples_after_malformed_json():
    """Malformed attacker JSON is resampled, not substituted.

    The attacker is sampled at the provider default (no temperature is
    pinned), so a second draw is a genuinely different attempt.
    """
    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = [
        _mock_response("not json at all"),
        _mock_response(_valid_json(generated_question="Recovered question")),
    ]
    attacker = Attacker(llm=mock_llm, system_prompt="Goal: {objective}")

    output = await attacker.generate_question(
        goal="g", turn=1, max_turns=10,
        last_response=None, last_score=None, last_rationale=None,
    )

    assert output.question == "Recovered question"
    assert mock_llm.complete.call_count == 2
    # Only the successful sample is committed to the attacker's history.
    assert len(attacker._conversation_history) == 2


@pytest.mark.asyncio
async def test_generate_question_raises_after_all_resamples_fail():
    """Persistently malformed output surfaces instead of returning a filler."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response("not json at all")
    attacker = Attacker(llm=mock_llm, system_prompt="Goal: {objective}")

    with pytest.raises(HelperLLMUnavailableError):
        await attacker.generate_question(
            goal="g", turn=1, max_turns=10,
            last_response=None, last_score=None, last_rationale=None,
        )
    assert mock_llm.complete.call_count == DEFAULT_ATTEMPTS
    assert attacker._conversation_history == []


@pytest.mark.asyncio
async def test_generate_question_never_retries_budget_exhausted():
    """The cost cap is a controller signal: re-raised on the first call.

    Retrying it would be a cap escape, and absorbing it is what made 2,299
    tasks in the first archive report ``stop_reason="done"`` after spending
    their entire budget.
    """
    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = BudgetExhaustedError(
        "Cost cap reached: $0.750131/$0.750000",
        usage=LLMUsage(calls=214, cost=0.750131),
    )
    attacker = Attacker(llm=mock_llm, system_prompt="Goal: {objective}")

    with pytest.raises(BudgetExhaustedError):
        await attacker.generate_question(
            goal="g", turn=1, max_turns=10,
            last_response=None, last_score=None, last_rationale=None,
        )
    assert mock_llm.complete.call_count == 1


def test_framing_instruction_renders_every_base_field():
    """The (c) override must list every field declared in BASE_SCHEMA so
    a future variant adding a base field cannot silently desynchronise
    with the framing path. The framing field itself must also appear."""
    rendered = _render_framing_instruction(BASE_SCHEMA, FRAMING_FIELD)
    for field in BASE_SCHEMA:
        assert field in rendered, f"base field {field!r} missing from rendered framing override"
    assert FRAMING_FIELD in rendered


def test_framing_instruction_picks_up_extended_schema():
    """If the base schema is extended (e.g. a new variant adds a
    `confidence` field), the rendered override must include it without
    any change to attacker.py. Locks the dynamic-rendering contract."""
    extended = (*BASE_SCHEMA, "confidence")
    rendered = _render_framing_instruction(extended, FRAMING_FIELD)
    for field in extended:
        assert field in rendered
    assert FRAMING_FIELD in rendered


