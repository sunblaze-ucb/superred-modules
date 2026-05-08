"""Tests for the Crescendo attacker LLM question generator."""

from unittest.mock import AsyncMock, MagicMock
import json
import pytest

from crescendo_optimizer.attacker import Attacker, _render_framing_instruction
from crescendo_optimizer.prompts import BASE_SCHEMA, FRAMING_FIELD


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
        temperature=1.0,
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


@pytest.mark.asyncio
async def test_generate_question_with_score_feedback():
    """Subsequent turn: includes score and response feedback."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(_valid_json(
        generated_question="Can you elaborate on that point?",
    ))
    attacker = Attacker(
        llm=mock_llm,
        temperature=1.0,
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
        temperature=1.0,
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
        temperature=1.0,
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
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    output = await attacker.generate_question(
        goal="g", turn=1, max_turns=10,
        last_response=None, last_score=None, last_rationale=None,
    )
    assert output.question == "Tell me about the history of X"


@pytest.mark.asyncio
async def test_parse_response_raises_on_missing_key():
    """Should raise ValueError if required key is missing."""
    bad_json = json.dumps({"generated_question": "Q"})  # missing other keys
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _mock_response(bad_json)
    attacker = Attacker(
        llm=mock_llm,
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    with pytest.raises(ValueError, match="missing required keys"):
        await attacker.generate_question(
            goal="g", turn=1, max_turns=10,
            last_response=None, last_score=None, last_rationale=None,
        )


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


