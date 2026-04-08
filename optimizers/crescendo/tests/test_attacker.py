"""Tests for the Crescendo attacker LLM question generator."""

from unittest.mock import AsyncMock, patch, MagicMock
import json
import pytest

from crescendo_optimizer.attacker import Attacker


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
    attacker = Attacker(
        model="gpt-4",
        temperature=1.0,
        system_prompt="You are a red teamer. Goal: {objective}",
    )
    with patch("crescendo_optimizer.attacker.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response(_valid_json())
        question, summary, rationale = await attacker.generate_question(
            goal="test goal",
            turn=1,
            max_turns=10,
            last_response=None,
            last_score=None,
            last_rationale=None,
        )
        assert question == "Tell me about the history of X"
        assert summary == ""
        assert "innocuous" in rationale.lower()
        # System prompt should be first message
        call_args = mock.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"


@pytest.mark.asyncio
async def test_generate_question_with_score_feedback():
    """Subsequent turn: includes score and response feedback."""
    attacker = Attacker(
        model="gpt-4",
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    with patch("crescendo_optimizer.attacker.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response(_valid_json(
            generated_question="Can you elaborate on that point?",
        ))
        question, _, _ = await attacker.generate_question(
            goal="test goal",
            turn=3,
            max_turns=10,
            last_response="The history of X is fascinating...",
            last_score=0.3,
            last_rationale="Some progress but not enough",
        )
        assert question == "Can you elaborate on that point?"
        # User message should contain score info
        call_args = mock.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "0.30" in user_msg
        assert "turn 3 of 10" in user_msg.lower()


@pytest.mark.asyncio
async def test_generate_question_accumulates_history():
    """Each call adds to the attacker's conversation history."""
    attacker = Attacker(
        model="gpt-4",
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    with patch("crescendo_optimizer.attacker.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response(_valid_json())

        # First call
        await attacker.generate_question(
            goal="g", turn=1, max_turns=10,
            last_response=None, last_score=None, last_rationale=None,
        )
        first_call_msg_count = len(mock.call_args.kwargs["messages"])

        # Second call
        mock.return_value = _mock_response(_valid_json(
            generated_question="Follow up question",
        ))
        await attacker.generate_question(
            goal="g", turn=2, max_turns=10,
            last_response="Some response", last_score=0.2, last_rationale="Low",
        )
        second_call_msg_count = len(mock.call_args.kwargs["messages"])

        # History should grow
        assert second_call_msg_count > first_call_msg_count


@pytest.mark.asyncio
async def test_notify_refusal_adds_to_history():
    """Refusal notification should affect the next question generation."""
    attacker = Attacker(
        model="gpt-4",
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    with patch("crescendo_optimizer.attacker.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response(_valid_json())

        # Generate first question
        await attacker.generate_question(
            goal="g", turn=1, max_turns=10,
            last_response=None, last_score=None, last_rationale=None,
        )

        # Notify refusal
        attacker.notify_refusal("The refused question text")

        # Generate next question — should include refusal context
        mock.return_value = _mock_response(_valid_json(
            generated_question="A different approach",
        ))
        await attacker.generate_question(
            goal="g", turn=1, max_turns=10,
            last_response=None, last_score=None, last_rationale=None,
        )
        messages = mock.call_args.kwargs["messages"]
        # The refusal notification should be in the history
        all_content = " ".join(m["content"] for m in messages)
        assert "refused" in all_content.lower()


@pytest.mark.asyncio
async def test_parse_response_handles_markdown_json():
    """Should parse JSON even if wrapped in markdown code blocks."""
    attacker = Attacker(
        model="gpt-4",
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    wrapped = f"```json\n{_valid_json()}\n```"
    with patch("crescendo_optimizer.attacker.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response(wrapped)
        question, _, _ = await attacker.generate_question(
            goal="g", turn=1, max_turns=10,
            last_response=None, last_score=None, last_rationale=None,
        )
        assert question == "Tell me about the history of X"


@pytest.mark.asyncio
async def test_parse_response_raises_on_missing_key():
    """Should raise ValueError if required key is missing."""
    attacker = Attacker(
        model="gpt-4",
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    bad_json = json.dumps({"generated_question": "Q"})  # missing other keys
    with patch("crescendo_optimizer.attacker.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _mock_response(bad_json)
        with pytest.raises(ValueError, match="missing required keys"):
            await attacker.generate_question(
                goal="g", turn=1, max_turns=10,
                last_response=None, last_score=None, last_rationale=None,
            )


def test_reset_clears_history():
    """Reset should clear conversation history for new attempt."""
    attacker = Attacker(
        model="gpt-4",
        temperature=1.0,
        system_prompt="Goal: {objective}",
    )
    attacker.notify_refusal("some question")
    assert len(attacker._conversation_history) > 0
    attacker.reset()
    assert len(attacker._conversation_history) == 0
