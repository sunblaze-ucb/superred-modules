"""run_agent_capture tests: real Agent + Runner driven by a scripted model, plus
guardrail-tripwire and error capture. Offline."""

from __future__ import annotations

from agents import Agent, GuardrailFunctionOutput, input_guardrail

from openai_agents_target import (
    SENSITIVE_TOOL,
    ScriptedModel,
    build_demo_agent,
    function_call_output,
    message_output,
    run_agent_capture,
)


async def test_message_only_run() -> None:
    agent = build_demo_agent()
    model = ScriptedModel([[message_output("hello there")]])
    result = await run_agent_capture(agent=agent, user_input="hi", model=model)
    assert result.final_response == "hello there"
    assert result.called_tool_names == [] and not result.guardrail_tripped


async def test_tool_call_then_answer() -> None:
    agent = build_demo_agent()
    model = ScriptedModel(
        [
            [function_call_output(SENSITIVE_TOOL, '{"payload": "secret"}')],
            [message_output("done")],
        ]
    )
    result = await run_agent_capture(agent=agent, user_input="hi", model=model)
    assert SENSITIVE_TOOL in result.called_tool_names
    assert result.final_response == "done"
    assert result.tool_calls[0].arguments  # arguments captured


async def test_input_guardrail_tripwire_recorded() -> None:
    @input_guardrail
    def block_everything(context, agent, user_input):  # noqa: ANN001, ARG001
        return GuardrailFunctionOutput(output_info="blocked", tripwire_triggered=True)

    agent = Agent(
        name="guarded",
        instructions="help",
        tools=[],
        input_guardrails=[block_everything],
    )
    model = ScriptedModel([[message_output("should never reach here")]])
    result = await run_agent_capture(agent=agent, user_input="anything", model=model)
    assert result.guardrail_tripped is True and result.guardrail_stage == "input"
    assert result.final_response == ""


async def test_run_error_is_captured() -> None:
    class _Boom(ScriptedModel):
        async def get_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("model down")

    agent = build_demo_agent()
    result = await run_agent_capture(agent=agent, user_input="hi", model=_Boom([]))
    assert "model down" in result.error and result.final_response == ""
