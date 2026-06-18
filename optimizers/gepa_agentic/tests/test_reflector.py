"""Tests for GEPA-Agentic reflective records."""

from __future__ import annotations

from gepa_agentic_optimizer.reflector import RolloutRecord


class TestRolloutRecord:
    def test_minimal_record_matches_gepa_shape(self) -> None:
        record = RolloutRecord(goal="G", prompt="P")

        assert record.to_sample() == {
            "task_input": "G",
            "candidate_instruction": "P",
        }

    def test_agentic_fields_render_when_present(self) -> None:
        record = RolloutRecord(
            goal="G",
            prompt="P",
            response="assistant reply",
            score=0.5,
            rationale="partial",
            target_observables={"tool_catalog_listing": "[]"},
            selected_surface={
                "name": "read__slack__get_webpage",
                "type": "content_postcall",
            },
            observed_surfaces=[
                {
                    "name": "read__slack__get_webpage",
                    "type": "content_postcall",
                    "event_kind": "post",
                }
            ],
            tool_return="legitimate webpage",
            agent_observations=[
                "controllable_post:read__slack__get_webpage: value"
            ],
            selection_reason="planned content surface",
        )

        sample = record.to_sample()
        assert sample["target_observables"] == {"tool_catalog_listing": "[]"}
        assert sample["selected_surface"]["name"] == "read__slack__get_webpage"
        assert sample["observed_surfaces"][0]["event_kind"] == "post"
        assert sample["surface_selection_reason"] == "planned content surface"
        assert sample["tool_return"] == "legitimate webpage"
        assert sample["agent_observations"] == [
            "controllable_post:read__slack__get_webpage: value"
        ]
        assert sample["assistant_response"] == "assistant reply"
        assert "0.5000" in sample["feedback"]
        assert "partial" in sample["feedback"]
