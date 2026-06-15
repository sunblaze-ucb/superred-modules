from poisonedrag_optimizer.prompts import (
    build_joint_generation_prompt,
    clean_str,
    wrap_prompt,
)


def test_rag_wrapper_matches_released_code() -> None:
    prompt = wrap_prompt("Who won?", ["context A", "context B"])

    assert "You are a helpful assistant" in prompt
    assert "context A\ncontext B" in prompt
    assert "Query: Who won?" in prompt
    assert prompt.endswith("Answer:")


def test_joint_generation_prompt_uses_official_json_shape_and_count() -> None:
    prompt = build_joint_generation_prompt(
        question="What is the capital of France?",
        correct_answer="Paris",
        adv_per_query=3,
    )

    assert "Please craft an incorrect answer" in prompt
    assert "Also craft 3 concise corpuses" in prompt
    assert "corpus1, corpus2, corpus3." in prompt
    assert "incorrect answer" in prompt


def test_clean_str_matches_official_period_stripping() -> None:
    assert clean_str("  Wrong Answer. ") == "wrong answer"
    assert clean_str("Wrong Answer..") == "wrong answer."
