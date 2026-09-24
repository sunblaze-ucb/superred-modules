
import json
from pathlib import Path

_TEMPLATE_MAP = {
    "generate_prompt_evolution":      "generate_prompt_evolution.json",
    "generate_prompt_transformation": "generate_prompt_transformation.json",
    "polish":                         "polish_content.json",
    "align":                          "align_response.json",
    "detect":                         "detect_align.json",
    "generate_intermediate_prompt":   "generate_intermediate_prompt.json",
    "judge":                          "judge_harmful_content.json",
    "compare":                        "compare_similarity.json",
}

_FITD_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def format_prompt(template_name: str, **kwargs) -> str:
    def _load_prompt_template(filename: str) -> dict:
        path = _FITD_PROMPTS_DIR / filename
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    tmpl = _load_prompt_template(_TEMPLATE_MAP[template_name])
    prompt_template = tmpl["prompt_template"]
    variables = dict(tmpl.get("example_variables") or {})
    variables.update(kwargs)
    return prompt_template.format(**variables)