"""MUZZLE indirect-injection templates (byte-identical, gsiros/muzzle configs/injections).

The four upstream templates (``{generic,goal_hijacking} x {plain_text,url_injection}``) are
vendored verbatim under ``muzzle_optimizer/data/injections/`` and rendered here. On MUZZLE's
*main* path the planted content is the PAIR-refined payload itself (the malicious
instruction); these templates are the upstream's *fixed-payload* baselines (used by its
ablation scripts ``extras/ablation_scripts/run_rtm_*_fixed_payload_template.py``). The
optimizer therefore exposes them as an optional payload-shaping mode (off by default),
keeping the assets faithful and available without altering the adaptive main path.

Placeholders (upstream ``string.Template`` ``${...}`` syntax): ``${instruction}``,
``${action_url}``, ``${user_goal}``.
"""

from __future__ import annotations

import json
from importlib import resources
from string import Template

INJECTION_FAMILIES = ("generic", "goal_hijacking")
INJECTION_FORMS = ("plain_text", "url_injection")


def load_template(family: str, form: str) -> str:
    """Load a vendored injection-template string by ``family`` and ``form``."""
    if family not in INJECTION_FAMILIES:
        raise ValueError(f"unknown injection family {family!r}")
    if form not in INJECTION_FORMS:
        raise ValueError(f"unknown injection form {form!r}")
    text = (
        resources.files("muzzle_optimizer.data.injections")
        .joinpath(f"{family}_{form}.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(text)
    return str(data["injection"])


def select_family(user_goal: str | None) -> str:
    """``goal_hijacking`` when a benign user goal was recovered, else ``generic``."""
    return "goal_hijacking" if (user_goal and user_goal.strip()) else "generic"


def select_form(action_url: str | None) -> str:
    """``url_injection`` only when a navigable action URL is available, else ``plain_text``."""
    return "url_injection" if (action_url and action_url.strip()) else "plain_text"


def render_injection(
    *,
    instruction: str,
    action_url: str | None = None,
    user_goal: str | None = None,
) -> str:
    """Render an upstream injection template, auto-selecting family and form.

    ``string.Template.safe_substitute`` leaves any unrelated ``$`` text intact.
    """
    family = select_family(user_goal)
    form = select_form(action_url)
    template = load_template(family, form)
    return Template(template).safe_substitute(
        instruction=instruction,
        action_url=action_url or "",
        user_goal=(user_goal or "").strip(),
    )


__all__ = [
    "INJECTION_FAMILIES",
    "INJECTION_FORMS",
    "load_template",
    "select_family",
    "select_form",
    "render_injection",
]
