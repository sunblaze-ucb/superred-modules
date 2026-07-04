"""Byte-faithful out-of-band (OOB) runner for the DecodingTrust-Agent judge.

DTAP ships per-task graders as ``<task_dir>/judge.py`` files exposing a ``Judge``
class with ``eval_task`` / ``eval_attack``. The upstream harness grades a run by
calling :func:`utils.judge_helpers.run_judge` (from the installed
``decodingtrust-agent-sdk``), which dynamically imports that ``judge.py`` and
returns a verdict dict. Many DTAP judges are *verifiable*: they re-query the
LIVE environment over HTTP, reading per-instance container host ports from
process environment variables; some judges additionally call an LLM through
``utils.llm.call_llm``.

This module runs that upstream ``run_judge`` **unchanged** in a child process so
that the per-instance state can be wired through the child's environment without
polluting (or being polluted by) the parent:

* every ``env_ports`` entry is exported as ``VAR=port`` so the verifiable judges
  reach *this* instance's containers;
* ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` are set to the judge endpoint so the
  LLM-judge path (``utils.llm`` -> ``OpenAI()``) talks to the experiment proxy;
* ``JUDGE_MODEL`` selects the judge model (default
  :data:`DEFAULT_JUDGE_MODEL`).

The judge's token spend is therefore **out of band** -- it never goes through
the optimizer's ``LLMClient`` and is not counted against the optimizer budget,
exactly like a target's own inference.

Faithfulness notes
------------------
* **Judge call.** The upstream harness grades the *final* response
  (``eval/task_runner.py:374`` calls ``run_judge(task_dir, response=final_output,
  trajectory=...)``); ``run_judge`` wraps a ``str`` into ``[response]`` as the
  judge's ``agent_responses``. We mirror that exactly: ``response=final_response``.
  The full per-turn list (``agent_responses``) is accepted for interface
  parity with the scaffold query surface but is not forwarded; judges that need
  the whole run read it from ``trajectory_json`` (the standardized DTAP-schema
  trace), which IS forwarded.
* **Judge model (the one permitted judge substitution).** ``utils/llm.py`` takes
  the model as a function argument and reads no model env var; the only judge-
  model env var upstream is ``JUDGE_MODEL`` (consulted ONLY by the
  customer_service domain). Every other domain's per-task judge calls
  ``super().__init__(domain=...)`` and so inherits ``BaseJudge``'s *placeholder*
  default ``"gpt-5.4"`` -- a name the proxy does not serve, so those LLM judges
  would fail and mis-score ``attack_success`` as ``False``. The child therefore
  rewrites ONLY that default (:func:`_install_judge_model_override`, run before any
  judge) to the configured ``JUDGE_MODEL``, leaving every judge's LOGIC
  byte-identical. Judges import ``BaseJudge`` as either ``from
  dt_arena.src.types.judge import BaseJudge`` or ``from judge import BaseJudge``,
  so the override unifies both onto one patched class. ``OPENAI_BASE_URL`` /
  ``OPENAI_API_KEY`` still route the endpoint; the caller supplies a
  ``JUDGE_MODEL`` the endpoint actually serves (``utils.llm.call_llm`` keeps the
  temperature path for non-``gpt-5`` models and drops it for ``gpt-5*``, so the
  substitute works either way).
* **Hardcoded endpoints.** A handful of benign customer_service judges post
  directly to ``https://api.openai.com`` and so ignore ``OPENAI_BASE_URL``;
  those calls are not reroutable without editing upstream and are left untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

__all__ = [
    "run_dtap_judge",
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_MODEL_ENV",
    "OPENAI_BASE_URL_ENV",
    "OPENAI_API_KEY_ENV",
    "JUDGE_TIMEOUT_ENV",
    "DEFAULT_JUDGE_TIMEOUT",
]

# The judge-model substitution. The task harness runs against a LiteLLM proxy;
# this is the canonical OOB judge model for the DTAP port.
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-2024-05-13"

# Environment variable names wired into the judge subprocess.
JUDGE_MODEL_ENV = "JUDGE_MODEL"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Judges hit env HTTP and (sometimes) an LLM; allow a generous wall-clock budget,
# overridable for operations without changing the frozen signature.
JUDGE_TIMEOUT_ENV = "DTAP_JUDGE_TIMEOUT"
DEFAULT_JUDGE_TIMEOUT = 1200.0  # seconds (20 min)


# The child runs the UPSTREAM run_judge unchanged. It reads a JSON payload from
# stdin (task_dir / final_response / trajectory_json), routes any chatty judge
# `print()` output to stderr so this process's stdout carries ONLY the result
# JSON, and writes that JSON via the original stdout fd. Per-instance env (ports,
# OPENAI_*, JUDGE_MODEL) is supplied by the parent through `env=`.
_CHILD_SOURCE = r"""
import json
import os
import sys
import traceback


def _emit(obj):
    # sys.__stdout__ is the ORIGINAL stdout (the fd the parent captures), even
    # after we repoint sys.stdout at stderr below.
    out = sys.__stdout__
    out.write(json.dumps(obj))
    out.flush()


def _install_judge_model_override():
    # The ONLY permitted judge change: substitute the judge MODEL, never logic.
    # DTAP's BaseJudge.__init__ defaults judge_model to the placeholder "gpt-5.4";
    # per-task judges call super().__init__(domain=...) and never pass a model, so
    # absent this they request "gpt-5.4" even though OPENAI_BASE_URL/KEY already
    # point at the proxy -> the LLM judges fail and attack_success is mis-scored
    # False. Rewrite ONLY that default to the configured JUDGE_MODEL; judge LOGIC
    # is untouched. Judges import BaseJudge as `from dt_arena.src.types.judge
    # import BaseJudge` OR `from judge import BaseJudge` (the SDK puts
    # dt_arena/src/types on sys.path); unify both onto the one patched module so a
    # single patch covers every judge.
    model = os.environ.get("JUDGE_MODEL")
    if not model:
        return
    import dt_arena.src.types.judge as dtjudge

    base = dtjudge.BaseJudge
    orig_init = base.__init__

    def patched_init(self, domain, judge_model="gpt-5.4"):
        orig_init(self, domain, model if judge_model == "gpt-5.4" else judge_model)

    base.__init__ = patched_init
    sys.modules["judge"] = dtjudge


def _main():
    payload = json.loads(sys.stdin.read())
    # Send any judge/import-time print() to stderr; stdout = result JSON only.
    sys.stdout = sys.stderr
    try:
        _install_judge_model_override()
    except Exception:
        # dt_arena absent / layout differs: fall back to endpoint-only routing
        # (judges keep their default model) -- no regression vs. the pre-fix path.
        pass
    from pathlib import Path

    from utils.judge_helpers import run_judge  # installed decodingtrust-agent-sdk
    result = run_judge(
        Path(payload["task_dir"]),  # upstream run_judge does task_dir / "judge.py"
        response=payload.get("final_response", ""),
        trajectory=payload.get("trajectory_json"),
    )
    _emit(result)


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        _emit(
            {
                "task_success": None,
                "attack_success": None,
                "task_message": "",
                "attack_message": "",
                "error": "judge subprocess failed:\n" + traceback.format_exc(),
            }
        )
"""


def run_dtap_judge(
    task_dir: str,
    final_response: str,
    agent_responses: list[str],
    trajectory_json: dict[str, Any] | None,
    env_ports: dict[str, int],
    *,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
) -> dict[str, Any]:
    """Grade a DTAP run with the upstream judge, out of band, in a subprocess.

    Args:
        task_dir: The DTAP task directory holding ``judge.py``.
        final_response: The agent's final-turn text -- the judge's ``response``.
        agent_responses: Full per-turn outputs. Accepted for interface parity;
            NOT forwarded (upstream grades ``final_response`` -- see module docs).
        trajectory_json: Standardized DTAP-schema trace, or ``None``. Forwarded
            to judges that declare a ``trajectory`` kwarg.
        env_ports: ``{ENV_VAR_NAME: host_port}`` for this instance's live
            containers; each is exported into the judge subprocess so verifiable
            judges reach the right env.
        judge_model: Judge model; defaults to :data:`DEFAULT_JUDGE_MODEL`.
        judge_api_base: Judge LLM base URL (``OPENAI_BASE_URL``); left to the
            ambient env when ``None``.
        judge_api_key: Judge LLM key (``OPENAI_API_KEY``); left to the ambient
            env when ``None``.

    Returns:
        ``{"attack_success", "task_success", "attack_message", "task_message",
        "error"}``. ``attack_success``/``task_success`` are ``bool`` or ``None``
        (``None`` = not applicable / unavailable). On any subprocess failure both
        successes are ``None`` and ``error`` carries the diagnostic.
    """
    # agent_responses is intentionally unused: the byte-faithful upstream call
    # grades `final_response` (see module docstring).
    del agent_responses

    child_env = _build_child_env(env_ports, judge_model, judge_api_base, judge_api_key)
    payload = json.dumps(
        {
            "task_dir": str(task_dir),
            "final_response": final_response or "",
            "trajectory_json": trajectory_json,
        }
    )
    cwd = str(task_dir) if _is_dir(task_dir) else None

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SOURCE],
            input=payload,
            text=True,
            capture_output=True,
            env=child_env,
            cwd=cwd,
            timeout=_subprocess_timeout(),
        )
    except subprocess.TimeoutExpired as exc:
        return _error_result(f"judge subprocess timed out after {exc.timeout}s")
    except Exception as exc:  # noqa: BLE001 - surface any launch failure as an error result
        return _error_result(f"judge subprocess could not be launched: {type(exc).__name__}: {exc}")

    result = _parse_judge_stdout(proc.stdout)
    if result is None:
        tail = (proc.stderr or "").strip()[-2000:]
        return _error_result(
            f"judge subprocess produced no parseable JSON "
            f"(returncode={proc.returncode}). stderr tail:\n{tail}"
        )
    return _map_result(result)


def _build_child_env(
    env_ports: dict[str, int] | None,
    judge_model: str | None,
    judge_api_base: str | None,
    judge_api_key: str | None,
) -> dict[str, str]:
    """Copy the parent env and overlay the per-instance judge wiring."""
    child_env = dict(os.environ)
    for var, port in (env_ports or {}).items():
        child_env[str(var)] = str(port)
    if judge_api_base is not None:
        child_env[OPENAI_BASE_URL_ENV] = str(judge_api_base)
    if judge_api_key is not None:
        child_env[OPENAI_API_KEY_ENV] = str(judge_api_key)
    child_env[JUDGE_MODEL_ENV] = judge_model or DEFAULT_JUDGE_MODEL
    return child_env


def _parse_judge_stdout(stdout: str | None) -> dict[str, Any] | None:
    """Parse the child's stdout JSON, tolerating stray leading output."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        obj = _extract_json_obj(text)
    return obj if isinstance(obj, dict) else None


def _extract_json_obj(text: str) -> Any:
    """Last-resort: parse the first ``{`` .. last ``}`` slice (cf. upstream)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _map_result(result: dict[str, Any]) -> dict[str, Any]:
    """Map the upstream verdict dict onto the frozen return contract."""
    return {
        "attack_success": result.get("attack_success"),
        "task_success": result.get("task_success"),
        "attack_message": result.get("attack_message") or "",
        "task_message": result.get("task_message") or "",
        "error": result.get("error"),
    }


def _error_result(message: str) -> dict[str, Any]:
    """A verdict carrying only a diagnostic; both successes unknown (``None``)."""
    return {
        "attack_success": None,
        "task_success": None,
        "attack_message": "",
        "task_message": "",
        "error": message,
    }


def _subprocess_timeout() -> float:
    raw = os.getenv(JUDGE_TIMEOUT_ENV)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_JUDGE_TIMEOUT


def _is_dir(path: Any) -> bool:
    try:
        return os.path.isdir(path)
    except (TypeError, ValueError):
        return False
