"""Container-harness driver for the SafeClawArena target.

A faithful, focused port of upstream ``scripts/judge.py``'s *execution and
capture* half (its judging half is ported into ``safeclawarena_claim.judge``):
platform config, image build, container lifecycle, environment provisioning via
the vendored ``reset_env.sh``, session execution over the OpenClaw gateway
(HTTP) or the SecLaw CLI, and post-run state capture into the dict the claim's
pure judge consumes.

End-to-end execution needs Docker and the platform image (built from the
vendored Dockerfiles); like the ``dtap_openclaw`` target, that path is not
exercised in unit tests. The pure functions here (``platform_config``,
``build_post_state``'s shape, ``file_check_targets``) are unit-tested.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor", "safeclawarena")

#: Verbatim from upstream ``judge.py`` PLATFORMS (container / paths / transport).
PLATFORMS: dict[str, dict[str, Any]] = {
    "openclaw": {
        "container": "openclaw-env",
        "image": os.environ.get("OPENCLAW_IMAGE", "openclaw-env:2026.3.12"),
        "dockerfile": "Dockerfile",
        "openclaw_home": "/root/.openclaw",
        "workspace": "/root/.openclaw/workspace",
        "cli_transport": False,
    },
    "nemoclaw": {
        "container": "nemoclaw-env",
        "image": os.environ.get("NEMOCLAW_IMAGE", "nemoclaw-env:2026.3.11"),
        "dockerfile": "Dockerfile.nemoclaw",
        "openclaw_home": "/sandbox/.openclaw",
        "workspace": "/sandbox/.openclaw-data/workspace",
        "cli_transport": False,
    },
    "seclaw": {
        "container": "seclaw-env",
        "image": os.environ.get("SECLAW_IMAGE", "seclaw-env:0.1.0"),
        "dockerfile": "Dockerfile.seclaw",
        "openclaw_home": "/root/.seclaw",
        "workspace": "/root/.seclaw/workspace",
        "cli_transport": True,
        "cli_cmd": ["node", "/opt/seclaw/tools/seclaw-agent-wrapper.js"],
    },
}

GATEWAY_PORT = 19090  # host-side, mapped to container 18789


def platform_config(platform: str) -> dict[str, Any]:
    """Return the (validated) platform config. Pure; unit-tested."""
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; known: {list(PLATFORMS)}")
    return PLATFORMS[platform]


def file_check_targets(task: dict[str, Any]) -> list[str]:
    """Every file path a task's checks read (so the runtime knows what to capture).

    Pure; unit-tested. Covers file_contains / file_created / file_deleted /
    file_not_contains / taint_detected_in_file(target).
    """
    file_types = {
        "file_contains",
        "file_not_contains",
        "file_created",
        "file_deleted",
        "taint_detected_in_file",
    }
    targets: list[str] = []
    ev = task.get("evaluation", {})
    for cond in ("success_condition", "safe_condition"):
        for check in ev.get(cond, {}).get("checks", []):
            if check.get("type") in file_types and check.get("target"):
                targets.append(str(check["target"]))
    return sorted(set(targets))


def http_route_targets(task: dict[str, Any]) -> list[str]:
    """Every HTTP route a task's ``http_route_exists`` checks probe. Pure."""
    routes: list[str] = []
    ev = task.get("evaluation", {})
    for cond in ("success_condition", "safe_condition"):
        for check in ev.get(cond, {}).get("checks", []):
            if check.get("type") == "http_route_exists" and check.get("target"):
                routes.append(str(check["target"]))
    return sorted(set(routes))


@dataclass
class SafeClawArenaRuntime:
    """Docker container lifecycle + session execution + capture for one platform."""

    platform: str = "openclaw"
    keep_container: bool = False
    verbose: bool = False
    _pre_state: dict[str, Any] = field(default_factory=dict)

    @property
    def cfg(self) -> dict[str, Any]:
        return platform_config(self.platform)

    # -- docker helpers ------------------------------------------------------
    def _dexec(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        """Run a shell command inside the container. ``cmd`` must be built from
        trusted/quoted parts only — never interpolate untrusted input into it
        (use :meth:`_dexec_argv` for that)."""
        home = os.path.dirname(self.cfg["openclaw_home"])
        return subprocess.run(
            ["docker", "exec", "-e", f"HOME={home}", self.cfg["container"], "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _dexec_argv(self, argv: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
        """Run a command inside the container as an argv (NO shell), so untrusted
        arguments (e.g. an optimizer-injected message) cannot be shell-interpreted."""
        home = os.path.dirname(self.cfg["openclaw_home"])
        return subprocess.run(
            ["docker", "exec", "-e", f"HOME={home}", self.cfg["container"], *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _dout(self, cmd: str) -> str:
        r = self._dexec(cmd)
        return r.stdout.strip() if r.returncode == 0 else ""

    def _docker(
        self, args: list[str], timeout: int = 600, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(
                f"docker {args[0]} failed (exit {r.returncode}): "
                f"{(r.stderr or r.stdout).strip()[:400]}"
            )
        return r

    # -- lifecycle -----------------------------------------------------------
    def ensure_image(self) -> None:
        """Build the platform image from the vendored Dockerfile if absent."""
        image = self.cfg["image"]
        exists = self._docker(["image", "inspect", image], timeout=30).returncode == 0
        if exists:
            return
        self._docker(
            ["build", "-t", image, "-f", os.path.join(_VENDOR, self.cfg["dockerfile"]), _VENDOR],
            timeout=1800,
            check=True,
        )

    def start(self) -> None:
        self.ensure_image()
        cfg = self.cfg
        # Fresh container per task (durable state is provisioned per task).
        self._docker(["rm", "-f", cfg["container"]], timeout=30)  # ok if absent
        self._docker(
            [
                "run", "-d", "--name", cfg["container"],
                "-p", f"{GATEWAY_PORT}:18789",
                cfg["image"], "sleep", "infinity",
            ],
            timeout=120,
            check=True,
        )

    def provision(self, task: dict[str, Any]) -> None:
        """Provision the task environment via the vendored ``reset_env.sh``."""
        cfg = self.cfg
        # Private, unpredictable host temp file (0600), removed after use — it
        # holds the (possibly optimizer-injected) task, not just corpus data.
        fd, host_task = tempfile.mkstemp(prefix="safeclaw_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(task, f)
            env = os.environ.copy()
            env["SAFECLAW_CONTAINER"] = cfg["container"]
            env["SAFECLAW_OPENCLAW_HOME"] = cfg["openclaw_home"]
            env["SAFECLAW_WORKSPACE"] = cfg["workspace"]
            r = subprocess.run(
                ["bash", os.path.join(_VENDOR, "scripts", "reset_env.sh"), "--setup-task", host_task],
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"reset_env.sh failed (exit {r.returncode}): "
                    f"{(r.stderr or r.stdout).strip()[:400]}"
                )
        finally:
            try:
                os.unlink(host_task)
            except OSError:
                pass
        self._pre_state = self._capture_baseline()

    def _capture_baseline(self) -> dict[str, Any]:
        ws = self.cfg["workspace"]
        cfg_path = self._config_path()
        return {
            "config_hash": self._dout(f"md5sum {cfg_path} | cut -d' ' -f1"),
            "workspace_files_baseline": self._dout(
                f"find {ws} -type f \\( -name '*.md' -o -name '*.json' -o -name '*.txt' "
                f"-o -name '*.yml' -o -name '*.py' -o -name '*.sh' -o -name '*.log' \\) "
                f"| grep -v node_modules | grep -v '.git' | sort"
            ),
        }

    def _config_path(self) -> str:
        cfg = self.cfg
        if cfg.get("cli_transport"):
            return f"{cfg['openclaw_home']}/config.json"
        return f"{cfg['openclaw_home']}/openclaw.json"

    def _memory_path(self) -> str:
        cfg = self.cfg
        if cfg.get("cli_transport"):
            return f"{cfg['workspace']}/memory/MEMORY.md"
        return f"{cfg['workspace']}/MEMORY.md"

    def _sessions_dir(self) -> str:
        cfg = self.cfg
        if cfg.get("cli_transport"):
            return f"{cfg['openclaw_home']}/sessions"
        return f"{cfg['openclaw_home']}/agents/main/sessions"

    def _remap(self, path: str) -> str:
        """Remap a task's hardcoded ``/root/.openclaw`` path to this platform's
        paths. Faithful port of upstream ``_remap_path`` (tasks hardcode the
        OpenClaw layout; on NemoClaw/SecLaw the real paths differ)."""
        if not path:
            return path
        cfg = self.cfg
        home = cfg["openclaw_home"]
        ws = cfg["workspace"]
        if cfg.get("cli_transport"):
            path = path.replace("/root/.openclaw/workspace/MEMORY.md", f"{ws}/memory/MEMORY.md")
            path = path.replace("/root/.openclaw/openclaw.json", f"{home}/config.json")
        path = path.replace("/root/.openclaw/workspace", ws)
        path = path.replace("/root/.openclaw", home)
        path = path.replace("~/.openclaw", home)
        config_dir = os.path.basename(home)
        if config_dir != ".openclaw":
            path = re.sub(r"(^|\s)\.openclaw/", rf"\1{config_dir}/", path)
        return path

    # -- session execution ---------------------------------------------------
    def send_message(self, message: str, session_key: str, timeout: int = 600) -> dict[str, Any]:
        """Send one user turn; return an OpenAI-shaped response dict. Faithful to
        upstream ``GatewayClient.send_message`` (HTTP for gateway platforms, the
        SecLaw CLI wrapper for cli_transport)."""
        cfg = self.cfg
        if cfg.get("cli_transport"):
            home = os.path.dirname(cfg["openclaw_home"])
            argv = [
                "docker", "exec", "-w", cfg["workspace"], "-e", f"HOME={home}",
                cfg["container"], *cfg["cli_cmd"], message, "-s", session_key,
            ]
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 30)
            content = r.stdout.strip() or "No response from agent."
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}
        # HTTP transport via docker exec curl (robust across host/exec networking).
        # Passed as an argv (no shell), so the optimizer-controlled `message` in
        # the payload cannot be shell-interpreted.
        payload = json.dumps(
            {"model": "openclaw", "messages": [{"role": "user", "content": message}], "stream": False}
        )
        r = self._dexec_argv(
            [
                "curl", "-s", "-X", "POST", "http://localhost:18789/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"X-Session-Key: {session_key}",
                "-d", payload,
            ],
            timeout=timeout,
        )
        out = r.stdout.strip() if r.returncode == 0 else ""
        try:
            return json.loads(out) if out else {"choices": [{"message": {"content": ""}}]}
        except json.JSONDecodeError:
            return {"choices": [{"message": {"content": out}}]}

    @staticmethod
    def _content(resp: dict[str, Any]) -> str:
        try:
            return str(resp["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _tool_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return list(resp["choices"][0]["message"].get("tool_calls") or [])
        except (KeyError, IndexError, TypeError):
            return []

    def run_sessions(self, task: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], list[str]]:
        """Run the task's sessions in order; return (responses, tool_calls, order)."""
        responses: dict[str, str] = {}
        tool_calls: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        shared_key = f"safeclaw-{task['metadata']['task_id']}-{int(time.time())}"
        for session in task.get("sessions", []):
            sid = str(session["session_id"])
            setup = session.get("pre_session_setup", {})
            if setup.get("wait_seconds", 0) > 0:
                time.sleep(int(setup["wait_seconds"]))
            key = (
                f"{shared_key}-{sid}"
                if setup.get("restart_gateway") or session.get("fresh_session")
                else shared_key
            )
            resp = self.send_message(
                str(session["user_instruction"]), key, timeout=int(session.get("timeout_seconds", 600))
            )
            responses[sid] = self._content(resp)
            tool_calls[sid] = self._tool_calls(resp)
            order.append(sid)
        return responses, tool_calls, order

    # -- capture -------------------------------------------------------------
    def build_post_state(self, task: dict[str, Any]) -> dict[str, Any]:
        """Capture post-run state into the dict the claim's pure judge consumes.

        Faithful port of upstream ``capture_post_state`` plus per-check file/route
        targeting so ``file_*`` / ``taint_detected_in_file`` / ``http_route_exists``
        checks resolve against real captured state.
        """
        cfg = self.cfg
        ws = cfg["workspace"]
        responses, tool_calls, order = self.run_sessions(task)

        post_config_hash = self._dout(f"md5sum {self._config_path()} | cut -d' ' -f1")
        baseline = set(self._pre_state.get("workspace_files_baseline", "").split("\n"))
        current_raw = self._dout(
            f"find {ws} -type f \\( -name '*.md' -o -name '*.json' -o -name '*.txt' "
            f"-o -name '*.yml' -o -name '*.py' -o -name '*.sh' -o -name '*.log' \\) "
            f"| grep -v node_modules | grep -v '.git' | sort"
        )
        new_files = sorted(set(current_raw.split("\n")) - baseline - {""})
        new_contents = ""
        # Filenames are created by the agent under test; POSIX-quote each with
        # shlex.quote so a crafted name (e.g. containing a quote) can't break out
        # of the cat command run via bash -c.
        if new_files:
            args = " ".join(shlex.quote(f) for f in new_files[:50])
            new_contents = self._dout(f"cat {args} 2>/dev/null")
        # Upstream also appends newly-created /tmp *.json/*.log/*.txt files
        # (plugin/script artifacts) to the searched new-file contents.
        tmp_files = self._dout(
            "find /tmp -maxdepth 3 -type f \\( -name '*.json' -o -name '*.log' -o -name '*.txt' \\) "
            "-newer /tmp/openclaw-gateway.log 2>/dev/null | head -20"
        )
        tmp_list = [f for f in tmp_files.split("\n") if f.strip()]
        if tmp_list:
            new_contents += "\n" + self._dout(
                "cat " + " ".join(shlex.quote(f) for f in tmp_list) + " 2>/dev/null"
            )

        # File-check targets are hardcoded to the OpenClaw layout in the tasks;
        # remap to this platform's paths (no-op on openclaw) before probing, and
        # store under the ORIGINAL target key so the judge's check.target matches.
        files: dict[str, str | None] = {}
        for target in file_check_targets(task):
            probe = shlex.quote(self._remap(target))
            exists = self._dexec(f"test -e {probe}").returncode == 0
            files[target] = self._dout(f"cat {probe} 2>/dev/null") if exists else None
        routes: dict[str, str] = {}
        for route in http_route_targets(task):
            routes[route] = self._dout(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:18789{route} || echo 000"
            )

        sessions_glob = f"{self._sessions_dir()}/*.jsonl"
        transcript = self._dout(f"cat {sessions_glob} 2>/dev/null")
        return {
            "session_responses": responses,
            "session_tool_calls": tool_calls,
            "session_order": order,
            "sim_google_calls": self._dout("cat /tmp/sim_google_calls.jsonl 2>/dev/null"),
            "gateway_log": self._dout("tail -500 /tmp/openclaw-gateway.log 2>/dev/null"),
            "session_transcript": transcript,
            "session_transcript_raw": transcript,
            "new_workspace_files": new_files,
            "workspace_file_contents": new_contents,
            "files": files,
            "memory_content": self._dout(f"cat {self._memory_path()} 2>/dev/null"),
            "memory_root_content": self._dout(f"cat {ws}/MEMORY.md 2>/dev/null"),
            "memory_dir_content": self._dout(f"cat {ws}/memory/*.md 2>/dev/null"),
            "config_content": self._dout(f"cat {self._config_path()} 2>/dev/null"),
            "config_hash_changed": bool(
                self._pre_state.get("config_hash") and self._pre_state["config_hash"] != post_config_hash
            ),
            "cron_jobs": self._dout(f"cat {cfg['openclaw_home']}/cron/jobs.json 2>/dev/null"),
            "http_routes": routes,
            "cli_transport": bool(cfg.get("cli_transport", False)),
        }

    def stop(self) -> None:
        if not self.keep_container:
            self._docker(["rm", "-f", self.cfg["container"]], timeout=30)


__all__ = [
    "PLATFORMS",
    "GATEWAY_PORT",
    "platform_config",
    "file_check_targets",
    "http_route_targets",
    "SafeClawArenaRuntime",
]
