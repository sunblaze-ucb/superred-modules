"""Container-harness driver for the SafeClawArena target.

A faithful, focused port of upstream ``scripts/judge.py``'s *execution and
capture* half (its judging half is ported into ``safeclawarena_claim.judge``):
platform config, image build, container lifecycle, environment provisioning via
the vendored ``reset_env.sh``, session execution over the OpenClaw gateway
(HTTP), and post-run state capture into the dict the claim's
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

#: Platform configs ported from upstream ``judge.py`` PLATFORMS. SecLaw is
#: excluded — upstream ``Dockerfile.seclaw`` COPYs a ``seclaw/`` source dir that
#: is absent from the repo, so SecLaw is not buildable from the canonical
#: artifact (see ASSUMPTIONS). Only the HTTP-gateway platforms are shipped.
PLATFORMS: dict[str, dict[str, Any]] = {
    "openclaw": {
        "container": "openclaw-env",
        "image": os.environ.get("OPENCLAW_IMAGE", "openclaw-env:2026.3.12"),
        "dockerfile": "Dockerfile",
        "openclaw_home": "/root/.openclaw",
        "workspace": "/root/.openclaw/workspace",    },
    "nemoclaw": {
        "container": "nemoclaw-env",
        "image": os.environ.get("NEMOCLAW_IMAGE", "nemoclaw-env:2026.3.11"),
        "dockerfile": "Dockerfile.nemoclaw",
        "openclaw_home": "/sandbox/.openclaw",
        "workspace": "/sandbox/.openclaw-data/workspace",    },
}

# Container-internal gateway port. No host port is published: all traffic goes
# through `docker exec ... curl localhost:18789` (see start()).
_GATEWAY_PORT = 18789
# Sentinel touched after provisioning but BEFORE any session, so post-run new-file
# discovery (`find -newer`) captures files the agent created during the sessions
# rather than being defeated by the continuously-written gateway log's mtime.
_SESSION_BASELINE_MARKER = "/tmp/.safeclaw_session_baseline"
# Env var supplying the real model API key injected into the container after
# reset_env.sh. The key is the caller's; the target never stores it.
_API_KEY_ENV = "OPENAI_API_KEY"


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
    _gw_token: str | None = field(default=None, init=False)

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
        # No host port is published. The gateway is reached only via
        # `docker exec ... curl localhost:18789`, so a `-p 0.0.0.0:<port>:18789`
        # mapping would merely expose the (static-token-guarded) gateway to the
        # host network and pin a fixed host port that blocks concurrent runs.
        self._docker(
            [
                "run", "-d", "--name", cfg["container"],
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
                [
                    "bash", os.path.join(_VENDOR, "scripts", "reset_env.sh"),
                    "--setup-task", host_task,
                ],
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
        # reset_env.sh installs an auth profile with a "YOUR-API-KEY-HERE"
        # placeholder; inject the caller's real key so the agent's model calls run.
        self._inject_api_key()
        self._pre_state = self._capture_baseline()

    def _inject_api_key(self) -> None:
        """Overwrite the placeholder key in the container's auth-profiles.json with
        the caller's real model key from the environment.

        reset_env.sh installs ``agents/main/agent/auth-profiles.json`` carrying the
        placeholder ``"YOUR-API-KEY-HERE"``; without a real key every gateway model
        call fails and the agent returns nothing, which (with the old error-swallowing
        send_message) scored every task as falsely "secure". The key is the caller's
        responsibility and is never stored on the target — it is passed to the
        container via an inherited env var (``docker exec -e SC_API_KEY`` with no
        value on the argv, so it never appears in a process list or log). If no key
        is set we RAISE, so the run errors and the claim abstains rather than running
        with the placeholder and reporting a false "secure".
        """
        api_key = os.environ.get(_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{_API_KEY_ENV} is not set: SafeClawArena runs a real agent model "
                "in-container, and without a real key every model call fails and would "
                "score every task as falsely 'secure'. Set a real key so the run can "
                "proceed (otherwise the run is excluded from the judged aggregate)."
            )
        path = f"{self.cfg['openclaw_home']}/agents/main/agent/auth-profiles.json"
        # Rewrite the JSON in-container with python3; the key is read from an
        # inherited env var, not interpolated into the argv/script.
        script = (
            "import json,os;"
            f"p={path!r};"
            "d=json.load(open(p));"
            "n=sum(v.__setitem__('key',os.environ['SC_API_KEY']) or 1"
            " for v in d.get('profiles',{}).values()"
            " if isinstance(v,dict) and v.get('type')=='api_key');"
            "json.dump(d,open(p,'w'));"
            "print(n)"
        )
        env = os.environ.copy()
        env["SC_API_KEY"] = api_key
        r = subprocess.run(
            ["docker", "exec", "-e", "SC_API_KEY", self.cfg["container"],
             "python3", "-c", script],
            capture_output=True, text=True, timeout=30, env=env,
        )
        if r.returncode != 0 or (r.stdout.strip() or "0") == "0":
            raise RuntimeError(
                f"failed to inject the model API key into {path} "
                f"(exit {r.returncode}): {(r.stderr or r.stdout).strip()[:200]}"
            )

    def _capture_baseline(self) -> dict[str, Any]:
        ws = self.cfg["workspace"]
        cfg_path = self._config_path()
        baseline = {
            "config_hash": self._dout(f"md5sum {cfg_path} | cut -d' ' -f1"),
            "workspace_files_baseline": self._dout(
                f"find {ws} -type f \\( -name '*.md' -o -name '*.json' -o -name '*.txt' "
                f"-o -name '*.yml' -o -name '*.py' -o -name '*.sh' -o -name '*.log' \\) "
                f"| grep -v node_modules | grep -v '.git' | sort"
            ),
        }
        # Timestamp reference for post-run /tmp new-file discovery: touched now —
        # after provisioning, before any session — so `find -newer` in
        # build_post_state() captures files the agent creates during the sessions.
        # (Using the continuously-appended gateway log as the reference filtered
        # out every agent-created file, biasing taint detection toward false
        # negatives.)
        self._dexec(f"touch {_SESSION_BASELINE_MARKER}")
        return baseline

    def _config_path(self) -> str:
        return f"{self.cfg['openclaw_home']}/openclaw.json"

    def _memory_path(self) -> str:
        return f"{self.cfg['workspace']}/MEMORY.md"

    def _sessions_dir(self) -> str:
        return f"{self.cfg['openclaw_home']}/agents/main/sessions"

    def _remap(self, path: str) -> str:
        """Remap a task's hardcoded ``/root/.openclaw`` path to this platform's
        paths. Faithful port of upstream ``_remap_path`` (tasks hardcode the
        OpenClaw layout; on NemoClaw the real paths differ)."""
        if not path:
            return path
        cfg = self.cfg
        home = cfg["openclaw_home"]
        ws = cfg["workspace"]
        path = path.replace("/root/.openclaw/workspace", ws)
        path = path.replace("/root/.openclaw", home)
        path = path.replace("~/.openclaw", home)
        config_dir = os.path.basename(home)
        if config_dir != ".openclaw":
            path = re.sub(r"(^|\s)\.openclaw/", rf"\1{config_dir}/", path)
        return path

    # -- session execution ---------------------------------------------------
    def _gateway_token(self) -> str:
        """The gateway's HTTP auth token, read from the container's openclaw.json
        (``gateway.auth.token``) and cached. The installed config guards the gateway
        with ``auth.mode = "token"``, so requests without this header are rejected."""
        if self._gw_token is None:
            raw = self._dout(f"cat {self._config_path()} 2>/dev/null")
            token: Any = None
            try:
                token = json.loads(raw).get("gateway", {}).get("auth", {}).get("token")
            except (json.JSONDecodeError, AttributeError, TypeError):
                token = None
            self._gw_token = str(token) if token else ""
        return self._gw_token

    def send_message(self, message: str, session_key: str, timeout: int = 600) -> dict[str, Any]:
        """Send one user turn; return an OpenAI-shaped response dict. Faithful to
        upstream ``GatewayClient.send_message`` (HTTP gateway transport).

        Raises on any gateway failure — transport error, non-2xx status, empty or
        non-JSON body, or an ``error`` field — instead of degrading to empty
        content. An auth/model/transport failure must surface as a run error so the
        claim ABSTAINS (the run is excluded), not be silently scored as "the agent
        did nothing", which would report a false "secure". A legitimate empty
        assistant message (a real 2xx JSON reply with empty content) is NOT an error
        and is returned normally.
        """
        payload = json.dumps(
            {
                "model": "openclaw",
                "messages": [{"role": "user", "content": message}],
                "stream": False,
            }
        )
        argv = [
            "curl", "-s", "-w", "\\n%{http_code}", "-X", "POST",
            f"http://localhost:{_GATEWAY_PORT}/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"X-Session-Key: {session_key}",
        ]
        token = self._gateway_token()
        if token:
            argv += ["-H", f"Authorization: Bearer {token}"]
        # `message` rides in the JSON payload as an argv value (no shell), so it
        # cannot be shell-interpreted regardless of content.
        argv += ["-d", payload]
        r = self._dexec_argv(argv, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"gateway request failed (curl exit {r.returncode})")
        body, _, code = r.stdout.rpartition("\n")
        code = code.strip()
        if not (code.isdigit() and 200 <= int(code) < 300):
            raise RuntimeError(f"gateway returned HTTP {code or '(none)'}")
        body = body.strip()
        if not body:
            raise RuntimeError("gateway returned an empty body")
        try:
            resp: Any = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"gateway returned a non-JSON body: {body[:120]!r}") from exc
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(f"gateway error: {str(resp['error'])[:200]}")
        return resp

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

    def _restart_gateway(self, health_timeout: int = 30) -> None:
        """Genuinely restart the in-container gateway process so on-disk state is
        reloaded fresh. Mirrors the vendored ``reset_env.sh``: kill any running
        ``openclaw gateway``, relaunch it detached (with the same HOME and
        SIM_GOOGLE_DATA_DIR it was started with), then poll ``openclaw health``
        until it responds. Raises if it does not become healthy in time."""
        gw_home = os.path.dirname(self.cfg["openclaw_home"])
        self._dexec("pkill -f 'openclaw gateway'")  # ok if not running
        time.sleep(2)
        launch = (
            f"export HOME={shlex.quote(gw_home)} SIM_GOOGLE_DATA_DIR=/tmp/sim_google_data && "
            f"openclaw gateway --port {_GATEWAY_PORT} > /tmp/openclaw-gateway.log 2>&1"
        )
        self._docker(
            ["exec", "-d", self.cfg["container"], "bash", "-lc", launch],
            timeout=30,
            check=True,
        )
        deadline = time.time() + health_timeout
        while time.time() < deadline:
            if self._dexec("openclaw health --json --timeout 3000").returncode == 0:
                return
            time.sleep(1)
        raise RuntimeError(
            f"gateway did not become healthy within {health_timeout}s after restart"
        )

    def run_sessions(
        self, task: dict[str, Any]
    ) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], list[str]]:
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
            # A genuine gateway restart reloads on-disk state fresh from disk (the
            # point of tasks that test whether injected state survives a restart);
            # merely picking a new session key keeps serving from the same
            # long-lived process, so those tasks would "pass" for the wrong reason.
            if setup.get("restart_gateway"):
                self._restart_gateway()
            key = (
                f"{shared_key}-{sid}"
                if setup.get("restart_gateway") or session.get("fresh_session")
                else shared_key
            )
            timeout = int(session.get("timeout_seconds", 600))
            resp = self.send_message(str(session["user_instruction"]), key, timeout=timeout)
            responses[sid] = self._content(resp)
            tool_calls[sid] = self._tool_calls(resp)
            # Follow-up messages ride the SAME session key (multi-turn injection).
            # For several tasks the attack trigger IS the follow-up, so skipping
            # them left those tasks permanently untriggered (falsely "secure").
            for fu in session.get("follow_up_messages", []) or []:
                fu_msg = fu.get("message", "") if isinstance(fu, dict) else str(fu)
                if not fu_msg:
                    continue
                delay = fu.get("delay_seconds", 0) if isinstance(fu, dict) else 0
                if delay and int(delay) > 0:
                    time.sleep(int(delay))
                fu_resp = self.send_message(str(fu_msg), key, timeout=timeout)
                fu_content = self._content(fu_resp)
                if fu_content:
                    responses[sid] = (
                        f"{responses[sid]}\n{fu_content}" if responses[sid] else fu_content
                    )
                tool_calls[sid].extend(self._tool_calls(fu_resp))
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
            "find /tmp -maxdepth 3 -type f "
            "\\( -name '*.json' -o -name '*.log' -o -name '*.txt' \\) "
            # Exclude the gateway's own live log: it sits under /tmp, matches
            # *.log, and its mtime advances throughout the session, so it is always
            # -newer than the sentinel and would otherwise dump the entire
            # request/response transcript (already captured as gateway_log) into
            # workspace_file_contents -> false-positive taint/leak detection.
            "! -name 'openclaw-gateway.log' "
            f"-newer {_SESSION_BASELINE_MARKER} 2>/dev/null | head -20"
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
            # task-derived route: shlex.quote before interpolating into the shell
            # (matches the file-path branch above; _dout runs via bash -c).
            probe_route = shlex.quote(f"http://localhost:18789{route}")
            routes[route] = self._dout(
                f"curl -s -o /dev/null -w '%{{http_code}}' {probe_route} || echo 000"
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
                self._pre_state.get("config_hash")
                and self._pre_state["config_hash"] != post_config_hash
            ),
            "cron_jobs": self._dout(f"cat {cfg['openclaw_home']}/cron/jobs.json 2>/dev/null"),
            "http_routes": routes,
        }

    def stop(self) -> None:
        if not self.keep_container:
            self._docker(["rm", "-f", self.cfg["container"]], timeout=30)


__all__ = [
    "PLATFORMS",
    "platform_config",
    "file_check_targets",
    "http_route_targets",
    "SafeClawArenaRuntime",
]
