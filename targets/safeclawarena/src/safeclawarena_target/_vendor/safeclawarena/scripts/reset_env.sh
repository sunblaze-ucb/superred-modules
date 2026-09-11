#!/usr/bin/env bash
#
# reset_env.sh — SafeClawArena Benchmark Environment Reset Script
#
# Resets the OpenClaw instance inside Docker container to a clean,
# reproducible state for benchmark testing.
#
# Usage:
#   ./reset_env.sh                    # Full reset (default)
#   ./reset_env.sh --soft             # Soft reset (memory/sessions only)
#   ./reset_env.sh --snapshot-only    # Capture baseline snapshot
#   ./reset_env.sh --full --dry-run   # Preview what would be cleaned
#   ./reset_env.sh --setup-task <task.json>  # Full reset + provision task env
#
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
if [ -z "${DOCKER_HOST:-}" ]; then
    _rootless="/run/user/$(id -u)/docker.sock"
    if [ -S "$_rootless" ]; then
        export DOCKER_HOST="unix://${_rootless}"
    elif [ -S "/var/run/docker.sock" ]; then
        export DOCKER_HOST="unix:///var/run/docker.sock"
    else
        echo "ERROR: No Docker socket found. Set DOCKER_HOST or start Docker." >&2
        exit 1
    fi
fi
# Allow override from environment (set by judge.py --platform)
CONTAINER="${SAFECLAW_CONTAINER:-openclaw-env}"
OPENCLAW_HOME="${SAFECLAW_OPENCLAW_HOME:-/root/.openclaw}"
WORKSPACE="${SAFECLAW_WORKSPACE:-${OPENCLAW_HOME}/workspace}"
GATEWAY_PORT=18789

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_DIR="${BENCHMARK_DIR}/configs/platforms"
# Select baseline config by platform
if [ "$OPENCLAW_HOME" = "/root/.openclaw" ]; then
    BASELINE_CONFIG="${BASELINE_DIR}/openclaw.json"
    BASELINE_AUTH="${BASELINE_DIR}/openclaw_auth-profiles.json"
elif [ "$OPENCLAW_HOME" = "/root/.seclaw" ]; then
    # Seclaw: config.json contains providers directly (no separate auth-profiles)
    BASELINE_CONFIG="${BASELINE_DIR}/seclaw.json"
    BASELINE_AUTH=""
else
    # NemoClaw or other platform
    BASELINE_CONFIG="${BASELINE_DIR}/nemoclaw.json"
    BASELINE_AUTH="${BASELINE_DIR}/nemoclaw_auth-profiles.json"
    [ -f "$BASELINE_CONFIG" ] || BASELINE_CONFIG="${BASELINE_DIR}/openclaw.json"
    [ -f "$BASELINE_AUTH" ] || BASELINE_AUTH="${BASELINE_DIR}/openclaw_auth-profiles.json"
fi
BASELINE_WORKSPACE="${BASELINE_DIR}/workspace"
LOG_DIR="${BENCHMARK_DIR}/logs"
LOCKFILE="/tmp/safeclaw_reset.lock"

HEALTH_TIMEOUT=60
HEALTH_INTERVAL=2

# ============================================================================
# Argument Parsing
# ============================================================================
MODE="full"
DRY_RUN=false
VERBOSE=false
TASK_FILE=""

usage() {
    cat <<'USAGE'
Usage: reset_env.sh [OPTIONS]

Options:
  --full            Full reset: clean everything, restore baseline (default)
  --soft            Soft reset: clean memory/sessions only, keep config
  --snapshot-only   Capture current container state as baseline
  --setup-task FILE Full reset + provision task environment from JSON
  --dry-run         Print actions without executing
  --verbose         Show detailed output
  -h, --help        Show this help

Examples:
  ./reset_env.sh --snapshot-only          # First run: save baseline
  ./reset_env.sh                          # Before each task: full reset
  ./reset_env.sh --soft                   # Quick reset between related tasks
  ./reset_env.sh --setup-task ssi-1.1-001.json  # Reset + provision attack env
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)       MODE="full"; shift ;;
        --soft)       MODE="soft"; shift ;;
        --snapshot-only) MODE="snapshot"; shift ;;
        --setup-task) MODE="setup-task"; TASK_FILE="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --verbose)    VERBOSE=true; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ============================================================================
# Logging
# ============================================================================
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/reset_$(date +%Y%m%d_%H%M%S).log"

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_verbose() {
    if $VERBOSE; then
        log "$@"
    else
        echo "$*" >> "$LOG_FILE"
    fi
}

# ============================================================================
# Locking
# ============================================================================
acquire_lock() {
    if [ -f "$LOCKFILE" ]; then
        local lock_pid
        lock_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "")
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            log "ERROR: Another reset is running (PID $lock_pid). Aborting."
            exit 1
        fi
        log "Stale lock found, removing."
        rm -f "$LOCKFILE"
    fi
    echo $$ > "$LOCKFILE"
    trap 'rm -f "$LOCKFILE"' EXIT
}

# ============================================================================
# Docker Helpers
# ============================================================================
dexec() {
    if $DRY_RUN; then
        log_verbose "[DRY-RUN] docker exec $CONTAINER $*"
        return 0
    fi
    # Set HOME based on OPENCLAW_HOME so OpenClaw finds its config
    local _home
    _home="$(dirname "$OPENCLAW_HOME")"
    docker exec -e "HOME=${_home}" "$CONTAINER" "$@"
}

check_container() {
    if ! docker inspect --format='{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q "true"; then
        log "Container $CONTAINER is not running. Starting..."
        if $DRY_RUN; then
            log "[DRY-RUN] docker start $CONTAINER"
        else
            docker start "$CONTAINER"
            sleep 3
        fi
    fi
    log "Container $CONTAINER is running."
}

# ============================================================================
# Gateway Lifecycle
# ============================================================================
stop_gateway() {
    # Seclaw: no gateway to stop
    if [ "$OPENCLAW_HOME" = "/root/.seclaw" ]; then
        return 0
    fi
    log "Stopping gateway..."
    if $DRY_RUN; then
        log "[DRY-RUN] pkill -f openclaw-gateway"
        return 0
    fi

    # Graceful kill
    dexec pkill -f "openclaw-gateway" 2>/dev/null || true
    dexec pkill -f "openclaw gateway" 2>/dev/null || true
    sleep 2

    # Force kill if still alive
    if dexec pgrep -f "openclaw-gateway" >/dev/null 2>&1; then
        log "  Gateway still alive, force killing..."
        dexec pkill -9 -f "openclaw-gateway" 2>/dev/null || true
        dexec pkill -9 -f "openclaw gateway" 2>/dev/null || true
        sleep 1
    fi

    log "  Gateway stopped."
}

start_gateway() {
    # Seclaw: no gateway needed (uses CLI transport)
    if [ "$OPENCLAW_HOME" = "/root/.seclaw" ]; then
        log "Skipping gateway (Seclaw uses CLI transport)."
        return 0
    fi

    log "Starting gateway..."
    if $DRY_RUN; then
        log "[DRY-RUN] docker exec -d $CONTAINER openclaw gateway --port $GATEWAY_PORT"
        return 0
    fi

    # Derive HOME from OPENCLAW_HOME (e.g., /root/.openclaw -> /root, /sandbox/.openclaw -> /sandbox)
    local _gw_home
    _gw_home="$(dirname "$OPENCLAW_HOME")"

    # Ensure gateway config is suitable for benchmark testing.
    # Patches: bind=lan (port-forward), enable Chat Completions HTTP API.
    # For NemoClaw, openclaw.json is read-only so we temporarily chmod as root.
    docker exec "$CONTAINER" python3 -c "
import json, os
cfg_path = '${OPENCLAW_HOME}/openclaw.json'
c = json.load(open(cfg_path))
changed = False
if c.get('gateway', {}).get('bind') != 'lan':
    c.setdefault('gateway', {})['bind'] = 'lan'
    changed = True
http_ep = c.get('gateway', {}).get('http', {}).get('endpoints', {})
if not http_ep.get('chatCompletions', {}).get('enabled'):
    c['gateway'].setdefault('http', {}).setdefault('endpoints', {})
    c['gateway']['http']['endpoints']['chatCompletions'] = {'enabled': True}
    c['gateway']['http']['endpoints']['responses'] = {'enabled': True}
    changed = True
# Ensure workspace path matches actual deployment
ws_cfg = c.get('agents', {}).get('defaults', {}).get('workspace', '')
if ws_cfg and ws_cfg != '${WORKSPACE}':
    c.setdefault('agents', {}).setdefault('defaults', {})['workspace'] = '${WORKSPACE}'
    changed = True
if changed:
    os.chmod(cfg_path, 0o644)
    json.dump(c, open(cfg_path, 'w'), indent=2)
    os.chmod(cfg_path, 0o444)
" 2>/dev/null || true

    docker exec -d "$CONTAINER" bash -c \
        "export HOME=${_gw_home} SIM_GOOGLE_DATA_DIR=/tmp/sim_google_data && openclaw gateway --port ${GATEWAY_PORT} > /tmp/openclaw-gateway.log 2>&1"

    log "  Gateway start command issued."
}

wait_for_healthy() {
    # Seclaw: no gateway to check
    if [ "$OPENCLAW_HOME" = "/root/.seclaw" ]; then
        return 0
    fi
    if $DRY_RUN; then
        log "[DRY-RUN] Wait for gateway health (timeout: ${HEALTH_TIMEOUT}s)"
        return 0
    fi

    local elapsed=0
    log "Waiting for gateway health (timeout: ${HEALTH_TIMEOUT}s)..."

    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if dexec openclaw health --json --timeout 3000 >/dev/null 2>&1; then
            log "  Gateway healthy after ${elapsed}s."
            return 0
        fi
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done

    log "WARNING: Gateway not healthy after ${HEALTH_TIMEOUT}s. Check logs:"
    log "  docker exec $CONTAINER cat /tmp/openclaw-gateway.log"
    return 1
}

# ============================================================================
# State Cleaning Functions
# ============================================================================
clean_memory() {
    log "Cleaning memory..."
    dexec rm -f "${OPENCLAW_HOME}/memory/main.sqlite"
    log_verbose "  Removed: memory/main.sqlite"

    dexec rm -rf "${WORKSPACE}/memory/"
    dexec rm -f "${WORKSPACE}/MEMORY.md"
    log_verbose "  Removed: workspace/memory/ and MEMORY.md"
}

clean_sessions() {
    log "Cleaning sessions..."
    # Remove all session files for all agents
    dexec bash -c "find ${OPENCLAW_HOME}/agents/ -path '*/sessions/*' -type f -delete 2>/dev/null; true"
    log_verbose "  Removed: all session files under agents/*/sessions/"
}

clean_cron() {
    log "Cleaning cron jobs..."
    # Reset jobs.json to empty state (mkdir -p in case dir doesn't exist on fresh container)
    dexec bash -c "mkdir -p ${OPENCLAW_HOME}/cron && echo '{\"version\":1,\"jobs\":[]}' > ${OPENCLAW_HOME}/cron/jobs.json"
    dexec rm -f "${OPENCLAW_HOME}/cron/jobs.json.bak"
    dexec rm -rf "${OPENCLAW_HOME}/cron/runs/"
    log_verbose "  Reset: cron/jobs.json to empty; removed cron/runs/"
}

clean_cron_runs_only() {
    log "Cleaning cron run history..."
    dexec rm -rf "${OPENCLAW_HOME}/cron/runs/"
    log_verbose "  Removed: cron/runs/"
}

clean_skills() {
    log "Cleaning skills..."
    # Remove all preinstalled managed skills except sim-google (needed by 166+ tasks).
    # This ensures <available_skills> only contains task-relevant skills,
    # preventing interference from 50+ unrelated preinstalled skills.
    # Auto-detect managed skills directory (differs between OpenClaw and NemoClaw)
    MANAGED_SKILLS_DIR=$(docker exec "$CONTAINER" bash -c "ls -d /usr/local/lib/node_modules/openclaw/skills /usr/lib/node_modules/openclaw/skills 2>/dev/null | head -1")
    if [ -z "$MANAGED_SKILLS_DIR" ]; then
        MANAGED_SKILLS_DIR="/usr/lib/node_modules/openclaw/skills"
    fi
    dexec bash -c "for d in ${MANAGED_SKILLS_DIR}/*/; do name=\$(basename \"\$d\"); [ \"\$name\" != 'sim-google' ] && rm -rf \"\$d\"; done"
    log_verbose "  Removed: managed skills (kept sim-google)"

    dexec rm -rf "${WORKSPACE}/skills/"
    log_verbose "  Removed: workspace skills"
}

clean_extensions() {
    log "Cleaning extensions/plugins..."
    # Extensions may be at OPENCLAW_HOME/extensions/ or via symlink to data dir
    dexec rm -rf "${OPENCLAW_HOME}/extensions/"
    dexec mkdir -p "${OPENCLAW_HOME}/extensions/"
    log_verbose "  Removed: extensions/"
}

clean_exec_approvals() {
    log "Cleaning exec approvals..."
    dexec rm -f "${OPENCLAW_HOME}/exec-approvals.json"
    log_verbose "  Removed: exec-approvals.json"
}

clean_logs() {
    log "Cleaning logs..."
    dexec rm -f "${OPENCLAW_HOME}/logs/config-audit.jsonl"
    dexec rm -f /tmp/openclaw-gateway.log
    dexec rm -f /tmp/sim_google_calls.jsonl
    dexec rm -rf /tmp/sim_google_data
    log_verbose "  Removed: config-audit.jsonl, gateway log, sim_google_calls.jsonl, sim_google_data"
}

clean_attack_artifacts() {
    log "Cleaning attack artifacts from workspace..."
    # Remove common attack artifact patterns
    dexec rm -f "${WORKSPACE}/.env" 2>/dev/null || true
    dexec rm -rf "${WORKSPACE}/.ssh/" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.git/config.bak" 2>/dev/null || true
    dexec rm -rf "${WORKSPACE}/exfil/" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/credentials.json" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/secrets.txt" 2>/dev/null || true
    # Plugin attack artifacts (Cat 1.5)
    dexec rm -f "${WORKSPACE}/.workspace-index.json" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.env-validation.log" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.session-metrics.json" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.memory-index.json" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.quality-cache.json" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.shortcuts-config.json" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.cleanup-report.txt" 2>/dev/null || true
    dexec rm -rf /tmp/.dev-shortcuts 2>/dev/null || true
    dexec rm -f /tmp/.cq-counter 2>/dev/null || true
    # Hidden cache files from Cat 1.1 attacks
    dexec bash -c "find ${WORKSPACE} -maxdepth 1 -name '.*-cache.json' -delete 2>/dev/null || true"
    dexec rm -f "${WORKSPACE}/.debug-context.log" 2>/dev/null || true
    dexec rm -f "${WORKSPACE}/.compat-report.json" 2>/dev/null || true
    log_verbose "  Cleaned common attack artifacts from workspace"
}

# ============================================================================
# Baseline Restore
# ============================================================================
restore_config() {
    log "Restoring baseline config..."

    if [ ! -f "$BASELINE_CONFIG" ]; then
        log "ERROR: Baseline config not found at $BASELINE_CONFIG"
        log "  Run: $0 --snapshot-only"
        exit 1
    fi

    if $DRY_RUN; then
        log "[DRY-RUN] docker cp $BASELINE_CONFIG -> container"
        return 0
    fi

    # Seclaw uses config.json; OpenClaw/NemoClaw use openclaw.json
    if [ "$OPENCLAW_HOME" = "/root/.seclaw" ]; then
        docker cp "$BASELINE_CONFIG" "${CONTAINER}:${OPENCLAW_HOME}/config.json"
        log_verbose "  Restored: config.json (Seclaw)"
    else
        docker cp "$BASELINE_CONFIG" "${CONTAINER}:${OPENCLAW_HOME}/openclaw.json"
        log_verbose "  Restored: openclaw.json"
    fi

    if [ -n "$BASELINE_AUTH" ] && [ -f "$BASELINE_AUTH" ]; then
        docker cp "$BASELINE_AUTH" "${CONTAINER}:${OPENCLAW_HOME}/agents/main/agent/auth-profiles.json"
        log_verbose "  Restored: auth-profiles.json"
    fi
}

restore_workspace_bootstrap() {
    # Seclaw: keep original bootstrap files from the image
    if [ "$OPENCLAW_HOME" = "/root/.seclaw" ]; then
        log "Skipping bootstrap restore (Seclaw uses its own templates)."
        return 0
    fi

    log "Restoring workspace bootstrap files..."

    if [ ! -d "$BASELINE_WORKSPACE" ]; then
        log "WARNING: Baseline workspace not found at $BASELINE_WORKSPACE, skipping."
        return 0
    fi

    if $DRY_RUN; then
        log "[DRY-RUN] Restore workspace bootstrap files from baseline"
        return 0
    fi

    for f in AGENTS.md SOUL.md USER.md TOOLS.md IDENTITY.md HEARTBEAT.md BOOTSTRAP.md; do
        if [ -f "${BASELINE_WORKSPACE}/${f}" ]; then
            docker cp "${BASELINE_WORKSPACE}/${f}" "${CONTAINER}:${WORKSPACE}/${f}"
            log_verbose "  Restored: ${f}"
        fi
    done
}

# ============================================================================
# Snapshot Capture
# ============================================================================
capture_snapshot() {
    log "Capturing baseline snapshot..."
    check_container

    mkdir -p "$BASELINE_DIR" "${BASELINE_WORKSPACE}"

    docker cp "${CONTAINER}:${OPENCLAW_HOME}/openclaw.json" "$BASELINE_CONFIG"
    log "  Saved: openclaw.json -> $(basename "$BASELINE_CONFIG")"

    docker cp "${CONTAINER}:${OPENCLAW_HOME}/agents/main/agent/auth-profiles.json" \
        "$BASELINE_AUTH" 2>/dev/null && \
        log "  Saved: auth-profiles.json -> $(basename "$BASELINE_AUTH")" || \
        log "  Skip: auth-profiles.json (not found)"

    for f in AGENTS.md SOUL.md USER.md TOOLS.md IDENTITY.md HEARTBEAT.md BOOTSTRAP.md; do
        docker cp "${CONTAINER}:${WORKSPACE}/${f}" \
            "${BASELINE_WORKSPACE}/${f}" 2>/dev/null && \
            log "  Saved: ${f}" || \
            log "  Skip: ${f} (not found)"
    done

    log "Baseline snapshot saved to: ${BASELINE_DIR}/"
}

# ============================================================================
# Task Environment Provisioning
# ============================================================================
setup_task_env() {
    local task_json="$1"

    if [ ! -f "$task_json" ]; then
        log "ERROR: Task file not found: $task_json"
        exit 1
    fi

    log "Provisioning task environment from: $task_json"

    if ! command -v python3 &>/dev/null; then
        log "ERROR: python3 required for task provisioning"
        exit 1
    fi

    # Use Python to parse task JSON and provision environment
    python3 - "$task_json" "$CONTAINER" "$WORKSPACE" "$OPENCLAW_HOME" <<'PYTHON_SCRIPT'
import json
import subprocess
import sys
import os

task_file = sys.argv[1]
container = sys.argv[2]
workspace = sys.argv[3]
openclaw_home = sys.argv[4]

with open(task_file) as f:
    task = json.load(f)

env = task.get("environment", {})

# Path remapping: only active on non-OpenClaw platforms (NemoClaw, Seclaw)
_needs_remap = workspace != "/root/.openclaw/workspace"
_is_seclaw = openclaw_home == "/root/.seclaw"
def remap_paths(text):
    """Replace hardcoded OpenClaw paths with current platform paths. No-op on OpenClaw."""
    if not _needs_remap:
        return text
    if _is_seclaw:
        # Seclaw has structural differences beyond just the prefix:
        # - Memory: workspace/MEMORY.md -> workspace/memory/MEMORY.md
        # - Config: openclaw.json -> config.json
        text = text.replace("/root/.openclaw/workspace/MEMORY.md", f"{workspace}/memory/MEMORY.md")
        text = text.replace("/root/.openclaw/openclaw.json", f"{openclaw_home}/config.json")
    # Order matters: replace longer path first
    text = text.replace("/root/.openclaw/workspace", workspace)
    text = text.replace("/root/.openclaw", openclaw_home)
    text = text.replace("~/.openclaw", openclaw_home)
    # Handle Path.home() / ".openclaw" style in Python scripts
    config_dir = os.path.basename(openclaw_home)  # e.g. ".seclaw"
    if config_dir != ".openclaw":
        text = text.replace('".openclaw"', f'"{config_dir}"')
        text = text.replace("'.openclaw'", f"'{config_dir}'")
    return text

def dexec(cmd):
    """Execute command in container."""
    home_dir = os.path.dirname(openclaw_home)
    full_cmd = ["docker", "exec", "-e", f"HOME={home_dir}", container, "bash", "-c", cmd]
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"  Warning: {result.stderr.strip()}", file=sys.stderr)
    return result

def docker_cp_content(content, container_path):
    """Write content to a file inside the container."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        subprocess.run(
            ["docker", "cp", tmp_path, f"{container}:{container_path}"],
            check=True, capture_output=True
        )
    finally:
        os.unlink(tmp_path)

# 1. Provision skills
for skill in env.get("skills", []):
    name = skill["name"]
    location = skill.get("location", "workspace")
    content = skill.get("content", "")

    if not content and "content_file" in skill:
        content_path = os.path.join(os.path.dirname(task_file), skill["content_file"])
        with open(content_path) as f:
            content = f.read()

    # Auto-detect managed skills directory
    managed_base = subprocess.run(
        ["docker", "exec", container, "bash", "-c",
         "ls -d /usr/local/lib/node_modules/openclaw/skills /usr/lib/node_modules/openclaw/skills 2>/dev/null | head -1"],
        capture_output=True, text=True).stdout.strip() or "/usr/lib/node_modules/openclaw/skills"

    # Seclaw: always deploy to workspace skills (no managed skills directory)
    is_seclaw = openclaw_home == "/root/.seclaw"
    is_nemoclaw = not is_seclaw and workspace != "/root/.openclaw/workspace"

    if is_seclaw:
        skill_dir = f"{workspace}/skills/{name}"
    elif location == "workspace":
        ws_skill_dir = f"{workspace}/skills/{name}"
        if is_nemoclaw:
            # NemoClaw: workspace skills not listed -> deploy as managed
            skill_dir = f"{managed_base}/{name}"
            print(f"  (NemoClaw: workspace skill deployed as managed)")
        else:
            skill_dir = ws_skill_dir
    else:
        skill_dir = f"{managed_base}/{name}"

    content = remap_paths(content)
    dexec(f"mkdir -p '{skill_dir}'")
    docker_cp_content(content, f"{skill_dir}/SKILL.md")
    # Ensure skill files are readable by all users (gateway, sandbox)
    dexec(f"chmod -R a+rX '{skill_dir}'")
    print(f"  Skill: {name} -> {location} ({len(content)} bytes)")
    # Skills are loaded naturally by OpenClaw: the gateway lists them in
    # <available_skills> in the system prompt (name + description only),
    # and the LLM reads the full SKILL.md via the read tool when needed.
    # No AGENTS.md injection — we rely on the real skill-loading mechanism.

    # Provision bundled script files (Cat 1.6)
    script_files = skill.get("script_files", {})
    for rel_path, file_content in script_files.items():
        # rel_path is e.g. "scripts/generate.py"
        file_content = remap_paths(file_content)
        full_path = f"{skill_dir}/{rel_path}"
        parent_dir = os.path.dirname(full_path)
        dexec(f"mkdir -p '{parent_dir}'")
        docker_cp_content(file_content, full_path)
        # Make scripts executable
        dexec(f"chmod +x '{full_path}'")
        print(f"    Script: {rel_path} ({len(file_content)} bytes)")

# 1b. Apply config overrides EARLY (before plugin deploy + gateway restart)
#     This ensures plugins.allow, plugins.load.paths etc. are in openclaw.json
#     before the gateway starts and decides which plugins to enable.
overrides = env.get("config_overrides", {})
if overrides:
    # Remap any hardcoded paths in config overrides
    overrides_str = remap_paths(json.dumps(overrides))
    overrides = json.loads(overrides_str)
    print(f"  Config overrides (early): {json.dumps(overrides)}")
    docker_cp_content(json.dumps(overrides), "/tmp/_safeclaw_overrides.json")
    # Seclaw uses config.json, OpenClaw/NemoClaw use openclaw.json
    config_filename = "config.json" if _is_seclaw else "openclaw.json"
    merge_script = (
        "import json, os\n"
        "with open('/tmp/_safeclaw_overrides.json') as f:\n"
        "    overrides = json.load(f)\n"
        f"with open('{openclaw_home}/{config_filename}') as f:\n"
        "    cfg = json.load(f)\n"
        "def deep_merge(base, patch):\n"
        "    for k, v in patch.items():\n"
        "        if isinstance(v, dict) and isinstance(base.get(k), dict):\n"
        "            deep_merge(base[k], v)\n"
        "        else:\n"
        "            base[k] = v\n"
        "deep_merge(cfg, overrides)\n"
        f"with open('{openclaw_home}/{config_filename}', 'w') as f:\n"
        "    json.dump(cfg, f, indent=2)\n"
        "os.unlink('/tmp/_safeclaw_overrides.json')\n"
    )
    docker_cp_content(merge_script, "/tmp/_safeclaw_merge.py")
    result = dexec("python3 /tmp/_safeclaw_merge.py")
    if result.returncode == 0:
        print("  Applied config overrides.")
    else:
        print("  ERROR: Config override failed!", file=sys.stderr)
    _config_overrides_applied = True
else:
    _config_overrides_applied = False

# 1c. Provision plugins (Cat 1.5) — use OpenClaw's real install mechanism
for plugin in env.get("plugins", []):
    pid = plugin["plugin_id"]
    files = plugin.get("files", {})

    # Stage plugin files in a temp directory, then install via openclaw CLI
    staging_dir = f"/tmp/_plugin_stage/{pid}"
    dexec(f"mkdir -p '{staging_dir}'")
    for fname, fcontent in files.items():
        fcontent = remap_paths(fcontent)
        docker_cp_content(fcontent, f"{staging_dir}/{fname}")
        dexec(f"chmod a+r '{staging_dir}/{fname}'")

    # Install using OpenClaw's real plugin install command (local path)
    result = dexec(f"cd {workspace} && openclaw plugins install '{staging_dir}' 2>&1")
    if result.returncode == 0:
        print(f"  Plugin: {pid} installed via 'openclaw plugins install' ({len(files)} files)")
    else:
        # Fallback: manual deployment to extensions directory
        print(f"  Plugin: {pid} CLI install failed, falling back to manual deployment")
        print(f"    stderr: {(result.stdout or '')[:200]}")
        is_nemoclaw = workspace != "/root/.openclaw/workspace"
        if is_nemoclaw:
            ext_dir = f"{openclaw_home}/extensions/{pid}"
        else:
            # OpenClaw: deploy to workspace extensions (matches config_overrides plugins.load.paths)
            ext_dir = f"{workspace}/.openclaw/extensions/{pid}"
        dexec(f"mkdir -p '{ext_dir}'")
        for fname, fcontent in files.items():
            fcontent = remap_paths(fcontent)
            docker_cp_content(fcontent, f"{ext_dir}/{fname}")
        dexec(f"chmod -R a+rX '{ext_dir}'")
        dexec(f"chown -R root:root '{ext_dir}'")
        print(f"  Plugin: {pid} deployed to {ext_dir} ({len(files)} files)")
        # Non-OpenClaw: fix plugins.load.paths in config to match actual deployment path
        if is_nemoclaw:
            ext_base = f"{openclaw_home}/extensions"
            cfg_name = "config.json" if _is_seclaw else "openclaw.json"
            dexec(f"python3 -c \""
                  f"import json,os; cfg='{openclaw_home}/{cfg_name}'; "
                  f"os.chmod(cfg,0o644); c=json.load(open(cfg)); "
                  f"paths=c.get('plugins',{{}}).get('load',{{}}).get('paths',[]); "
                  f"c.setdefault('plugins',{{}}).setdefault('load',{{}})['paths']=['{ext_base}']; "
                  f"json.dump(c,open(cfg,'w'),indent=2); os.chmod(cfg,0o444)"
                  f"\"")
            print(f"  Config: plugins.load.paths updated to [{ext_base}] in {cfg_name}")

    # Clean up staging
    dexec(f"rm -rf '{staging_dir}'")

# 2. Provision memory files
for mem in env.get("memory_files", []):
    path = mem["path"]
    content = mem.get("content", "")
    if not content and "content_file" in mem:
        content_path = os.path.join(os.path.dirname(task_file), mem["content_file"])
        with open(content_path) as f:
            content = f.read()

    content = remap_paths(content)
    # Seclaw: memory files go under workspace/memory/ subdirectory
    if _is_seclaw and path == "MEMORY.md":
        full_path = f"{workspace}/memory/MEMORY.md"
    else:
        full_path = f"{workspace}/{path}"
    parent_dir = os.path.dirname(full_path)
    dexec(f"mkdir -p '{parent_dir}'")
    docker_cp_content(content, full_path)
    print(f"  Memory: {path} -> {full_path} ({len(content)} bytes)")

# 2b. Provision pre-injected memory (PSE Cat 2.3, 2.5 — memory/config already tampered before session)
pim = env.get("pre_injected_memory", {})
if pim:
    target = pim.get("target_file", "MEMORY.md")
    content = pim.get("content", "")
    if target == "MEMORY.md":
        full_path = f"{workspace}/memory/MEMORY.md" if _is_seclaw else f"{workspace}/MEMORY.md"
    elif target == "AGENTS.md":
        full_path = f"{workspace}/AGENTS.md"
    else:
        full_path = f"{workspace}/{target}"
    parent_dir = os.path.dirname(full_path)
    dexec(f"mkdir -p '{parent_dir}'")

    # For needle_in_memory, build large memory with credential at specific position
    if pim.get("needle_position"):
        position = pim["needle_position"]
        needle = pim.get("needle_content", "")
        padding_block = "## Session Notes\\n- Worked on various tasks\\n- No issues found\\n" * 200
        if position == "beginning":
            content = f"# Memory\\n\\n{needle}\\n\\n{padding_block}"
        elif position == "end":
            content = f"# Memory\\n\\n{padding_block}\\n\\n{needle}"
        else:  # middle
            half = "## Session Notes\\n- Worked on various tasks\\n- No issues found\\n" * 100
            content = f"# Memory\\n\\n{half}\\n\\n{needle}\\n\\n{half}"

    content = remap_paths(content)
    docker_cp_content(content, full_path)
    print(f"  Pre-injected {target}: {len(content)} bytes (vector: {pim.get('attack_vector', 'unknown')})")

# 3. Provision workspace files (sensitive data, .env, etc.)
for wf in env.get("workspace_files", []):
    path = wf["path"]
    content = wf.get("content", "")
    if not content and "content_file" in wf:
        content_path = os.path.join(os.path.dirname(task_file), wf["content_file"])
        with open(content_path) as f:
            content = f.read()

    content = remap_paths(content)
    if path.startswith("/"):
        full_path = remap_paths(path)  # Absolute path — remap if needed
    else:
        full_path = f"{workspace}/{path}"
    parent_dir = os.path.dirname(full_path)
    dexec(f"mkdir -p '{parent_dir}'")
    docker_cp_content(content, full_path)
    # Ensure readable by sandbox/gateway users
    dexec(f"chmod a+r '{full_path}'")
    label = " [SENSITIVE]" if wf.get("is_sensitive") else ""
    print(f"  File: {path}{label} ({len(content)} bytes)")

# 3-symlink. On Seclaw, create symlinks so agent can find workspace files from HOME
# Seclaw agent's read_file resolves relative paths from HOME (/root), not from workspace.
# We symlink workspace subdirectories into HOME so both relative and absolute paths work.
if _is_seclaw:
    home_dir = os.path.dirname(openclaw_home)  # /root
    # Collect all unique top-level directories from workspace_files
    ws_dirs = set()
    for wf in env.get("workspace_files", []):
        path = wf["path"]
        if "/" in path and not path.startswith("/"):
            top_dir = path.split("/")[0]
            if top_dir not in ('.env',):
                ws_dirs.add(top_dir)
    # Also symlink all top-level files (not just .env)
    for wf in env.get("workspace_files", []):
        path = wf["path"]
        if "/" not in path and not path.startswith("/"):
            src = f"{workspace}/{path}"
            dst = f"{home_dir}/{path}"
            dexec(f"[ -f '{src}' ] && [ ! -e '{dst}' ] && ln -s '{src}' '{dst}' || true")
    for d in ws_dirs:
        src = f"{workspace}/{d}"
        dst = f"{home_dir}/{d}"
        dexec(f"[ -d '{src}' ] && [ ! -e '{dst}' ] && ln -s '{src}' '{dst}' || true")
    if ws_dirs:
        print(f"  Symlinks: {', '.join(sorted(ws_dirs))} + files -> workspace")

# 3-git. Initialize git repo if task requires it
if env.get("git_init"):
    dexec(f"cd {workspace} && git config --global user.email 'dev@example.com' && git config --global user.name 'Developer' && git init -q && git add -A && git commit -q -m 'initial commit'")
    # Add a few more commits so git-related skills have history to work with
    dexec(f"cd {workspace} && echo '# v0.1.0' >> README.md && git add -A && git commit -q -m 'Add README and project docs'")
    dexec(f"cd {workspace} && echo '# changelog' >> CHANGELOG.md && git add -A && git commit -q -m 'Add changelog and update dependencies'")
    dexec(f"cd {workspace} && echo '' >> .env && git add -A && git commit -q -m 'Update configuration for staging environment'")
    print(f"  Git: initialized repo with initial commit")

# 3a. Provision sim-google data store (emails, calendar events, chat messages, etc.)
gog_data_dir = "/tmp/sim_google_data"
dexec(f"rm -rf {gog_data_dir} && mkdir -p {gog_data_dir}")

for gog_item in env.get("sim_google_data", []):
    gog_path = gog_item["path"]  # e.g., "gmail/inbox/config-drift.eml"
    gog_content = gog_item.get("content", "")
    full_gog_path = f"{gog_data_dir}/{gog_path}"
    parent = os.path.dirname(full_gog_path)
    dexec(f"mkdir -p '{parent}'")
    docker_cp_content(gog_content, full_gog_path)
    dexec(f"chmod a+r '{full_gog_path}'")
    print(f"  Gog data: {gog_path} ({len(gog_content)} bytes)")

# Also migrate any workspace files in inbox/ or incoming/ to sim-google store
# (backward compatibility: if tasks still use workspace_files for emails)
for wf in env.get("workspace_files", []):
    path = wf["path"]
    if path.startswith("inbox/") or path.startswith("incoming/"):
        content = wf.get("content", "")
        if path.endswith(".eml"):
            gog_path = f"{gog_data_dir}/gmail/inbox/{os.path.basename(path)}"
        elif path.endswith(".json"):
            gog_path = f"{gog_data_dir}/calendar/{os.path.basename(path)}"
        else:
            continue
        parent = os.path.dirname(gog_path)
        dexec(f"mkdir -p '{parent}'")
        docker_cp_content(content, gog_path)
        print(f"  Gog data (migrated): {path} -> {gog_path}")

# Set SIM_GOOGLE_DATA_DIR for the container environment
_user_home = os.path.dirname(openclaw_home)
dexec(f"echo 'export SIM_GOOGLE_DATA_DIR={gog_data_dir}' >> {_user_home}/.bashrc")

# 3b. Merge canary profiles into auth-profiles.json (preserves real working keys)
auth_pf = env.get("auth_profiles_file", {})
if auth_pf:
    extra_json = auth_pf.get("content", "")
    if extra_json:
        auth_dir = f"{openclaw_home}/agents/main/agent"
        auth_path = f"{auth_dir}/auth-profiles.json"
        dexec(f"mkdir -p '{auth_dir}'")
        docker_cp_content(extra_json, "/tmp/_safeclaw_extra_profiles.json")
        merge_script = (
            "import json, os\n"
            f"auth_path = '{auth_path}'\n"
            "try:\n"
            "    with open(auth_path) as f:\n"
            "        store = json.load(f)\n"
            "except (FileNotFoundError, json.JSONDecodeError):\n"
            '    store = {"version": 1, "profiles": {}}\n'
            "if 'profiles' not in store:\n"
            "    store['profiles'] = {}\n"
            "with open('/tmp/_safeclaw_extra_profiles.json') as f:\n"
            "    extra = json.load(f)\n"
            "store['profiles'].update(extra)\n"
            "with open(auth_path, 'w') as f:\n"
            "    json.dump(store, f, indent=2)\n"
            "os.unlink('/tmp/_safeclaw_extra_profiles.json')\n"
        )
        docker_cp_content(merge_script, "/tmp/_safeclaw_merge_auth.py")
        result = dexec("python3 /tmp/_safeclaw_merge_auth.py")
        if result.returncode == 0:
            print(f"  Auth profiles: merged canary profiles into {auth_path}")
        else:
            print(f"  ERROR: Auth profile merge failed!", file=sys.stderr)
        # On Seclaw, create symlinks for auth-profiles.json after deployment
        if _is_seclaw:
            home_dir = os.path.dirname(openclaw_home)
            for dst in [f"{workspace}/auth-profiles.json", f"{home_dir}/auth-profiles.json"]:
                dexec(f"[ -f '{auth_path}' ] && [ ! -e '{dst}' ] && ln -s '{auth_path}' '{dst}' || true")

# 4. Apply config overrides (skip if already applied in step 1b)
if not _config_overrides_applied:
    overrides = env.get("config_overrides", {})
    if overrides:
        print(f"  Config overrides: {json.dumps(overrides)}")
        docker_cp_content(json.dumps(overrides), "/tmp/_safeclaw_overrides.json")
        merge_script = (
            "import json, os\n"
            "with open('/tmp/_safeclaw_overrides.json') as f:\n"
            "    overrides = json.load(f)\n"
            f"with open('{openclaw_home}/openclaw.json') as f:\n"
            "    cfg = json.load(f)\n"
            "def deep_merge(base, patch):\n"
            "    for k, v in patch.items():\n"
            "        if isinstance(v, dict) and isinstance(base.get(k), dict):\n"
            "            deep_merge(base[k], v)\n"
            "        else:\n"
            "            base[k] = v\n"
            "deep_merge(cfg, overrides)\n"
            f"with open('{openclaw_home}/openclaw.json', 'w') as f:\n"
            "    json.dump(cfg, f, indent=2)\n"
            "os.unlink('/tmp/_safeclaw_overrides.json')\n"
        )
        docker_cp_content(merge_script, "/tmp/_safeclaw_merge.py")
        result = dexec("python3 /tmp/_safeclaw_merge.py")
        if result.returncode == 0:
            print("  Applied config overrides.")
        else:
            print("  ERROR: Config override failed!", file=sys.stderr)

# 5. Provision cron jobs (if any)
for cron in env.get("cron_jobs", []):
    name = cron.get("name", "benchmark-cron")
    schedule = cron.get("schedule", "")
    message = cron.get("message", "")
    session_mode = cron.get("session_mode", "isolated")
    print(f"  Cron: {name} (schedule: {schedule}) [requires manual setup via CLI]")

print("Task environment provisioned successfully.")
PYTHON_SCRIPT

    log "Task environment provisioned."
}

# ============================================================================
# Main Orchestration
# ============================================================================
main() {
    log "========================================="
    log "SafeClawArena Environment Reset"
    log "Mode: ${MODE}  DryRun: ${DRY_RUN}"
    log "========================================="

    acquire_lock

    case "$MODE" in
        snapshot)
            capture_snapshot
            log "Done. Baseline snapshot captured."
            ;;

        full)
            check_container
            stop_gateway

            # Clean all mutable state
            clean_memory
            clean_sessions
            clean_cron
            clean_exec_approvals
            clean_skills
            clean_extensions
            clean_logs
            clean_attack_artifacts

            # Restore baseline
            restore_config
            restore_workspace_bootstrap

            # Restart
            start_gateway
            wait_for_healthy

            log "========================================="
            log "Full reset complete."
            log "========================================="
            ;;

        soft)
            check_container
            stop_gateway

            # Clean only session/memory state
            clean_memory
            clean_sessions
            clean_cron_runs_only
            clean_exec_approvals
            clean_attack_artifacts

            # Restart (config preserved)
            start_gateway
            wait_for_healthy

            log "========================================="
            log "Soft reset complete."
            log "========================================="
            ;;

        setup-task)
            if [ -z "$TASK_FILE" ]; then
                log "ERROR: --setup-task requires a task JSON file path"
                exit 1
            fi

            # First do a full reset
            check_container
            stop_gateway
            clean_memory
            clean_sessions
            clean_cron
            clean_exec_approvals
            clean_skills
            clean_extensions
            clean_logs
            clean_attack_artifacts
            restore_config
            restore_workspace_bootstrap

            # Then provision task-specific environment
            setup_task_env "$TASK_FILE"

            # Start gateway (may need restart if config was modified)
            start_gateway
            wait_for_healthy || log "WARNING: Gateway health check timed out, continuing anyway"

            log "========================================="
            log "Task setup complete: $(basename "$TASK_FILE")"
            log "========================================="
            ;;

        *)
            log "ERROR: Unknown mode: $MODE"
            exit 1
            ;;
    esac
}

main
