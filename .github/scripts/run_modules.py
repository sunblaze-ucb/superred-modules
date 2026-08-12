"""Install and test a set of modules, one fresh venv each, in a single job.

The per-module venv is the point: each module is installed the way a consumer
gets it (its in-repo dependency closure from source via local_deps.py,
everything else from PyPI), and no sibling's install can mask a missing or
wrong dependency. Consolidating modules into one job per category -- rather
than one job per module -- keeps that isolation while cutting the job count
from ~36 to 4, so a push does not fan out past private-repo concurrency
limits or pay 36 checkout/setup overheads.

Failures do not stop the run: every module in the set is tested and the
script exits non-zero at the end if any failed, mirroring fail-fast: false.

Usage:
  python .github/scripts/run_modules.py <category>... [--shard K/N]
  python .github/scripts/run_modules.py --heavy
"""

import os
import pathlib
import subprocess
import sys
import tomllib

# gptfuzzer pulls torch/transformers; dra pins detoxify==0.5.1, whose
# tokenizers dependency has no 3.13 wheel and fails to build from source, so
# the module cannot be installed on the version it claims to support
# (requires-python >=3.11,<3.14). Excluded from the per-push categories and
# run by the opt-in job instead; dra's install failure there is the point,
# visible rather than silent.
HEAVY = ["optimizers/gptfuzzer", "optimizers/dra"]

CATEGORIES = ("optimizers", "targets", "security_claims", "shared")


def discover(category: str) -> list[str]:
    return sorted(
        str(p.parent)
        for p in pathlib.Path(category).glob("*/pyproject.toml")
        if (p.parent / "tests").is_dir() and str(p.parent) not in HEAVY
    )


def run(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, "CI": "true"})


def test_module(mod: str) -> bool:
    # Absolute: the pytest step below runs with cwd=mod, where a relative
    # interpreter path would no longer resolve.
    venv = (pathlib.Path(".ci-venvs") / mod.replace("/", "-")).resolve()
    vpy = str(venv / "bin" / "python")
    try:
        run([sys.executable, "-m", "venv", "--clear", str(venv)])
        # Siblings are declared as ordinary PyPI requirements, so a bare
        # install would pull the last *published* sibling instead of the one
        # in this commit; local_deps.py emits the in-repo closure in install
        # order and those paths come from source.
        closure = subprocess.run(
            [sys.executable, ".github/scripts/local_deps.py", mod],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        for pkg in closure:
            run([vpy, "-m", "pip", "install", "-q", "-e", pkg])
        # Several modules set an addopts with --cov, so pytest-cov has to come
        # from the module's test extra (or the fixed list below).
        extras = tomllib.load(open(pathlib.Path(mod) / "pyproject.toml", "rb"))[
            "project"
        ].get("optional-dependencies", {})
        if "test" in extras:
            run([vpy, "-m", "pip", "install", "-q", "-e", f"{mod}[test]"])
        run([vpy, "-m", "pip", "install", "-q", "pytest", "pytest-asyncio", "pytest-cov"])
        # HarmBench's copyright matcher needs a spaCy model that cannot be a
        # wheel dependency; its README documents this step, mirror it here.
        if mod == "security_claims/harmbench":
            run([vpy, "-m", "spacy", "download", "en_core_web_sm"])
        run([vpy, "-m", "pytest", "tests", "-q"], cwd=mod)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--heavy":
        mods = list(HEAVY)
    else:
        shard = None
        if "--shard" in args:
            at = args.index("--shard")
            shard = args[at + 1]
            args = args[:at] + args[at + 2 :]
        for category in args:
            if category not in CATEGORIES:
                print(f"unknown category {category!r}", file=sys.stderr)
                return 2
        mods = [m for category in args for m in discover(category)]
        if shard is not None:
            k, n = (int(x) for x in shard.split("/"))
            mods = [m for i, m in enumerate(mods) if i % n == k - 1]
    results: dict[str, bool] = {}
    for mod in mods:
        print(f"::group::{mod}", flush=True)
        results[mod] = test_module(mod)
        print("::endgroup::", flush=True)
        if not results[mod]:
            print(f"::error::{mod} failed", flush=True)
    print()
    for mod, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {mod}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
