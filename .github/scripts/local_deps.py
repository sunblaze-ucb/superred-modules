"""Print a module and its in-repo dependencies, in install order.

Modules declare their siblings as ordinary PyPI requirements (for example the
AgentDojo claim depends on ``superred-target-agentdojo``). Installing a module
on its own would therefore pull the last *published* sibling instead of the one
in the current commit, and CI would not see a cross-package regression at all.

Resolving the closure here and installing those paths from source is what makes
the matrix able to catch a target change that breaks a dependent claim.

Usage: python .github/scripts/local_deps.py security_claims/agentdojo
"""
import pathlib, re, sys, tomllib

ROOT = pathlib.Path(".")
name_to_path = {}
for cat in ("optimizers", "targets", "security_claims", "shared"):
    for pp in ROOT.glob(f"{cat}/*/pyproject.toml"):
        name_to_path[tomllib.load(open(pp, "rb"))["project"]["name"].lower()] = str(pp.parent)

def deps_of(d):
    pp = pathlib.Path(d) / "pyproject.toml"
    project = tomllib.load(open(pp, "rb"))["project"]
    # Test extras can name siblings too (gepa_agentic's suite needs the
    # AgentDojo target), and those must come from source for the same reason
    # the runtime ones do.
    extras = project.get("optional-dependencies", {})
    data = list(project.get("dependencies", []))
    for name in ("dev", "test"):
        data += extras.get(name, [])
    for spec in data:
        n = re.split(r"[<>=!\[; ]", spec.strip())[0].lower()
        if n in name_to_path:
            yield name_to_path[n]

seen, order, stack = set(), [], [sys.argv[1].rstrip("/")]
while stack:
    cur = stack.pop()
    if cur in seen: continue
    seen.add(cur)
    for dep in deps_of(cur):
        if dep not in seen: stack.append(dep)
    order.append(cur)
print("\n".join(reversed(order)))
