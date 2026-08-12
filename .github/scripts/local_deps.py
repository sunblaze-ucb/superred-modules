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


def canon(name):
    """PEP 503 normalized name: runs of - _ . collapse to a single -.

    ``superred_target_agentdojo`` and ``superred-target-agentdojo`` are the same
    distribution to pip. Matching on the raw string instead means a sibling
    spelled the other way is not recognised as in-repo, so it silently installs
    from PyPI -- the exact failure this script exists to prevent, and a silent
    one, because the install still succeeds.
    """
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


name_to_path = {}
for cat in ("optimizers", "targets", "security_claims", "shared"):
    for pp in ROOT.glob(f"{cat}/*/pyproject.toml"):
        name_to_path[canon(tomllib.load(open(pp, "rb"))["project"]["name"])] = str(pp.parent)

def deps_of(d):
    pp = pathlib.Path(d) / "pyproject.toml"
    project = tomllib.load(open(pp, "rb"))["project"]
    # The test extra can name siblings too (gepa_agentic's suite needs the
    # AgentDojo target), and those must come from source for the same reason
    # the runtime ones do.
    data = list(project.get("dependencies", []))
    data += project.get("optional-dependencies", {}).get("test", [])
    for spec in data:
        # Strip the version specifier, extras and environment marker. ~ and ,
        # matter: "pkg~=1.0" and "pkg>=1,<2" would otherwise keep a trailing
        # character and miss the lookup.
        n = canon(re.split(r"[<>=!~,\[;( ]", spec.strip())[0])
        if n in name_to_path:
            yield name_to_path[n]

# Post-order DFS: a package is emitted only after everything it depends on.
# Reversing a pre-order walk is not equivalent -- it holds for a chain, but not
# once two packages share a dependency. With A -> {B, C} and B -> C, the
# pre-order walk reaches C before B, and reversing puts B first, so pip is asked
# to install B while C is still unbuilt and resolves C from PyPI instead.
state, order = {}, []


def visit(node):
    if state.get(node) == "done":
        return
    if state.get(node) == "visiting":
        # A dependency cycle between in-repo packages. No order satisfies it;
        # emit what we have rather than recursing forever and let pip report it.
        print(f"warning: dependency cycle at {node}", file=sys.stderr)
        return
    state[node] = "visiting"
    for dep in deps_of(node):
        visit(dep)
    state[node] = "done"
    order.append(node)


visit(sys.argv[1].rstrip("/"))
print("\n".join(order))
