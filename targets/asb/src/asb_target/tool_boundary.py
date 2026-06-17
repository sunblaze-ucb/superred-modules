"""Map a tool name to its OPI trust-boundary tag.

Each ASB normal tool is its own backing system, so OPI on that tool's
observation fires tagged to the tool's own ``tools.<name>`` leaf. Unknown
tools, and the per-task attacker tool (which is not a fixed backing system),
fire at the ``tools`` root. This is a mechanical lookup over
:data:`asb_target.security_tags.TOOL_OBSERVATION_TAGS`, with no hand-authored
grouping: the upstream-faithful, provenance-blind OPI scope is simply
``{tools}`` (the root subsumes every leaf).
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomainTag

from asb_target.security_tags import TOOL_OBSERVATION_TAGS, TOOLS_TAG


def tool_boundary_tag(tool_name: str) -> SecurityDomainTag:
    """Return the OPI trust-boundary tag for *tool_name*.

    Known normal tools map to their ``tools.<name>`` leaf; the attacker tool
    and any unknown tool map to the ``tools`` root (still in scope for any
    ``{tools}`` attacker, faithful to upstream's provenance-blind OPI).
    """
    return TOOL_OBSERVATION_TAGS.get(tool_name, TOOLS_TAG)


__all__ = ["tool_boundary_tag"]
