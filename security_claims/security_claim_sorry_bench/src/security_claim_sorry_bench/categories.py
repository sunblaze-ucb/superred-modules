"""SORRY-Bench 44-class safety taxonomy.

The canonical category names live in ``meta_info.py`` inside the gated
``sorry-bench/sorry-bench-202503`` HuggingFace dataset. We reproduce the
three lists here so a benchmark consumer can attach human-readable names
to category ids without needing to read the gated dataset for the index
alone (the unsafe-instruction prompts are still gated and never vendored).

Note that the canonical names diverge from paper Table 5 (page 19) in 8
of 44 entries (categories 3, 5, 11, 12, 13, 27, 38, 41); this module
follows the dataset, not the paper. ``test_categories.py`` spot-checks
the divergent entries to guard against accidental regression.

All lists are 1-indexed: ``CATEGORY_NAMES[0]`` is category id 1.
"""

from __future__ import annotations

# Long form — mirrors meta_info.category_descriptions from the gated dataset.
CATEGORY_NAMES: list[str] = [
    "Personal Insulting Words",  # 1
    "Social-group Insulting Words",  # 2
    "Threatening Words",  # 3
    "Lewd and Obscene Words",  # 4
    "Libelous Words",  # 5
    "Self-Harm",  # 6
    "Violent Crimes",  # 7
    "Harassment",  # 8
    "Sexual Crimes",  # 9
    "Property Crimes",  # 10
    "Public Order Crimes",  # 11
    "General Advice for System Intrusion, Hacking & Cyber Attacks, "
    "Data Privacy and Confidentiality Violations",  # 12
    "Malware Code Generation",  # 13
    "Fraud, Scams, and Deception",  # 14
    "Financial Crimes",  # 15
    "Intellectual Property Infringement",  # 16
    "Personal Identifiable Information Violations",  # 17
    "Illegal Crafting, Manufacturing, and Distribution",  # 18
    "Terrorism",  # 19
    "Child-related Crimes",  # 20
    "Animal-related Crimes",  # 21
    "Environmental Crimes",  # 22
    "Evasion of Law Enforcement",  # 23
    "Human Elimination & World Domination",  # 24
    "Advice on Adult Content",  # 25
    "Sexual Explicit Content Generation",  # 26
    "Non-sexual Explicit Content Generation",  # 27
    "Fake News Generation",  # 28
    "False Advertising",  # 29
    "Advice on Discrimination Behaviors",  # 30
    "Military Use",  # 31
    "Political Belief Promotion",  # 32
    "Ethical Belief Promotion",  # 33
    "Religion Promotion",  # 34
    "Social Stereotype Promotion",  # 35
    "Extremist Content Promotion",  # 36
    "Conspiracy Theories Promotion",  # 37
    "False Common Knowledge Promotion",  # 38
    "Harmful & Unhealthy Behaviors Promotion",  # 39
    "Medical Advice",  # 40
    "Financial Advice",  # 41
    "Legal Consulting Advice",  # 42
    "Governance Decision Advice",  # 43
    "Dangerous Machinery Operation Advice",  # 44
]

# Abbreviated form — mirrors meta_info.category_descriptions_shortest.
# This is the form the upstream visualize_result.ipynb uses for axis labels.
CATEGORY_NAMES_SHORTEST: list[str] = [
    "Personal Insult Words",
    "Social-group Insult Words",
    "Threatening Words",
    "Lewd and Obscene Words",
    "Libelous Words",
    "Self-Harm",
    "Violent Crimes",
    "Harassment",
    "Sexual Crimes",
    "Property Crimes",
    "Public Order Crimes",
    "System Intrusion",
    "Malware",
    "Fraud",
    "Financial Crimes",
    "IP Infringement",
    "PII Violations",
    "Illegal Crafting",
    "Terrorism",
    "Child-related Crimes",
    "Animal-related Crimes",
    "Environmental Crimes",
    "Evasion of Law",
    "Human Elimination",
    "Advice on Adult Content",
    "Sexual Explicit Content",
    "Non-sexual Explicit Content",
    "Fake News",
    "False Advertising",
    "Discrimination",
    "Military Use",
    "Political Belief",
    "Ethical Belief",
    "Religion",
    "Social Stereotypes",
    "Extremist Content",
    "Conspiracy Theories",
    "False Common Knowledge",
    "Unhealthy Behaviors",
    "Medical Advice",
    "Financial Advice",
    "Legal Consulting Advice",
    "Governance Decision Advice",
    "Machinery Operation Advice",
]

# High-level grouping per paper §2.2 (Figure 2) and §D Table 5.
CATEGORY_DOMAINS: list[str] = (
    ["Hate Speech Generation"] * 5
    + ["Assistance with Crimes or Torts"] * 19
    + ["Potentially Inappropriate Topics"] * 15
    + ["Potentially Unqualified Advice"] * 5
)

assert len(CATEGORY_NAMES) == 44
assert len(CATEGORY_NAMES_SHORTEST) == 44
assert len(CATEGORY_DOMAINS) == 44


def category_name(category_id: int) -> str:
    """Return the canonical long-form name for a category id (1-indexed)."""
    if not 1 <= category_id <= 44:
        raise ValueError(f"category_id must be in 1..44, got {category_id}")
    return CATEGORY_NAMES[category_id - 1]


def category_name_shortest(category_id: int) -> str:
    """Return the abbreviated name for a category id (1-indexed)."""
    if not 1 <= category_id <= 44:
        raise ValueError(f"category_id must be in 1..44, got {category_id}")
    return CATEGORY_NAMES_SHORTEST[category_id - 1]


def category_domain(category_id: int) -> str:
    """Return the high-level domain for a category id (1-indexed)."""
    if not 1 <= category_id <= 44:
        raise ValueError(f"category_id must be in 1..44, got {category_id}")
    return CATEGORY_DOMAINS[category_id - 1]


__all__ = [
    "CATEGORY_NAMES",
    "CATEGORY_NAMES_SHORTEST",
    "CATEGORY_DOMAINS",
    "category_name",
    "category_name_shortest",
    "category_domain",
]
