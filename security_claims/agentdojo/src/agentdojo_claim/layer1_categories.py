"""Per-suite injection-task category labels (refined attack-semantic axis).

User-confirmed taxonomy 2026-05-15:

- **Banking** (9 tasks, 4 categories): ``exfil_via_memo`` (IT 0,1,2,3,8),
  ``unauth_transfer`` (IT 5,6), ``recurring_redirect`` (IT 4),
  ``credential_change`` (IT 7).
- **Workspace** (6 tasks, 5 categories): ``unauth_email_send`` (IT 0),
  ``email_exfil`` (IT 3), ``twofa_exfil`` (IT 4,5),
  ``file_deletion`` (IT 1), ``calendar_manip`` (IT 2).
- **Slack** (5 tasks, 4 categories): ``phishing`` (IT 1),
  ``exfil_to_web`` (IT 2,4), ``drive_by`` (IT 3),
  ``membership_manip`` (IT 5).
- **Travel** (7 tasks, 4 categories): ``unauth_booking`` (IT 0,4),
  ``pii_exfil`` (IT 1,3,5), ``calendar_manip`` (IT 2),
  ``output_manip`` (IT 6).

Total: 27 injection tasks across 17 categories.
"""

from __future__ import annotations

INJECTION_CATEGORIES: dict[tuple[str, str], str] = {
    # ---- banking ----
    ("banking", "injection_task_0"): "exfil_via_memo",
    ("banking", "injection_task_1"): "exfil_via_memo",
    ("banking", "injection_task_2"): "exfil_via_memo",
    ("banking", "injection_task_3"): "exfil_via_memo",
    ("banking", "injection_task_8"): "exfil_via_memo",
    ("banking", "injection_task_5"): "unauth_transfer",
    ("banking", "injection_task_6"): "unauth_transfer",
    ("banking", "injection_task_4"): "recurring_redirect",
    ("banking", "injection_task_7"): "credential_change",
    # ---- workspace ----
    ("workspace", "injection_task_0"): "unauth_email_send",
    ("workspace", "injection_task_3"): "email_exfil",
    ("workspace", "injection_task_4"): "twofa_exfil",
    ("workspace", "injection_task_5"): "twofa_exfil",
    ("workspace", "injection_task_1"): "file_deletion",
    ("workspace", "injection_task_2"): "calendar_manip",
    # ---- slack ----
    ("slack", "injection_task_1"): "phishing",
    ("slack", "injection_task_2"): "exfil_to_web",
    ("slack", "injection_task_4"): "exfil_to_web",
    ("slack", "injection_task_3"): "drive_by",
    ("slack", "injection_task_5"): "membership_manip",
    # ---- travel ----
    ("travel", "injection_task_0"): "unauth_booking",
    ("travel", "injection_task_4"): "unauth_booking",
    ("travel", "injection_task_1"): "pii_exfil",
    ("travel", "injection_task_3"): "pii_exfil",
    ("travel", "injection_task_5"): "pii_exfil",
    ("travel", "injection_task_2"): "calendar_manip",
    ("travel", "injection_task_6"): "output_manip",
}
"""Static (suite, injection_task_id) -> category label map.  17 distinct
labels across 27 entries."""

CATEGORIES_BY_SUITE: dict[str, frozenset[str]] = {
    "banking": frozenset({
        "exfil_via_memo", "unauth_transfer",
        "recurring_redirect", "credential_change",
    }),
    "workspace": frozenset({
        "unauth_email_send", "email_exfil", "twofa_exfil",
        "file_deletion", "calendar_manip",
    }),
    "slack": frozenset({
        "phishing", "exfil_to_web", "drive_by", "membership_manip",
    }),
    "travel": frozenset({
        "unauth_booking", "pii_exfil", "calendar_manip", "output_manip",
    }),
}
"""Per-suite set of categories present, for per-category factory
discovery without re-iterating INJECTION_CATEGORIES."""

ALL_CATEGORIES: frozenset[str] = frozenset(
    cat for cats in CATEGORIES_BY_SUITE.values() for cat in cats
)
"""Union of all category labels.  17 entries (note: ``calendar_manip``
appears in both workspace and travel, but is a single label)."""


def category_of(suite: str, injection_task_id: str) -> str:
    """Return the category label for one (suite, injection_task) pair.

    Raises:
        KeyError: If the pair is unknown.
    """
    return INJECTION_CATEGORIES[(suite, injection_task_id)]


__all__ = [
    "INJECTION_CATEGORIES",
    "CATEGORIES_BY_SUITE",
    "ALL_CATEGORIES",
    "category_of",
]
