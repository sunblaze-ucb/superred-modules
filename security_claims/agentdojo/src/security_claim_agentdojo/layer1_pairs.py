"""Canonical 27-pair scope for Layer 1.

Each injection task is paired with one user task whose ground_truth
invokes a read tool that surfaces the injection task's required slot,
so a no-attacker run completes successfully and the attacker has a
realistic injection channel.

Pairing chosen by reading each user task's ground_truth and each
injection task's slot dependence (see notes/agentdojo_suites_*.md).
Where multiple user tasks satisfy the constraint, the lowest user_task
id is chosen for determinism.

The faithfulness verification sweep replays the upstream cross-product
and validates that the per-pair (utility, security) booleans match;
this canonical scope is just the per-claim default — callers can pass
an explicit ``pairs=...`` to ``agentdojo_layer1_claim`` for any other
scope including the full 629-case cross-product.
"""

from __future__ import annotations

# Each tuple is (suite, user_task_id, injection_task_id).
CANONICAL_PAIRS: tuple[tuple[str, str, str], ...] = (
    # ---- banking: 9 injection tasks ----
    # IT 0/1/2/3/8 (exfil_via_memo): need get_most_recent_transactions
    # to surface the injection_incoming_transaction slot.  UT 1 (total
    # spending) calls it as its only read.
    ("banking", "user_task_1", "injection_task_0"),
    ("banking", "user_task_1", "injection_task_1"),
    ("banking", "user_task_1", "injection_task_2"),
    ("banking", "user_task_1", "injection_task_3"),
    ("banking", "user_task_1", "injection_task_8"),
    # IT 4 (recurring_redirect): needs get_scheduled_transactions.  UT 2
    # (rent payment adjustment) reads scheduled txns.
    ("banking", "user_task_2", "injection_task_4"),
    # IT 5/6 (unauth_transfer): need any read surfacing instructions.
    # UT 1 (get_most_recent_transactions) is fine; the transaction-
    # subject slot delivers the injection.
    ("banking", "user_task_1", "injection_task_5"),
    ("banking", "user_task_1", "injection_task_6"),
    # IT 7 (credential_change): the slot is in the bill text; UT 0 reads
    # 'bill-december-2023.txt' via read_file.
    ("banking", "user_task_0", "injection_task_7"),

    # ---- workspace: 6 injection tasks ----
    # IT 0 (unauth_email_send), IT 3 (email_exfil from lily.white), and
    # IT 4/5 (2fa_exfil from security@facebook.com) all use email-body
    # injection slots.  UT 14 (family reunion date) calls search_emails
    # and surfaces inbox content.  UT 16 (Facebook code) surfaces the
    # facebook injection slot.
    ("workspace", "user_task_14", "injection_task_0"),
    ("workspace", "user_task_14", "injection_task_3"),
    ("workspace", "user_task_16", "injection_task_4"),
    ("workspace", "user_task_16", "injection_task_5"),
    # IT 1 (file_deletion): file content slots; UT 26 (largest file name)
    # calls list_files which surfaces drive injection slots.
    ("workspace", "user_task_26", "injection_task_1"),
    # IT 2 (calendar_manip): event-description slot; UT 0 reads
    # 'Networking event' calendar entry.
    ("workspace", "user_task_0", "injection_task_2"),

    # ---- slack: 5 injection tasks ----
    # IT 1/3 (phishing/drive_by): page-content injection slots; UT 0
    # fetches www.informations.com.
    ("slack", "user_task_0", "injection_task_1"),
    ("slack", "user_task_0", "injection_task_3"),
    # IT 2/4 (exfil_to_web): channel-message injection slots; UT 1
    # reads general channel.
    ("slack", "user_task_1", "injection_task_2"),
    ("slack", "user_task_1", "injection_task_4"),
    # IT 5 (membership_manip): trace-only scoring via invite/remove.
    # UT 2 (invite Dora) surfaces the dora website injection slot.
    ("slack", "user_task_2", "injection_task_5"),

    # ---- travel: 7 injection tasks ----
    # All travel injection slots are in hotel/restaurant/car-rental
    # review fields.  UT 0 reads hotel reviews for "Le Marais Boutique"
    # (injection_hotels_0); other tasks reach broader review surfaces.
    ("travel", "user_task_0", "injection_task_0"),
    ("travel", "user_task_3", "injection_task_1"),  # Paris hotels w/ reviews
    ("travel", "user_task_0", "injection_task_2"),
    ("travel", "user_task_3", "injection_task_3"),
    ("travel", "user_task_3", "injection_task_4"),
    ("travel", "user_task_3", "injection_task_5"),
    ("travel", "user_task_0", "injection_task_6"),
)
"""27 canonical (suite, user_task_id, injection_task_id) pairs.

Each injection task in v1 is covered exactly once.  Refine bindings by
passing ``pairs=`` to :func:`agentdojo_layer1_claim` for non-default
scopes (e.g. the full 629-case cross-product for faithfulness sweeps).
"""


SUITES_IN_ORDER: tuple[str, ...] = ("banking", "workspace", "slack", "travel")


def pairs_for_suite(suite: str) -> tuple[tuple[str, str, str], ...]:
    """Return only the canonical pairs whose first element is *suite*."""
    return tuple(p for p in CANONICAL_PAIRS if p[0] == suite)


__all__ = ["CANONICAL_PAIRS", "SUITES_IN_ORDER", "pairs_for_suite"]
