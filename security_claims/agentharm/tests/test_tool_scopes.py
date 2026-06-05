"""Unit tests for the AgentHarm tool trust-boundary scopes (offline)."""

from __future__ import annotations

from inspect_agent_target import TOOLS_TAG, InspectAgentTarget
from security_claim_agentharm.agentharm_tools import ALL_HARMFUL_TOOL_NAMES
from security_claim_agentharm.tool_scopes import TOOL_SCOPES


def test_covers_exactly_the_80_harmful_tools() -> None:
    assert set(TOOL_SCOPES) == set(ALL_HARMFUL_TOOL_NAMES)
    assert len(TOOL_SCOPES) == 80


def test_every_scope_is_a_leaf_under_the_tools_root() -> None:
    for name, tag in TOOL_SCOPES.items():
        chain = []
        cur = tag
        while cur is not None:
            chain.append(cur.name)
            cur = cur.parent
        assert chain[-1] == "tools", f"{name} -> {chain}"
        assert TOOLS_TAG.includes(tag), name


def test_28_distinct_leaf_boundaries() -> None:
    assert len({tag.name for tag in TOOL_SCOPES.values()}) == 28


def test_tools_sharing_a_system_share_a_scope() -> None:
    x_names = [
        "follow_x_account",
        "get_direct_messages_x",
        "get_latest_posts_on_x",
        "post_on_x",
        "post_x_reply",
        "query_posts_on_x",
        "query_x_user_info",
        "query_x_users",
        "report_x_post",
        "repost_x_post",
        "send_direct_message_x",
    ]
    assert {TOOL_SCOPES[n].name for n in x_names} == {"x"}


def test_third_party_reads_are_scoped_to_the_external_system() -> None:
    # "the user's own account" reads that actually return third-party content.
    assert TOOL_SCOPES["get_latest_emails"].name == "email"
    assert TOOL_SCOPES["get_paypal_transactions"].name == "paypal"
    assert TOOL_SCOPES["get_all_login_data"].name == "bank"
    # genuinely first-party stays first-party.
    assert TOOL_SCOPES["read_file"].name == "local_machine"
    assert TOOL_SCOPES["get_contacts"].name == "own_directories"


def test_target_builds_valid_domain_and_scopes_controllables() -> None:
    t = InspectAgentTarget(
        model="m",
        tool_resolver=lambda n: None,  # type: ignore[arg-type,return-value]
        tool_scopes=TOOL_SCOPES,
        api_base="b",
        api_key="k",
    )
    # 14 fixed tags + 34 boundary tags (28 leaves + 6 intermediate nodes).
    assert len(t.security_domain._tags) == 48
    assert {r.name for r in t.security_domain.roots} == {"system", "user", "tools"}
    t.set_config("tool_names", '["get_webpage", "post_on_x"]')
    by_name = {c.name: c for c in t.get_controllables()}
    assert by_name["tool:get_webpage"].security_domain.name == "web"
    assert by_name["tool:post_on_x"].security_domain.name == "x"
