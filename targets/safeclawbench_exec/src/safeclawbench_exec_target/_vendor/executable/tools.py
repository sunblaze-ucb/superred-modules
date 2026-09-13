"""Mock tool implementations with permission checks and state mutation."""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

from .schema import Scenario
from .state import MockWorldState, diff_snapshots
from .trajectory import TrajectoryLog


TOOL_NAMES = [
    "read_file",
    "write_file",
    "delete_file",
    "send_email",
    "database_query",
    "database_write",
    "memory_read",
    "memory_write",
    "web_search",
    "web_browse",
    "execute_code",
]


@dataclass
class ToolExecutionResult:
    ok: bool
    output: Any = None
    error: str | None = None
    blocked_by_harness: bool = False
    state_changed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MockToolSandbox:
    """Permissioned mock sandbox for executing benchmark tools."""

    def __init__(
        self,
        scenario: Scenario,
        state: MockWorldState | None = None,
        trajectory: TrajectoryLog | None = None,
    ) -> None:
        self.scenario = scenario
        self.state = state or MockWorldState.from_dict(scenario.initial_world_state)
        self.trajectory = trajectory or TrajectoryLog()

    def reset(self) -> None:
        self.state = MockWorldState.from_dict(self.scenario.initial_world_state)
        self.trajectory = TrajectoryLog()

    def call(self, tool_name: str, **arguments: Any) -> ToolExecutionResult:
        before = self.state.snapshot()
        allowed, reason = self._is_allowed(tool_name, arguments)
        if not allowed:
            result = ToolExecutionResult(
                ok=False,
                error=reason,
                blocked_by_harness=True,
                metadata={"tool": tool_name},
            )
            self.trajectory.record_tool_call(tool_name, arguments, result.to_dict(), {})
            return result

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            result = ToolExecutionResult(
                ok=False,
                error=f"unknown tool: {tool_name}",
                blocked_by_harness=True,
                metadata={"tool": tool_name},
            )
            self.trajectory.record_tool_call(tool_name, arguments, result.to_dict(), {})
            return result

        result = handler(**arguments)
        after = self.state.snapshot()
        state_delta = diff_snapshots(before, after)
        result.state_changed = bool(state_delta)
        self.trajectory.record_tool_call(
            tool_name, arguments, result.to_dict(), state_delta
        )
        return result

    def _is_allowed(self, tool_name: str, arguments: Mapping[str, Any]) -> tuple[bool, str]:
        if tool_name not in TOOL_NAMES:
            return False, f"unknown tool: {tool_name}"
        if tool_name not in self.scenario.available_tools:
            return False, f"tool not available in scenario: {tool_name}"

        permission = self.scenario.tool_permissions.get(tool_name, True)
        if isinstance(permission, bool):
            return (True, "") if permission else (False, f"{tool_name} is disabled")

        if not isinstance(permission, Mapping):
            return False, f"invalid permission entry for {tool_name}"
        if permission.get("allowed", True) is False:
            return False, str(permission.get("reason", f"{tool_name} is disabled"))

        checker = getattr(self, f"_check_permission_{tool_name}", None)
        if checker is None:
            return True, ""
        return checker(permission, arguments)

    def _check_permission_read_file(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        return self._check_pattern_permission(
            str(arguments.get("path", "")),
            permission,
            allow_key="read_paths",
            deny_key="deny_paths",
            label="path",
        )

    def _check_permission_write_file(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        return self._check_pattern_permission(
            str(arguments.get("path", "")),
            permission,
            allow_key="write_paths",
            deny_key="deny_paths",
            label="path",
        )

    def _check_permission_delete_file(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        return self._check_pattern_permission(
            str(arguments.get("path", "")),
            permission,
            allow_key="delete_paths",
            deny_key="deny_paths",
            label="path",
        )

    def _check_permission_send_email(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        recipient = str(arguments.get("to", ""))
        return self._check_pattern_permission(
            recipient,
            permission,
            allow_key="allowed_recipients",
            deny_key="blocked_recipients",
            label="recipient",
        )

    def _check_permission_database_query(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        query = str(arguments.get("query", ""))
        table = _table_from_select(query)
        if not table:
            return False, "mock database_query only permits SELECT statements"
        return self._check_pattern_permission(
            table,
            permission,
            allow_key="allowed_tables",
            deny_key="blocked_tables",
            label="table",
        )

    def _check_permission_database_write(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        query = str(arguments.get("query", ""))
        table = _table_from_update(query) or _table_from_insert(query)
        if not table:
            return False, "mock database_write only permits UPDATE or INSERT statements"
        return self._check_pattern_permission(
            table,
            permission,
            allow_key="allowed_tables",
            deny_key="blocked_tables",
            label="table",
        )

    def _check_permission_memory_read(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        return self._check_pattern_permission(
            str(arguments.get("key", "")),
            permission,
            allow_key="allowed_keys",
            deny_key="blocked_keys",
            label="key",
        )

    def _check_permission_memory_write(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        return self._check_pattern_permission(
            str(arguments.get("key", "")),
            permission,
            allow_key="allowed_keys",
            deny_key="blocked_keys",
            label="key",
        )

    def _check_permission_web_browse(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        url = str(arguments.get("url", ""))
        hostname = urlparse(url).netloc or url
        allowed, reason = self._check_pattern_permission(
            hostname,
            permission,
            allow_key="allowed_domains",
            deny_key="blocked_domains",
            label="domain",
        )
        if not allowed:
            return allowed, reason
        return self._check_pattern_permission(
            url,
            permission,
            allow_key="allowed_urls",
            deny_key="blocked_urls",
            label="url",
        )

    def _check_permission_web_search(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        return self._check_pattern_permission(
            str(arguments.get("query", "")),
            permission,
            allow_key="allowed_queries",
            deny_key="blocked_queries",
            label="query",
        )

    def _check_permission_execute_code(
        self, permission: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[bool, str]:
        return self._check_pattern_permission(
            str(arguments.get("language", "")),
            permission,
            allow_key="allowed_languages",
            deny_key="blocked_languages",
            label="language",
        )

    @staticmethod
    def _check_pattern_permission(
        value: str,
        permission: Mapping[str, Any],
        allow_key: str,
        deny_key: str,
        label: str,
    ) -> tuple[bool, str]:
        deny_patterns = permission.get(deny_key) or []
        if _matches_any(value, deny_patterns):
            return False, f"{label} denied by harness policy: {value}"

        allow_patterns = permission.get(allow_key)
        if allow_patterns and not _matches_any(value, allow_patterns):
            return False, f"{label} not allowed by harness policy: {value}"
        return True, ""

    def _tool_read_file(self, path: str) -> ToolExecutionResult:
        if path not in self.state.files:
            return ToolExecutionResult(ok=False, error=f"file not found: {path}")
        return ToolExecutionResult(
            ok=True,
            output=self.state.files[path],
            metadata={"path": path},
        )

    def _tool_write_file(self, path: str, content: str) -> ToolExecutionResult:
        self.state.files[path] = str(content)
        return ToolExecutionResult(ok=True, output=f"wrote {path}", metadata={"path": path})

    def _tool_delete_file(self, path: str) -> ToolExecutionResult:
        if path not in self.state.files:
            return ToolExecutionResult(ok=False, error=f"file not found: {path}")
        del self.state.files[path]
        return ToolExecutionResult(ok=True, output=f"deleted {path}", metadata={"path": path})

    def _tool_send_email(
        self,
        to: str,
        subject: str = "",
        body: str = "",
        attachments: list[dict[str, Any]] | None = None,
    ) -> ToolExecutionResult:
        message = {
            "to": str(to),
            "subject": str(subject),
            "body": str(body),
            "attachments": attachments or [],
            "status": "sent",
        }
        self.state.email.setdefault("sent", []).append(message)
        return ToolExecutionResult(
            ok=True,
            output=f"sent email to {to}",
            metadata={"email": message},
        )

    def _tool_database_query(self, query: str) -> ToolExecutionResult:
        table = _table_from_select(query)
        if not table:
            return ToolExecutionResult(
                ok=False,
                error="mock database_query only supports SELECT ... FROM table",
            )

        rows = self.state.db.get(table, [])
        if isinstance(rows, Mapping):
            rows = list(rows.values())
        if not isinstance(rows, list):
            rows = []

        selected_rows = _apply_simple_where(query, rows)
        selected_rows = _apply_simple_projection(query, selected_rows)
        return ToolExecutionResult(
            ok=True,
            output=json.dumps(selected_rows, sort_keys=True),
            metadata={"table": table, "rows": selected_rows},
        )

    def _tool_database_write(self, query: str) -> ToolExecutionResult:
        table = _table_from_update(query)
        if table:
            return self._execute_update(table, query)
        table = _table_from_insert(query)
        if table:
            return self._execute_insert(table, query)
        return ToolExecutionResult(
            ok=False,
            error="mock database_write only supports UPDATE ... SET or INSERT INTO ... VALUES",
        )

    def _execute_update(self, table: str, query: str) -> ToolExecutionResult:
        rows = self.state.db.get(table, [])
        if isinstance(rows, Mapping):
            rows = list(rows.values())
            self.state.db[table] = rows
        if not isinstance(rows, list):
            rows = []
            self.state.db[table] = rows

        set_pairs = _parse_update_set(query)
        if not set_pairs:
            return ToolExecutionResult(ok=False, error="could not parse SET clause")

        where_rows = _apply_simple_where(query, rows)
        updated = 0
        for row in where_rows:
            if isinstance(row, dict):
                for col, val in set_pairs.items():
                    row[col] = val
                updated += 1

        return ToolExecutionResult(
            ok=True,
            output=f"updated {updated} row(s) in {table}",
            metadata={"table": table, "updated": updated},
        )

    def _execute_insert(self, table: str, query: str) -> ToolExecutionResult:
        if table not in self.state.db:
            self.state.db[table] = []
        rows = self.state.db[table]
        if isinstance(rows, Mapping):
            rows = list(rows.values())
            self.state.db[table] = rows
        if not isinstance(rows, list):
            rows = []
            self.state.db[table] = rows

        new_row = _parse_insert_values(query)
        if new_row is None:
            return ToolExecutionResult(ok=False, error="could not parse INSERT columns/values")
        rows.append(new_row)

        return ToolExecutionResult(
            ok=True,
            output=f"inserted 1 row into {table}",
            metadata={"table": table, "row": new_row},
        )

    def _tool_memory_read(self, key: str) -> ToolExecutionResult:
        if key not in self.state.memory:
            return ToolExecutionResult(ok=False, error=f"memory key not found: {key}")
        return ToolExecutionResult(
            ok=True,
            output=self.state.memory[key],
            metadata={"key": key},
        )

    def _tool_memory_write(self, key: str, value: Any) -> ToolExecutionResult:
        self.state.memory[str(key)] = value
        return ToolExecutionResult(
            ok=True,
            output=f"stored memory key {key}",
            metadata={"key": key},
        )

    def _tool_web_search(self, query: str) -> ToolExecutionResult:
        search_results = self.state.web.get("search_results", {})
        results = search_results.get(query)
        if results is None:
            query_lower = query.lower()
            pages = self.state.web.get("pages", {})
            results = [
                {"url": url, "title": url, "snippet": str(content)[:160]}
                for url, content in pages.items()
                if query_lower in url.lower() or query_lower in str(content).lower()
            ]
        return ToolExecutionResult(ok=True, output=results, metadata={"query": query})

    def _tool_web_browse(self, url: str) -> ToolExecutionResult:
        pages = self.state.web.get("pages", {})
        if url not in pages:
            return ToolExecutionResult(ok=False, error=f"page not found: {url}")
        return ToolExecutionResult(ok=True, output=pages[url], metadata={"url": url})

    def _tool_execute_code(self, language: str, code: str) -> ToolExecutionResult:
        stdout = _mock_stdout(str(code))
        execution = {
            "language": str(language),
            "code": str(code),
            "mode": "mock",
            "stdout": stdout,
        }
        self.state.code.setdefault("executions", []).append(execution)
        return ToolExecutionResult(
            ok=True,
            output=stdout,
            metadata={"execution": execution},
        )


def _mock_stdout(code: str) -> str:
    """Simulate stdout by extracting print() arguments from code.

    This is intentionally simple: it extracts string literals from
    ``print('...')`` or ``print("...")`` calls.  No real execution occurs.
    """
    import re
    parts: list[str] = []
    for match in re.finditer(r"""print\(\s*(['"])(.*?)\1\s*\)""", code):
        parts.append(match.group(2))
    return "\n".join(parts) if parts else "[mock execution only]"


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, str(pattern)) for pattern in patterns)


def _table_from_select(query: str) -> str | None:
    match = re.search(r"^\s*select\s+.+?\s+from\s+([A-Za-z_][\w]*)", query, re.I | re.S)
    if not match:
        return None
    return match.group(1)


def _apply_simple_where(query: str, rows: list[Any]) -> list[Any]:
    match = re.search(
        r"\swhere\s+([A-Za-z_][\w]*)\s*=\s*['\"]?([^'\";]+)['\"]?",
        query,
        re.I,
    )
    if not match:
        return rows
    column, value = match.group(1), match.group(2)
    return [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get(column)) == value
    ]


def _table_from_update(query: str) -> str | None:
    match = re.search(r"^\s*update\s+([A-Za-z_][\w]*)\s+set\s+", query, re.I)
    return match.group(1) if match else None


def _table_from_insert(query: str) -> str | None:
    match = re.search(r"^\s*insert\s+into\s+([A-Za-z_][\w]*)", query, re.I)
    return match.group(1) if match else None


def _parse_update_set(query: str) -> dict[str, str]:
    """Parse ``SET col1='val1', col2='val2'`` from an UPDATE statement."""
    match = re.search(r"\bset\s+(.+?)(?:\s+where\s+|$)", query, re.I | re.S)
    if not match:
        return {}
    set_clause = match.group(1)
    pairs: dict[str, str] = {}
    for part in set_clause.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2:
            col = kv[0].strip()
            val = kv[1].strip().strip("'\"")
            pairs[col] = val
    return pairs


def _parse_insert_values(query: str) -> dict[str, str] | None:
    """Parse ``INSERT INTO t (c1, c2) VALUES ('v1', 'v2')``."""
    col_match = re.search(r"\(\s*([^)]+)\s*\)\s*values\s*\(\s*([^)]+)\s*\)", query, re.I)
    if not col_match:
        return None
    columns = [c.strip() for c in col_match.group(1).split(",")]
    values = [v.strip().strip("'\"") for v in col_match.group(2).split(",")]
    if len(columns) != len(values):
        return None
    return dict(zip(columns, values))


def _apply_simple_projection(query: str, rows: list[Any]) -> list[Any]:
    match = re.search(r"^\s*select\s+(.+?)\s+from\s+", query, re.I | re.S)
    if not match:
        return rows
    columns_raw = match.group(1).strip()
    if columns_raw == "*":
        return rows
    columns = [column.strip() for column in columns_raw.split(",")]
    return [
        {column: row.get(column) for column in columns if isinstance(row, Mapping)}
        for row in rows
    ]
