"""Vendored byte-identical from the upstream Attack Anything (SEATS) codebase.

See ../../../ASSUMPTIONS.md for provenance and the byte-identical-asset ledger.
These modules are pinned verbatim; do not edit. The optimizer reuses their
helper pieces (tree, archive, operators, decomposition, feedback, judges,
reward math) and reimplements only the event-driven orchestration that the
superred controller inverts.
"""
