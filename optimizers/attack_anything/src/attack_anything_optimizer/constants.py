"""Provenance + byte-identical asset pins for the vendored SEATS engine.

The upstream "Attack Anything" (SEATS) reference code was provided by the user as
an anonymous, under-review submission (README author "Anonymous", note "Under
review"). It carries no LICENSE file and no public repository or commit SHA; the
user asserts it is Apache-2.0, and it is redistributed here under that license
(see ``LICENSE`` + ``NOTICE``). Because there is no upstream commit to pin, the
byte-identical claim is anchored to a SHA-256 of each vendored file, captured at
vendoring time from the provided source. ``tests/test_assets_byte_identical.py``
recomputes and compares these, so any accidental edit to ``_vendor/`` fails loudly.
"""

from __future__ import annotations

#: Provenance of the vendored upstream, recorded for the ASSUMPTIONS ledger.
UPSTREAM_NAME = (
    "Attack Anything: Recursive Self-Evolving Attack Tree Search for Multi-Turn LLM Red-Teaming"
)
UPSTREAM_SOURCE = "user-provided reference implementation (anonymous, under review)"
UPSTREAM_REPO = ""  # no public repository available
UPSTREAM_COMMIT = ""  # no public commit SHA available (anonymous submission)
UPSTREAM_LICENSE = "Apache-2.0"

#: SHA-256 of every byte-identical vendored file, keyed by basename under
#: ``_vendor/``. Pinned at vendoring time from the provided source.
VENDORED_SHA256: dict[str, str] = {
    "tree.py": "233c9078fdd125ffba0ad6d54fd40ca6af76afccee763d4e97af6cd2f1ece0d6",
    "archive.py": "2c72ffde2990e40c2f98b0f0b6b6dc2b98431c674f7acc31df075adbb16cc8af",
    "operators.py": "50c93b2ed9c06c24abd83d113d8d0146b816238d9fd147a5d1285538fc266e20",
    "utils.py": "056ae6f93c005b86f0e5bebeac24a4a7e6128a4bb61153021edc3a59ed064982",
    "seats.py": "52606158397ae92d85fe24b1d812837dbaee8fbee85212bebda747107870cbc7",
    "seats_decon.py": "4e52311dd3bf0f65e6d85309e479e4922d6e8d1c8046c46f93ca503e6a3e24ba",
    "seats_feedback.py": "e92987af71566c8b400b06a456ecb1e0ca676c5a0bfd3a01e99b47b50cb3f197",
    "seats_feedback_decon.py": "e3d1d75f91ec2b1e6352e017ae106f85704447bad9ee4dc712606b899800fc5c",
    "seats_feedback_decon_separate.py": "0c3282b01c22015573982d2da3577fd358a6435f16ccfbdcb45df160b537ec04",
}
