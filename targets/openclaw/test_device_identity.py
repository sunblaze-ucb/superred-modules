"""Unit tests for Ed25519 device identity + pairing seed."""

from __future__ import annotations

import json
from pathlib import Path

from openclaw_target.device_identity import (
    OPERATOR_SCOPES,
    build_device_auth_payload_v3,
    device_id_from_public_key_pem,
    ensure_device_auth_for_state_dir,
    load_or_create_device_identity,
    public_key_raw_base64url,
    seed_operator_device_pairing,
    sign_device_payload,
)


def test_device_auth_payload_v3_format() -> None:
    payload = build_device_auth_payload_v3(
        device_id="abc123",
        client_id="cli",
        client_mode="cli",
        role="operator",
        scopes=list(OPERATOR_SCOPES),
        signed_at_ms=1700000000000,
        token="gw-token",
        nonce="nonce-1",
        platform="linux",
    )
    assert payload == (
        "v3|abc123|cli|cli|operator|"
        "operator.read,operator.write,operator.admin|"
        "1700000000000|gw-token|nonce-1|linux|"
    )


def test_load_or_create_device_identity_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "identity" / "device.json"
    first = load_or_create_device_identity(path)
    second = load_or_create_device_identity(path)
    assert first.device_id == second.device_id
    assert device_id_from_public_key_pem(first.public_key_pem) == first.device_id
    raw = public_key_raw_base64url(first.public_key_pem)
    assert len(raw) > 0

    sig = sign_device_payload(first.private_key_pem, "test-payload")
    assert isinstance(sig, str) and len(sig) > 0


def test_seed_operator_device_pairing(tmp_path: Path) -> None:
    identity_path = ensure_device_auth_for_state_dir(tmp_path)
    identity = load_or_create_device_identity(identity_path)
    paired = json.loads((tmp_path / "devices" / "paired.json").read_text())
    record = paired[identity.device_id]
    assert record["approvedScopes"] == list(OPERATOR_SCOPES)
    assert record["publicKey"] == public_key_raw_base64url(identity.public_key_pem)

    # Idempotent re-seed keeps the same device id.
    seed_operator_device_pairing(tmp_path, identity)
    paired_again = json.loads((tmp_path / "devices" / "paired.json").read_text())
    assert set(paired_again) == {identity.device_id}
