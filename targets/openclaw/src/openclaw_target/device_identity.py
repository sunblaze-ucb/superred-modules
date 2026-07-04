"""Ed25519 device identity for OpenClaw gateway WebSocket connect.

Remote connections (e.g. host → Docker-published gateway port) are not treated
as the direct-local ``gateway-client`` backend path, so the gateway clears
device-less scope requests. We generate a per-instance identity, pre-seed pairing
in the gateway state dir, and sign the connect challenge (protocol v3).
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

OPERATOR_SCOPES = ("operator.read", "operator.write", "operator.admin")

# NOTE: this must NOT be "identity/device.json" — that path is OpenClaw's own
# reserved *gateway self-identity* (loadOrCreateDeviceIdentity() in
# src/infra/device-identity.ts, used by heartbeat/APNs-relay/node-host), which
# the gateway unconditionally (re)writes on startup. Writing our *operator
# client* identity there gets silently clobbered the moment the container
# boots, and the pairing record we seeded no longer matches the connecting
# client's key. Use a namespaced path OpenClaw never looks at.
_DEVICE_JSON = "superred-operator/identity.json"
_PAIRED_JSON = "devices/paired.json"


@dataclass(frozen=True)
class DeviceIdentity:
    device_id: str
    public_key_pem: str
    private_key_pem: str


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)


def public_key_raw_base64url(public_key_pem: str) -> str:
    """Raw 32-byte Ed25519 public key as base64url (wire format)."""
    raw = _extract_raw_public_key(public_key_pem)
    return _b64url(raw)


def _extract_raw_public_key(public_key_pem: str) -> bytes:
    pub = serialization.load_pem_public_key(public_key_pem.encode())
    if not isinstance(pub, Ed25519PublicKey):
        raise TypeError("expected Ed25519 public key")
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def device_id_from_public_key_pem(public_key_pem: str) -> str:
    raw = _extract_raw_public_key(public_key_pem)
    return hashlib.sha256(raw).hexdigest()


def build_device_auth_payload_v3(
    *,
    device_id: str,
    client_id: str,
    client_mode: str,
    role: str,
    scopes: list[str],
    signed_at_ms: int,
    token: str | None,
    nonce: str,
    platform: str = "",
    device_family: str = "",
) -> str:
    """Build the v3 device signature payload (must match OpenClaw exactly)."""
    scope_str = ",".join(scopes)
    token_str = token or ""
    platform_norm = platform.strip().lower() if platform else ""
    family_norm = device_family.strip().lower() if device_family else ""
    return "|".join(
        [
            "v3",
            device_id,
            client_id,
            client_mode,
            role,
            scope_str,
            str(signed_at_ms),
            token_str,
            nonce,
            platform_norm,
            family_norm,
        ],
    )


def sign_device_payload(private_key_pem: str, payload: str) -> str:
    """Sign UTF-8 payload; return base64url signature bytes."""
    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("expected Ed25519 private key")
    sig = key.sign(payload.encode())
    return _b64url(sig)


def load_or_create_device_identity(path: Path) -> DeviceIdentity:
    """Load ``identity/device.json`` or create a fresh Ed25519 identity."""
    if path.exists():
        data = json.loads(path.read_text())
        if (
            data.get("version") == 1
            and isinstance(data.get("publicKeyPem"), str)
            and isinstance(data.get("privateKeyPem"), str)
        ):
            pub = data["publicKeyPem"]
            priv = data["privateKeyPem"]
            device_id = device_id_from_public_key_pem(pub)
            return DeviceIdentity(device_id=device_id, public_key_pem=pub, private_key_pem=priv)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    device_id = device_id_from_public_key_pem(public_pem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "deviceId": device_id,
                "publicKeyPem": public_pem,
                "privateKeyPem": private_pem,
                "createdAtMs": int(time.time() * 1000),
            },
            indent=2,
        ),
    )
    return DeviceIdentity(
        device_id=device_id,
        public_key_pem=public_pem,
        private_key_pem=private_pem,
    )


def seed_operator_device_pairing(
    state_dir: Path,
    identity: DeviceIdentity,
    *,
    client_id: str = "cli",
    client_mode: str = "cli",
    platform: str = "linux",
    scopes: tuple[str, ...] = OPERATOR_SCOPES,
) -> None:
    """Pre-approve the superred operator device in ``devices/paired.json``.

    The gateway derives a paired device's *effective* roles from its active
    ``tokens`` map, not the ``role``/``roles`` fields
    (``listEffectivePairedDeviceRoles`` in ``src/infra/device-pairing.ts`` — a
    tokenless record "fails closed" and returns no roles, which the connect
    handshake treats as a ``role-upgrade`` and rejects). So we must seed an
    active operator token entry, keyed by role, carrying the approved scopes.
    """
    paired_path = state_dir / _PAIRED_JSON
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    public_raw = public_key_raw_base64url(identity.public_key_pem)
    scope_list = list(scopes)
    token = _b64url(secrets.token_bytes(32))
    paired_path.write_text(
        json.dumps(
            {
                identity.device_id: {
                    "deviceId": identity.device_id,
                    "publicKey": public_raw,
                    "role": "operator",
                    "roles": ["operator"],
                    "scopes": scope_list,
                    "approvedScopes": scope_list,
                    "clientId": client_id,
                    "clientMode": client_mode,
                    "platform": platform,
                    "tokens": {
                        "operator": {
                            "token": token,
                            "role": "operator",
                            "scopes": scope_list,
                            "createdAtMs": now,
                        },
                    },
                    "createdAtMs": now,
                    "approvedAtMs": now,
                },
            },
            indent=2,
        ),
    )


def ensure_device_auth_for_state_dir(state_dir: Path) -> Path:
    """Create identity + pre-seeded pairing; return identity file path."""
    identity_path = state_dir / _DEVICE_JSON
    identity = load_or_create_device_identity(identity_path)
    seed_operator_device_pairing(state_dir, identity)
    return identity_path
