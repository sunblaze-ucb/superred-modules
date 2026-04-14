"""Security domain tags and defaults for the OpenClaw target module."""

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

SYSTEM_TAG = SecurityDomainTag("system")
USER_INPUT_TAG = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
EXTERNAL_DATA_TAG = SecurityDomainTag("external_data", parent=SYSTEM_TAG)
INTERNAL_CONTEXT_TAG = SecurityDomainTag("internal_context", parent=SYSTEM_TAG)
TOOL_CATALOG_TAG = SecurityDomainTag("tool_catalog", parent=SYSTEM_TAG)
MODEL_TAG = SecurityDomainTag("model", parent=SYSTEM_TAG)

OPENCLAW_DOMAIN = SecurityDomain([
    SYSTEM_TAG,
    USER_INPUT_TAG,
    EXTERNAL_DATA_TAG,
    INTERNAL_CONTEXT_TAG,
    TOOL_CATALOG_TAG,
    MODEL_TAG,
])

DEFAULT_GATEWAY_URL = "ws://127.0.0.1:18789"
DEFAULT_AGENT_TIMEOUT_S = 120
