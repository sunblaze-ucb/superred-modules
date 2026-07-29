"""Byte-preserving access to the pinned L1B3RT4S prompt corpus.

The upstream repository is a collection of text files, not a program or a
machine-readable attack catalogue.  This module therefore keeps two layers
separate:

* the bundled upstream files are immutable byte snapshots, verified by SHA-256;
* ``PromptTemplate`` objects are zero-normalization slices into those files.

Only replacing an explicit upstream goal marker changes a runnable prompt.
No Unicode normalization, whitespace stripping, newline conversion, or
re-encoding is performed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Literal, TypedDict, cast

UPSTREAM_REPOSITORY = "https://github.com/elder-plinius/L1B3RT4S"
UPSTREAM_COMMIT = "64960b783249d36f76a48a33103cc4b168332b9b"

Delivery = Literal["user", "system"]


class ManifestFile(TypedDict):
    source_path: str
    stored_path: str | None
    sha256: str
    size: int
    bundled: bool
    reason: str | None


class CorpusManifest(TypedDict):
    repository: str
    commit: str
    files: list[ManifestFile]


# Exact generic slots present in the pinned upstream corpus.  Longer markers
# come first so an overlapping future spelling cannot be partially replaced.
GOAL_MARKERS: tuple[str, ...] = tuple(
    sorted(
        {
            "{Z}",
            "{USER QUERY}",
            "{USER-QUERY}",
            "{insert-your-query-here}",
            "{the user_input/query}",
            "{usEr_QuEry}",
            "{user-input}",
            "{user-query}",
            "{user_input/query}",
            "{user_input}",
            "{𝔲𝔰𝔢𝔯_𝓺𝓾𝓮𝓻𝔂}",
            "<VARIABLE Z = ________>",
            "<user_query>",
        },
        key=len,
        reverse=True,
    )
)

# Upstream file -> provider family.  Files absent from this table are auxiliary
# corpora and are never mistaken for runnable model-specific prompts.
SOURCE_PROVIDERS: dict[str, str] = {
    "ALIBABA.mkd": "alibaba",
    "AMAZON.mkd": "amazon",
    "ANTHROPIC.mkd": "anthropic",
    "APPLE.mkd": "apple",
    "BRAVE.mkd": "brave",
    "CHATGPT.mkd": "openai",
    "COHERE.mkd": "cohere",
    "CURSOR.mkd": "cursor",
    "DEEPSEEK.mkd": "deepseek",
    "FETCHAI.mkd": "fetchai",
    "GOOGLE.mkd": "google",
    "GRAYSWAN.mkd": "grayswan",
    "GROK-MEGA.mkd": "xai",
    "HUME.mkd": "hume",
    "INCEPTION.mkd": "inception",
    "INFLECTION.mkd": "inflection",
    "LIQUIDAI.mkd": "liquidai",
    "META.mkd": "meta",
    "MICROSOFT.mkd": "microsoft",
    "MIDJOURNEY.mkd": "midjourney",
    "MISTRAL.mkd": "mistral",
    "MOONSHOT.mkd": "moonshot",
    "MULTION.mkd": "multion",
    "NOUS.mkd": "nous",
    "NVIDIA.mkd": "nvidia",
    "OPENAI.mkd": "openai",
    "PERPLEXITY.mkd": "perplexity",
    "REKA.mkd": "reka",
    "WINDSURF.mkd": "windsurf",
    "XAI.mkd": "xai",
    "ZAI.mkd": "zai",
    "ZYPHRA.mkd": "zyphra",
}

# These short upstream files are not vendor-specific.  With the default strict
# mode they are included only if they contain an explicit generic goal marker.
UNIVERSAL_SOURCES: tuple[str, ...] = (
    "#MOTHERLOAD.txt",
    "-MISCELLANEOUS-.mkd",
    "1337.mkd",
    "AAA.mkd",
    "README.md",
    "REFLECTION.mkd",
)

# Heading text is metadata, not part of the prompt pasted by an upstream user.
# These phrases identify entries upstream explicitly says belong in a
# privileged/custom-instruction surface.
_SYSTEM_DELIVERY_HINTS = (
    "custom gpt",
    "custom instruction",
    "instructions field",
    "set as system prompt",
    "sys prompt",
    "system prompt generator",
)

_SOURCE_HEADING_LEVELS: dict[str, tuple[int, ...]] = {
    "AMAZON.mkd": (1, 2),
    "GOOGLE.mkd": (1, 2),
    "GRAYSWAN.mkd": (2,),
    "META.mkd": (1, 2),
    "NVIDIA.mkd": (2,),
    "PERPLEXITY.mkd": (2,),
}


def _heading_pattern(source_file: str) -> re.Pattern[str]:
    levels = _SOURCE_HEADING_LEVELS.get(source_file, (1,))
    prefixes = "|".join(re.escape("#" * level) for level in sorted(levels, reverse=True))
    return re.compile(rf"(?m)^(?:{prefixes}) ([^\r\n]*)(?:\r\n|\n|\r|$)")


@dataclass(frozen=True)
class PromptTemplate:
    """One exact prompt-body slice from a pinned upstream source file."""

    id: str
    source_file: str
    section_index: int
    heading: str | None
    provider: str | None
    delivery: Delivery
    raw_template: str
    raw_sha256: str
    goal_markers: tuple[str, ...]

    @property
    def is_templated(self) -> bool:
        """Whether upstream supplied a recognized generic goal slot."""

        return bool(self.goal_markers)


def _data_root() -> Traversable:
    return files("libertas_optimizer").joinpath("data")


@cache
def load_manifest() -> CorpusManifest:
    """Load metadata for every file at the pinned upstream commit."""

    raw = _data_root().joinpath("upstream_manifest.json").read_text(encoding="utf-8")
    return cast(CorpusManifest, json.loads(raw))


def _manifest_entry(source_file: str) -> ManifestFile:
    for entry in load_manifest()["files"]:
        if entry["source_path"] == source_file:
            return entry
    raise KeyError(f"{source_file!r} is not in the pinned L1B3RT4S manifest")


def load_source_bytes(source_file: str) -> bytes:
    """Return freshly verified bundled bytes for an upstream root file."""

    entry = _manifest_entry(source_file)
    if not entry["bundled"] or entry["stored_path"] is None:
        reason = entry["reason"] or "not bundled"
        raise ValueError(f"{source_file!r} is recorded but unavailable: {reason}")
    content = _data_root().joinpath("upstream", entry["stored_path"]).read_bytes()
    if len(content) != entry["size"]:
        raise ValueError(
            f"{source_file}: size {len(content)} does not match manifest {entry['size']}"
        )
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != entry["sha256"]:
        raise ValueError(
            f"{source_file}: sha256 {actual_hash} does not match manifest {entry['sha256']}"
        )
    return content


def verify_bundled_corpus() -> list[str]:
    """Return parity errors for the bundled snapshot; an empty list means exact."""

    manifest = load_manifest()
    errors: list[str] = []
    if manifest["repository"] != UPSTREAM_REPOSITORY:
        errors.append(f"repository mismatch: {manifest['repository']!r} != {UPSTREAM_REPOSITORY!r}")
    if manifest["commit"] != UPSTREAM_COMMIT:
        errors.append(f"commit mismatch: {manifest['commit']!r} != {UPSTREAM_COMMIT!r}")

    for entry in manifest["files"]:
        if not entry["bundled"]:
            continue
        source = entry["source_path"]
        try:
            content = load_source_bytes(source)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            errors.append(f"{source}: {exc}")
            continue
        actual_hash = hashlib.sha256(content).hexdigest()
        if len(content) != entry["size"]:
            errors.append(f"{source}: size {len(content)} != {entry['size']}")
        if actual_hash != entry["sha256"]:
            errors.append(f"{source}: sha256 {actual_hash} != {entry['sha256']}")
    return errors


def _provider_source_files(provider: str | None) -> tuple[str, ...]:
    if provider is None:
        selected = set(SOURCE_PROVIDERS)
    else:
        normalized = provider.strip().lower()
        known = set(SOURCE_PROVIDERS.values())
        if normalized not in known:
            raise ValueError(
                f"unknown Libertas provider {provider!r}; expected one of {sorted(known)}"
            )
        selected = {
            source
            for source, source_provider in SOURCE_PROVIDERS.items()
            if source_provider == normalized
        }
    selected.update(UNIVERSAL_SOURCES)
    return tuple(sorted(selected))


def _delivery_for_heading(heading: str | None) -> Delivery:
    if heading is None:
        return "user"
    lowered = heading.casefold()
    if any(hint in lowered for hint in _SYSTEM_DELIVERY_HINTS) or (
        "instructions" in lowered and "field" in lowered
    ):
        return "system"
    return "user"


def _template_id(
    source_file: str,
    section_index: int,
    raw_sha256: str,
) -> str:
    safe_source = re.sub(r"[^a-z0-9]+", "-", source_file.casefold()).strip("-")
    return f"{safe_source}:{section_index:03d}:{raw_sha256[:16]}"


def _split_source(source_file: str) -> tuple[PromptTemplate, ...]:
    raw_bytes = load_source_bytes(source_file)
    # Strict decoding is intentional: replacement characters would destroy
    # upstream parity and can erase the very token sequence under test.
    text = raw_bytes.decode("utf-8", errors="strict")
    provider = SOURCE_PROVIDERS.get(source_file)
    matches = list(_heading_pattern(source_file).finditer(text))
    sections: list[tuple[int, str | None, str]] = []

    if not matches:
        if text:
            sections.append((0, None, text))
    else:
        preamble = text[: matches[0].start()]
        section_index = 0
        if preamble:
            sections.append((section_index, None, preamble))
            section_index += 1
        for match_index, match in enumerate(matches):
            end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(text)
            body = text[match.end() : end]
            if not body:
                continue
            sections.append((section_index, match.group(1), body))
            section_index += 1

    templates: list[PromptTemplate] = []
    for section_index, heading, body in sections:
        body_bytes = body.encode("utf-8")
        digest = hashlib.sha256(body_bytes).hexdigest()
        markers = tuple(
            marker
            for marker in GOAL_MARKERS
            if marker in body and not (marker == "<user_query>" and "</user_query>" in body)
        )
        templates.append(
            PromptTemplate(
                id=_template_id(source_file, section_index, digest),
                source_file=source_file,
                section_index=section_index,
                heading=heading,
                provider=provider,
                delivery=_delivery_for_heading(heading),
                raw_template=body,
                raw_sha256=digest,
                goal_markers=markers,
            )
        )
    return tuple(templates)


def _templates_for_source(source_file: str) -> tuple[PromptTemplate, ...]:
    return _split_source(source_file)


def load_prompt_templates(
    *,
    provider: str | None = None,
    include_untemplated: bool = False,
    include_system_templates: bool = False,
    source_files: tuple[str, ...] | None = None,
) -> tuple[PromptTemplate, ...]:
    """Load runnable prompt slices without changing their upstream text.

    Args:
        provider: Canonical provider family. ``None`` selects every provider,
            which is the fallback when no model identity is visible.
        include_untemplated: Include sections without an explicit upstream goal
            marker. Rendering those sections appends the goal and is therefore
            an adaptation, disabled by default.
        include_system_templates: Include entries upstream labels for system
            prompts or custom-instruction fields. Disabled by default because
            normal user-only scopes cannot reproduce that delivery surface.
        source_files: Optional exact upstream filenames, primarily for
            reproducibility studies. When set, this replaces provider routing.
    """

    selected_files = (
        tuple(source_files) if source_files is not None else _provider_source_files(provider)
    )
    templates: list[PromptTemplate] = []
    for source_file in selected_files:
        for template in _templates_for_source(source_file):
            if not include_untemplated and not template.is_templated:
                continue
            if not include_system_templates and template.delivery == "system":
                continue
            templates.append(template)
    return tuple(templates)


def render_prompt(
    template: PromptTemplate,
    goal: str,
    *,
    append_untemplated: bool = False,
) -> str:
    """Insert a goal while preserving every other upstream code point."""

    if template.goal_markers:
        marker_pattern = re.compile("|".join(re.escape(marker) for marker in template.goal_markers))
        return marker_pattern.sub(lambda _match: goal, template.raw_template)

    if not append_untemplated:
        raise ValueError(
            f"{template.id} has no explicit upstream goal marker; "
            "set append_untemplated=True to opt into adaptation"
        )

    separator = "" if template.raw_template.endswith(("\n", "\r")) else "\n"
    return f"{template.raw_template}{separator}{goal}"


_PROVIDER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("anthropic/", "claude")),
    ("openai", ("openai/", "azure/", "chatgpt", "gpt-", "o1", "o3", "o4")),
    ("google", ("google/", "vertex_ai/", "gemini", "gemma")),
    ("xai", ("xai/", "grok")),
    ("meta", ("meta/", "llama")),
    ("mistral", ("mistral/", "mixtral", "ministral")),
    ("deepseek", ("deepseek/", "deepseek")),
    ("alibaba", ("alibaba/", "dashscope/", "qwen")),
    ("amazon", ("bedrock/amazon", "amazon/", "nova")),
    ("zai", ("zai/", "glm-")),
    ("moonshot", ("moonshot/", "kimi")),
    ("nous", ("nous/", "hermes")),
    ("nvidia", ("nvidia/", "nemotron")),
    ("cohere", ("cohere/", "command-r")),
    ("perplexity", ("perplexity/", "sonar")),
)


def detect_provider(model_identity: str) -> str | None:
    """Infer the upstream vendor file family from a target model identifier."""

    lowered = model_identity.strip().casefold()
    if not lowered:
        return None
    for provider, patterns in _PROVIDER_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return provider
    return None
