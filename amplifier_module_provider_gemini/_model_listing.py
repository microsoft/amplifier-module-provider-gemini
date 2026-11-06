"""Provider-local selectors over authentic records, never inference policy.

Common include/exclude/hide_dated semantics follow OpenAI PR141 at 38b7b26.
Gemini adds explicit chat_only/latest_per_tier policy; {} disables both.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any


# Only understood ordinary chat names participate in version recommendations.
# Unknown families/suffixes remain accessible through explicit/unfiltered lists.
_FAMILY = re.compile(
    r"^(?P<scope>publishers/google/models/)?gemini-"
    r"(?P<major>\d+)(?:\.(?P<minor>\d+))?-"
    r"(?P<tier>flash-lite|flash|pro)"
    r"(?P<preview>-(?:preview|exp))?"
    r"(?P<stamp>-(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{4}|\d{2}-\d{2}|\d{3}))?$",
    re.IGNORECASE,
)
_MEDIA = re.compile(r"(?:^|[-/])(?:image|video|audio|tts|embedding|imagen|veo)(?:[-/]|$)", re.I)


def _family(model_id: str):
    return _FAMILY.fullmatch(model_id)


def _chat(record: dict[str, Any]) -> bool:
    if _MEDIA.search(record["id"]):
        return False
    methods = record.get("generation_methods")
    if methods is not None:
        return bool({"generateContent", "streamGenerateContent"}.intersection(methods))
    # Vertex publisher metadata may omit generation methods. A recognized
    # ordinary chat name is eligible, but does NOT manufacture capabilities.
    return _family(record["id"]) is not None


def _stamp_key(stamp: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in stamp.lstrip("-").split("-")) if stamp else ()
    if len(parts) == 2 and parts[1] >= 1000:  # MM-YYYY
        return parts[1], parts[0]
    return parts


def _latest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[dict, Any]]] = {}
    for record in records:
        match = _family(record["id"])
        if match:
            key = (match["scope"] or "").casefold(), match["tier"].casefold()
            groups.setdefault(key, []).append((record, match))

    selected = []
    for group in groups.values():
        def version(pair):
            match = pair[1]
            return int(match["major"]), int(match["minor"] or 0)

        def winner(items):
            newest = max(map(version, items))
            items = [pair for pair in items if version(pair) == newest]
            # Prefer an alias only within the SAME release channel/version.
            aliases = [pair for pair in items if not pair[1]["stamp"]]
            return max(aliases or items, key=lambda pair: (
                _stamp_key(pair[1]["stamp"] or ""), pair[0]["id"],
            ))

        stable = [pair for pair in group if not pair[1]["preview"]]
        preview = [pair for pair in group if pair[1]["preview"]]
        current = winner(stable) if stable else None
        if current:
            selected.append(current[0])
        if preview:
            newer = winner(preview)
            # Same-version previews are not evidence of a newer release.
            if current is None or version(newer) > version(current):
                selected.append(newer[0])
    return selected


@dataclass(frozen=True)
class ModelFilters:
    """Immutable policy snapshot, including every recommendation rule."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    hide_dated: bool = False
    chat_only: bool = False
    latest_per_tier: bool = False

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "ModelFilters":
        if not isinstance(value, Mapping):
            raise TypeError("model filters must be a mapping")
        if set(value) - {"include", "exclude", "hide_dated", "chat_only", "latest_per_tier"}:
            raise ValueError("unknown model filter key")

        def patterns(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(key, default)
            if not isinstance(raw, (list, tuple)) or any(
                not isinstance(pattern, str) or not pattern.strip() for pattern in raw
            ):
                raise ValueError(f"model filters {key} must be a list of nonblank strings")
            return tuple(raw)

        flags = []
        for key in ("hide_dated", "chat_only", "latest_per_tier"):
            flag = value.get(key, False)
            if not isinstance(flag, bool):
                raise TypeError(f"model filters {key} must be a boolean")
            flags.append(flag)
        return cls(patterns("include", ("*",)), patterns("exclude", ()), *flags)

    def select(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = {
            record["id"]: record for record in records
            if any(fnmatchcase(record["id"].casefold(), p.casefold()) for p in self.include)
            and not any(fnmatchcase(record["id"].casefold(), p.casefold()) for p in self.exclude)
            and (not self.chat_only or _chat(record))
        }
        if self.hide_dated:
            aliases = {model_id.casefold() for model_id in selected}
            selected = {
                model_id: record for model_id, record in selected.items()
                if not (
                    (match := re.search(r"-\d{4}-\d{2}-\d{2}$", model_id))
                    and model_id[:match.start()].casefold() in aliases
                )
            }
        records = list(selected.values())
        if self.latest_per_tier:
            records = _latest(records)
        return sorted(records, key=lambda record: record["id"])

    def as_dict(self) -> dict:
        return {
            "include": list(self.include), "exclude": list(self.exclude),
            "hide_dated": self.hide_dated, "chat_only": self.chat_only,
            "latest_per_tier": self.latest_per_tier,
        }


DEFAULT_FILTERS = ModelFilters(hide_dated=True, chat_only=True, latest_per_tier=True)


def selector_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "include": {
                "type": "array", "items": {"type": "string", "minLength": 1},
                "default": ["*"],
                "description": "Case-insensitive, OS-independent fnmatch patterns; [] includes none.",
            },
            "exclude": {
                "type": "array", "items": {"type": "string", "minLength": 1},
                "default": [], "description": "Case-insensitive fnmatch; exclusion wins.",
            },
            "hide_dated": {
                "type": "boolean", "default": False,
                "description": "Hide YYYY-MM-DD suffixes only when the selected undated alias exists.",
            },
            "chat_only": {
                "type": "boolean", "default": False,
                "description": "Require chat generation methods, or a recognized ordinary chat name when methods are absent; omit media-only names.",
            },
            "latest_per_tier": {
                "type": "boolean", "default": False,
                "description": "Latest numeric Gemini Pro/Flash/Flash-Lite per resource scope, plus a strictly newer preview; alias before matching snapshot. Unknown names omitted.",
            },
        },
    }
