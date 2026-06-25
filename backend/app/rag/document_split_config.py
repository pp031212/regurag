import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class DocumentSplitStrategy:
    name: str
    detection_patterns: tuple[str, ...]
    min_detection_matches: int
    boundary_pattern: str


@dataclass(frozen=True)
class DocumentSentenceSplitProfile:
    name: str
    source_formats: tuple[str, ...]
    boundary_pattern: str
    fallback_boundary_patterns: tuple[str, ...]


@dataclass(frozen=True)
class DocumentSplitConfig:
    default_strategy: str
    strategies: tuple[DocumentSplitStrategy, ...]
    default_sentence_profile: str
    sentence_profiles: tuple[DocumentSentenceSplitProfile, ...]


def _config_path(profile: str | None = None):
    return resolve_config_path("document_split_rules.json", profile=profile)


@lru_cache
def load_document_split_config(profile: str | None = None) -> DocumentSplitConfig:
    payload = json.loads(_config_path(profile=profile).read_text(encoding="utf-8"))
    strategies = tuple(
        DocumentSplitStrategy(
            name=str(item["name"]),
            detection_patterns=tuple(str(pattern) for pattern in list(item.get("detection_patterns") or [])),
            min_detection_matches=int(item.get("min_detection_matches", 0) or 0),
            boundary_pattern=str(item["boundary_pattern"]),
        )
        for item in list(payload.get("strategies") or [])
    )
    sentence_profiles = tuple(
        DocumentSentenceSplitProfile(
            name=str(item["name"]),
            source_formats=tuple(str(part) for part in list(item.get("source_formats") or []) if str(part).strip()),
            boundary_pattern=str(item["boundary_pattern"]),
            fallback_boundary_patterns=tuple(
                str(pattern) for pattern in list(item.get("fallback_boundary_patterns") or []) if str(pattern).strip()
            ),
        )
        for item in list(payload.get("sentence_profiles") or [])
    )
    return DocumentSplitConfig(
        default_strategy=str(payload["default_strategy"]),
        strategies=strategies,
        default_sentence_profile=str(payload.get("default_sentence_profile") or "generic_sentence"),
        sentence_profiles=sentence_profiles,
    )
