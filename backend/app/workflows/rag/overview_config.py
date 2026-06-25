import json
from dataclasses import dataclass
from functools import lru_cache

from ...core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class OverviewQueryConfig:
    overview_markers: tuple[str, ...]
    topic_markers: tuple[str, ...]


def _read_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in list(payload.get(key) or []) if str(item).strip())


@lru_cache
def load_overview_query_config() -> OverviewQueryConfig:
    payload = json.loads(resolve_config_path("overview_rules.json").read_text(encoding="utf-8"))
    return OverviewQueryConfig(
        overview_markers=_read_list(payload, "overview_markers"),
        topic_markers=_read_list(payload, "topic_markers"),
    )
