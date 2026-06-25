import json
from dataclasses import dataclass
from functools import lru_cache

from ..core.config_profiles import resolve_config_path


@dataclass(frozen=True)
class KnowledgeBaseDomainOption:
    value: str
    label: str
    description: str


@dataclass(frozen=True)
class KnowledgeBaseDomainConfig:
    default_domain: str
    options: tuple[KnowledgeBaseDomainOption, ...]


def _read_option(payload: object) -> KnowledgeBaseDomainOption | None:
    if not isinstance(payload, dict):
        return None
    value = str(payload.get("value") or "").strip()
    label = str(payload.get("label") or "").strip()
    if not value or not label:
        return None
    return KnowledgeBaseDomainOption(
        value=value,
        label=label,
        description=str(payload.get("description") or "").strip(),
    )


@lru_cache
def load_knowledge_base_domain_config() -> KnowledgeBaseDomainConfig:
    payload = json.loads(resolve_config_path("knowledge_base_domains.json").read_text(encoding="utf-8"))
    options = tuple(
        option
        for option in (_read_option(item) for item in list(payload.get("options") or []))
        if option is not None
    )
    default_domain = str(payload.get("default_domain") or "").strip()
    if options and not default_domain:
        default_domain = options[0].value
    configured_values = {item.value for item in options}
    if default_domain and configured_values and default_domain not in configured_values:
        default_domain = next(iter(configured_values))
    return KnowledgeBaseDomainConfig(
        default_domain=default_domain or "general",
        options=options,
    )
