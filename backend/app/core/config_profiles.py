import re
from pathlib import Path

from .config import BACKEND_ROOT, get_settings

_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_profile_name(profile: str | None) -> str | None:
    if profile is None:
        value = get_settings().config_profile.strip()
    else:
        value = profile.strip()
    if not value:
        return None
    if not _PROFILE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid config profile: {value}")
    return value


def resolve_config_path(filename: str, *, profile: str | None = None) -> Path:
    config_root = BACKEND_ROOT / "config"
    effective_profile = _normalize_profile_name(profile)
    if effective_profile is not None:
        candidate = config_root / "profiles" / effective_profile / filename
        if candidate.exists():
            return candidate
    return config_root / filename
