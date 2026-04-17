from pathlib import Path

from backend.app.config import DEFAULT_DOMAIN, ENABLED_LORA_DOMAINS, LORA_REGISTRY


def normalize_domain(domain: str | None) -> str:
    if domain is None:
        return DEFAULT_DOMAIN
    value = domain.strip().lower()
    return value or DEFAULT_DOMAIN


def resolve_domain(domain: str | None) -> str:
    normalized = normalize_domain(domain)

    if normalized == DEFAULT_DOMAIN:
        return DEFAULT_DOMAIN

    if normalized not in ENABLED_LORA_DOMAINS:
        return DEFAULT_DOMAIN

    adapter_path = LORA_REGISTRY.get(normalized)
    if adapter_path is None:
        return DEFAULT_DOMAIN

    if not Path(adapter_path).exists():
        return DEFAULT_DOMAIN

    return normalized


def get_adapter_path(domain: str | None) -> Path | None:
    resolved = resolve_domain(domain)
    if resolved == DEFAULT_DOMAIN:
        return None
    return Path(LORA_REGISTRY[resolved])