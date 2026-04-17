from backend.app.services.lora_registry import resolve_domain


def choose_domain(requested_domain: str | None) -> str:
    return resolve_domain(requested_domain)