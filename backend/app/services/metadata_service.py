import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from backend.app.config import METADATA_DIR


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pipeline_metadata(payload: dict, base_name: str | None = None) -> dict:
    ensure_directory(METADATA_DIR)

    if not base_name:
        base_name = uuid4().hex

    metadata_filename = f"{base_name}.json"
    metadata_path = METADATA_DIR / metadata_filename

    record = {
        "saved_at": datetime.now().isoformat(),
        "payload": payload,
    }

    metadata_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "metadata_filename": metadata_filename,
        "metadata_path": str(metadata_path.resolve()),
    }