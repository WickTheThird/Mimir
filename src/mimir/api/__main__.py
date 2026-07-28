"""``python -m mimir.api`` / ``mimir-api``."""

from __future__ import annotations

import uvicorn

from mimir.config import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "mimir.api.app:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.observability.log_level.lower(),
    )


if __name__ == "__main__":
    run()
