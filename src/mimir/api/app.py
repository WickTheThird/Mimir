"""FastAPI application (ADR 6.2 C1, 15, 16)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mimir import __version__
from mimir.api.deps import shutdown_runner
from mimir.api.openai_facade import router as facade_router
from mimir.api.routes.investigations import router as investigations_router
from mimir.api.routes.system import router as system_router
from mimir.config import Settings, get_settings
from mimir.logging import configure_logging, correlation_context, get_logger
from mimir.observability.metrics import METRICS

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(
        level=settings.observability.log_level,
        json_logs=settings.observability.json_logs,
        log_file=settings.observability.log_file,
        force=True,
    )
    from mimir.tools.base import load_all_tools

    registry = load_all_tools()
    log.info(
        "mimir_api_starting",
        version=__version__,
        host=settings.api.host,
        port=settings.api.port,
        tools=len(registry.all()),
        auth_required=not settings.api.allow_loopback_without_auth,
        keys_configured=len(settings.api.api_keys),
    )
    if settings.api.host not in ("127.0.0.1", "localhost", "::1") and not settings.api.api_keys:
        log.warning(
            "binding_non_loopback_without_keys",
            host=settings.api.host,
            hint="run 'mimir keys create' before exposing this service",
        )
    try:
        from mimir.persistence.db import get_database

        get_database(settings).ensure_schema()
    except Exception as exc:  # noqa: BLE001 - the API is still useful without history
        log.warning("persistence_unavailable", error=str(exc))

    yield

    await shutdown_runner()
    log.info("mimir_api_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    app = FastAPI(
        title="MIMIR",
        version=__version__,
        description=(
            "Local operations investigation platform. The /v1 routes are an "
            "OpenAI-compatible inference facade; /api routes are loopback-only."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        with correlation_context() as correlation_id:
            try:
                response = await call_next(request)
            except Exception:
                METRICS.increment("http.errors")
                log.exception("request_failed", path=request.url.path)
                raise
            elapsed = time.perf_counter() - started
            METRICS.observe("http", elapsed)
            METRICS.increment(f"http.{response.status_code // 100}xx")
            response.headers["X-Correlation-Id"] = correlation_id
            response.headers["X-Response-Time"] = f"{elapsed:.3f}"
            return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak a traceback to a caller that might be remote.
        log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": "internal error; see the MIMIR log for detail",
                }
            },
        )

    app.include_router(facade_router, prefix="/v1")
    app.include_router(system_router, prefix="/api")
    app.include_router(investigations_router, prefix="/api")

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "mimir",
            "version": __version__,
            "surfaces": {
                "/v1": "OpenAI-compatible inference facade (authenticated)",
                "/api": "local control plane (loopback only)",
            },
            "docs": "/docs",
        }

    return app


app = create_app
