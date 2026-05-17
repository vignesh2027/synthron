"""Standalone dashboard server — serves the React static files and proxies API events."""

from __future__ import annotations

import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


STATIC_DIR = Path(__file__).parent / "static"


def create_dashboard_app(api_url: str = "http://localhost:8080") -> FastAPI:
    app = FastAPI(title="Synthron Dashboard", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/config.js")
    async def config_js():
        """Serve runtime config so the React app knows where the API lives."""
        return HTMLResponse(
            content=f"window.SYNTHRON_API_URL = '{api_url}';",
            media_type="application/javascript",
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "synthron-dashboard"}

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


def run_dashboard(host: str = "0.0.0.0", port: int = 3000, api_url: str = "http://localhost:8080"):
    """Launch the dashboard server directly (used by CLI)."""
    app = create_dashboard_app(api_url=api_url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


async def run_dashboard_async(host: str = "0.0.0.0", port: int = 3000, api_url: str = "http://localhost:8080"):
    """Launch the dashboard server in async context."""
    app = create_dashboard_app(api_url=api_url)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
