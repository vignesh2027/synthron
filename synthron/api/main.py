"""Synthron FastAPI application — REST API + WebSocket server."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from synthron.api.models import HealthResponse
from synthron.api.routes import agents, memory, stream, tasks
from synthron.providers.smart_router import router as smart_router
from synthron.utils.config import settings
from synthron.utils.logger import get_logger, print_banner

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("[api] Synthron API starting...")
    print_banner()
    await smart_router.initialize()
    logger.info(f"[api] API ready on http://{settings.dashboard.dashboard_host}:{settings.dashboard.dashboard_port}")
    yield
    logger.info("[api] Synthron API shutting down.")


app = FastAPI(
    title="Synthron API",
    description="The Neural Fabric for Autonomous AI Agents",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.dashboard.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response

# Include routers
app.include_router(tasks.router, prefix="/api/v1")
app.include_router(agents.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")
app.include_router(stream.router)  # WebSocket routes at root level


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    from synthron.orchestrator.session_manager import session_manager
    return HealthResponse(
        status="ok",
        version="0.1.0",
        providers=list(smart_router._providers.keys()),
        memory_stats={},
        performance=session_manager.stats(),
    )


@app.get("/", tags=["system"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "Synthron",
        "tagline": "The Neural Fabric for Autonomous AI Agents",
        "version": "0.1.0",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "api": "/api/v1",
        "providers": list(smart_router._providers.keys()) if smart_router._initialized else [],
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"[api] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# Mount static dashboard files if they exist
import os
static_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard", "static")
if os.path.exists(static_dir):
    app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")
