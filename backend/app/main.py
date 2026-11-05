from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

# Timestamped logs for the cognicrew.* loggers (stage-timing diagnostics).
# basicConfig is a no-op if a root handler is already configured.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .routers import auth, contact, documents, history, research, study

app = FastAPI(title="CogniCrew API", version="0.3.0")

logger = logging.getLogger("cognicrew.main")

# CORS — restrict to same-origin in production, permissive in development for port 4000 HTTPS/HTTP.
_default_origins = (
    "https://localhost:4000,"
    "https://127.0.0.1:4000,"
    "http://localhost:4000,"
    "http://127.0.0.1:4000,"
    "http://127.0.0.1:8000"
)
_allowed_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


from .response_safety import normalize_server_exception


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Ensure any unhandled server error returns structured JSON with CORS headers and sanitized messages."""
    logger.exception("Unhandled server error processing %s %s: %s", request.method, request.url.path, exc)
    status_code, user_msg = normalize_server_exception(exc)
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": user_msg,
            "error": {
                "code": "SERVER_ERROR",
                "message": user_msg,
                "path": request.url.path,
            }
        },
    )


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(research.router)
app.include_router(study.router)
app.include_router(contact.router)
app.include_router(history.router)


@app.get("/health")
def health() -> dict[str, str]:
    # Does not connect to the database: useful for validating process startup.
    return {"status": "ok", "service": "cognicrew-python-backend"}


_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_dir = _project_root / "frontend"
_static_dir = _frontend_dir if _frontend_dir.exists() else Path(__file__).resolve().parent.parent / "static"


@app.api_route("/auth", methods=["GET", "HEAD"], include_in_schema=False)
def auth_page():
    return FileResponse(_static_dir / "auth.html")


@app.api_route("/dashboard", methods=["GET", "HEAD"], include_in_schema=False)
def dashboard_page():
    return FileResponse(_static_dir / "dashboard.html")


@app.api_route("/history", methods=["GET", "HEAD"], include_in_schema=False)
def history_page():
    return FileResponse(_static_dir / "history.html")


@app.api_route("/faq", methods=["GET", "HEAD"], include_in_schema=False)
def faq_page():
    return FileResponse(_static_dir / "faq.html")


@app.api_route("/contact", methods=["GET", "HEAD"], include_in_schema=False)
def contact_page():
    return FileResponse(_static_dir / "contact.html")


# Serve the static HTML/CSS/JS frontend from the backend itself.
# Mount AFTER all API routes to avoid shadowing them.
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
