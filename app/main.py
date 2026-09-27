from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.core.config import get_settings
from app.core.logging import configure_logging

configure_logging()
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description="Document-scoped contract intelligence with persistent source citations.",
    version="0.7.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.include_router(documents_router)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "contractlens-api"}


@app.get("/ready", tags=["system"])
async def readiness_check() -> dict[str, str]:
    from fastapi import HTTPException
    from app.services.document_store import check_database

    try:
        check_database()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is not available.") from exc
    return {"status": "ready", "service": "contractlens-api"}
