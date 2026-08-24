"""Operational health endpoint."""

from fastapi import APIRouter

from app.configuration.settings import get_settings
from app.validations.health import HealthResponse

router = APIRouter(tags=["operational"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a safe liveness response with no secrets or infrastructure data."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
        version=settings.api_v1_prefix
    )
