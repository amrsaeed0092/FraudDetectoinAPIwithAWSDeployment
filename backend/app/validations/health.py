"""Health endpoint response schema."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Safe, minimal health response."""

    status: str
    service: str
    environment: str
