"""FastAPI application entry point."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.health import router as health_router
from app.configuration.settings import get_settings
from app.log.logger import configure_logging
from app.api.fraud import router as fraud_router
from app.api.model_prediction import router as model_prediction_router

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize and close application resources in a predictable place."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logging.getLogger(__name__).info("application_started")
    yield
    logging.getLogger(__name__).info("application_stopped")


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

#end point GET /health
app.include_router(health_router)

#End point POST /api/v1/fraud/score
app.include_router(fraud_router, prefix=settings.api_v1_prefix)

#End point POST /api/v1/model/predict
app.include_router(
    model_prediction_router,
    prefix=settings.api_v1_prefix,
)
