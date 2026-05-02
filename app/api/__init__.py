# app/api/__init__.py
from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.routes import router as predict_router
from app.api.market import router as market_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(predict_router)
api_router.include_router(health_router)
api_router.include_router(market_router)
