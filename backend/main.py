"""
Main FastAPI application for OPZ Product Matcher
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger

from api.routes import data_import, product_search, opz_creation, auth, monitoring, batch
from config.settings import settings
from services.database import init_db, close_db
from services.monitoring_service import monitoring_service
from services.batch_service import batch_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting OPZ Product Matcher API...")
    await init_db()
    logger.info("Database initialized")

    # Initialize monitoring service
    await monitoring_service.initialize()
    logger.info("Monitoring service initialized")

    # Initialize batch service
    await batch_service.initialize()
    logger.info("Batch service initialized")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()


app = FastAPI(
    title="OPZ Product Matcher API",
    description="AI-powered IT procurement product matching and OPZ creation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    import traceback
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(data_import.router, prefix="/api/import", tags=["Data Import"])
app.include_router(product_search.router, prefix="/api/search", tags=["Product Search"])
app.include_router(opz_creation.router, prefix="/api/opz", tags=["OPZ Creation"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(batch.router, prefix="/api/batch", tags=["Batch Processing"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
