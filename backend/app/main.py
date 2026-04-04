from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.health import router as health_router
from backend.app.routes.upload import router as upload_router

app = FastAPI(
    title="Auto Subtitle Service API",
    description="Video upload, speech transcription, subtitle generation, and rendering API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(upload_router, prefix="/upload", tags=["Upload"])


@app.get("/")
def read_root():
    return {
        "message": "Auto Subtitle Service API is running",
        "docs_url": "/docs",
        "health_check": "/health/",
        "upload_endpoint": "/upload/",
    }