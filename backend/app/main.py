from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.download import router as download_router
from backend.app.routes.health import router as health_router
from backend.app.routes.upload import router as upload_router
from backend.app.routes.subtitle import router as subtitle_router
from backend.app.utils.logger import get_logger

logger = get_logger("main")

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
app.include_router(subtitle_router, prefix="/subtitle", tags=["Subtitle"])
app.include_router(download_router, prefix="/download", tags=["Download"])


@app.on_event("startup")
def on_startup():
    logger.info("Auto Subtitle Service API started")


@app.get("/")
def read_root():
    return {
        "message": "Auto Subtitle Service API is running",
        "docs_url": "/docs",
        "health_check": "/health/",
        "upload_endpoint": "/upload/",
        "upload_process_endpoint": "/upload/process",
        "audio_extract_endpoint": "/subtitle/extract-audio",
        "download_subtitle_example": "/download/subtitle/example.srt",
        "download_video_example": "/download/video/example.mp4",
    }