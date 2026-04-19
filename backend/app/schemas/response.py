from pydantic import BaseModel


class AudioExtractRequest(BaseModel):
    video_path: str


class AudioExtractResponse(BaseModel):
    status: str
    message: str
    audio_info: dict


class TranscribeRequest(BaseModel):
    audio_path: str


class TranscribeResponse(BaseModel):
    status: str
    message: str
    transcription: dict


class GenerateSRTRequest(BaseModel):
    segments: list[dict]
    base_name: str | None = None


class GenerateSRTResponse(BaseModel):
    status: str
    message: str
    subtitle_info: dict


class RenderVideoRequest(BaseModel):
    video_path: str
    subtitle_path: str
    render_mode: str = "ffmpeg"


class RenderVideoResponse(BaseModel):
    status: str
    message: str
    render_info: dict