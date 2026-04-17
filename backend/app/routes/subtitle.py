from fastapi import APIRouter, HTTPException

from backend.app.schemas.response import (
    AudioExtractRequest,
    GenerateSRTRequest,
    RenderVideoRequest,
    TranscribeRequest,
)
from backend.app.services.audio_extractor import extract_audio_from_video
from backend.app.services.render_service import render_video_with_subtitle
from backend.app.services.srt_service import save_srt_file
from backend.app.services.whisper_service import transcribe_audio

router = APIRouter()


@router.post("/extract-audio")
def extract_audio(request: AudioExtractRequest):
    try:
        result = extract_audio_from_video(request.video_path)
        return {
            "status": "success",
            "message": "오디오 추출이 완료되었습니다.",
            "audio_info": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"오디오 추출 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/transcribe")
def transcribe(request: TranscribeRequest):
    try:
        result = transcribe_audio(
            audio_path=request.audio_path,
            domain=request.domain,
        )
        return {
            "status": "success",
            "message": "음성 전사가 완료되었습니다.",
            "transcription": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"음성 전사 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/generate-srt")
def generate_srt(request: GenerateSRTRequest):
    try:
        result = save_srt_file(
            segments=request.segments,
            base_name=request.base_name,
        )
        return {
            "status": "success",
            "message": "SRT 파일 생성이 완료되었습니다.",
            "subtitle_info": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SRT 생성 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/render-video")
def render_video(request: RenderVideoRequest):
    try:
        result = render_video_with_subtitle(
            video_path=request.video_path,
            subtitle_path=request.subtitle_path,
        )
        return {
            "status": "success",
            "message": "자막 삽입 영상 생성이 완료되었습니다.",
            "render_info": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"자막 삽입 영상 생성 중 오류가 발생했습니다: {str(e)}"
        )