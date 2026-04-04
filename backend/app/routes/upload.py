from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.app.services.audio_extractor import extract_audio_from_video
from backend.app.services.render_service import render_video_with_subtitle
from backend.app.services.srt_service import save_srt_file
from backend.app.services.whisper_service import transcribe_audio
from backend.app.utils.file_utils import save_upload_file

router = APIRouter()


@router.post("/")
async def upload_video(file: UploadFile = File(...)):
    try:
        result = await save_upload_file(file)
        return {
            "status": "success",
            "message": "영상 업로드가 완료되었습니다.",
            "file_info": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/process")
async def process_video(request: Request, file: UploadFile = File(...)):
    """
    영상 업로드부터 자막 삽입 영상 생성까지 전체 파이프라인을 한 번에 수행한다.
    """
    try:
        # 1. 업로드 및 저장
        upload_result = await save_upload_file(file)
        video_path = upload_result["saved_path"]

        # 2. 오디오 추출
        audio_result = extract_audio_from_video(video_path)
        audio_path = audio_result["audio_path"]

        # 3. Whisper 전사
        transcription_result = transcribe_audio(audio_path)
        segments = transcription_result.get("segments", [])

        if not segments:
            raise ValueError("전사 결과가 비어 있습니다. 음성이 없거나 인식에 실패했습니다.")

        # 4. SRT 생성
        base_name = Path(upload_result["saved_filename"]).stem
        subtitle_result = save_srt_file(
            segments=segments,
            base_name=base_name,
        )
        subtitle_path = subtitle_result["subtitle_path"]

        # 5. 자막 삽입 영상 생성
        render_result = render_video_with_subtitle(
            video_path=video_path,
            subtitle_path=subtitle_path,
        )

        subtitle_filename = subtitle_result["subtitle_filename"]
        output_filename = render_result["output_filename"]

        subtitle_download_url = str(
            request.url_for("download_subtitle", filename=subtitle_filename)
        )
        video_download_url = str(
            request.url_for("download_video", filename=output_filename)
        )

        return {
            "status": "success",
            "message": "전체 영상 자막 처리 파이프라인이 완료되었습니다.",
            "pipeline_result": {
                "upload": upload_result,
                "audio_extraction": audio_result,
                "transcription": {
                    "audio_path": transcription_result["audio_path"],
                    "language": transcription_result["language"],
                    "full_text": transcription_result["full_text"],
                    "segment_count": transcription_result["segment_count"],
                    "model_name": transcription_result["model_name"],
                    "segments": transcription_result["segments"],
                },
                "subtitle": subtitle_result,
                "render": render_result,
                "downloads": {
                    "subtitle_download_url": subtitle_download_url,
                    "video_download_url": video_download_url,
                },
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"통합 처리 중 오류가 발생했습니다: {str(e)}"
        )