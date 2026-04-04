from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.app.services.audio_extractor import extract_audio_from_video
from backend.app.services.metadata_service import save_pipeline_metadata
from backend.app.services.render_service import render_video_with_subtitle
from backend.app.services.srt_service import save_srt_file
from backend.app.services.whisper_service import transcribe_audio
from backend.app.utils.file_utils import save_upload_file
from backend.app.utils.logger import get_logger

router = APIRouter()
logger = get_logger("upload-router")


@router.post("/")
async def upload_video(file: UploadFile = File(...)):
    start_time = perf_counter()
    logger.info("upload start | filename=%s", file.filename)

    try:
        result = await save_upload_file(file)
        elapsed = round(perf_counter() - start_time, 3)

        logger.info(
            "upload success | original=%s | saved=%s | elapsed=%.3fs",
            result["original_filename"],
            result["saved_filename"],
            elapsed,
        )

        return {
            "status": "success",
            "message": "영상 업로드가 완료되었습니다.",
            "file_info": result,
            "processing_time_seconds": elapsed,
        }
    except ValueError as e:
        logger.error("upload validation failed | filename=%s | error=%s", file.filename, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("upload failed | filename=%s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/process")
async def process_video(request: Request, file: UploadFile = File(...)):
    total_start = perf_counter()
    logger.info("pipeline start | filename=%s", file.filename)

    try:
        # 1. 업로드
        step_start = perf_counter()
        upload_result = await save_upload_file(file)
        upload_elapsed = round(perf_counter() - step_start, 3)
        video_path = upload_result["saved_path"]

        logger.info(
            "pipeline step success | step=upload | saved=%s | elapsed=%.3fs",
            upload_result["saved_filename"],
            upload_elapsed,
        )

        # 2. 오디오 추출
        step_start = perf_counter()
        audio_result = extract_audio_from_video(video_path)
        audio_elapsed = round(perf_counter() - step_start, 3)
        audio_path = audio_result["audio_path"]

        logger.info(
            "pipeline step success | step=audio_extraction | audio=%s | elapsed=%.3fs",
            audio_result["audio_filename"],
            audio_elapsed,
        )

        # 3. Whisper 전사
        step_start = perf_counter()
        transcription_result = transcribe_audio(audio_path)
        transcription_elapsed = round(perf_counter() - step_start, 3)
        segments = transcription_result.get("segments", [])

        if not segments:
            raise ValueError("전사 결과가 비어 있습니다. 음성이 없거나 인식에 실패했습니다.")

        logger.info(
            "pipeline step success | step=transcription | segments=%s | elapsed=%.3fs",
            transcription_result["segment_count"],
            transcription_elapsed,
        )

        # 4. SRT 생성
        step_start = perf_counter()
        base_name = Path(upload_result["saved_filename"]).stem
        subtitle_result = save_srt_file(
            segments=segments,
            base_name=base_name,
        )
        subtitle_elapsed = round(perf_counter() - step_start, 3)
        subtitle_path = subtitle_result["subtitle_path"]

        logger.info(
            "pipeline step success | step=generate_srt | subtitle=%s | elapsed=%.3fs",
            subtitle_result["subtitle_filename"],
            subtitle_elapsed,
        )

        # 5. 자막 삽입 영상 생성
        step_start = perf_counter()
        render_result = render_video_with_subtitle(
            video_path=video_path,
            subtitle_path=subtitle_path,
        )
        render_elapsed = round(perf_counter() - step_start, 3)

        logger.info(
            "pipeline step success | step=render_video | output=%s | elapsed=%.3fs",
            render_result["output_filename"],
            render_elapsed,
        )

        subtitle_filename = subtitle_result["subtitle_filename"]
        output_filename = render_result["output_filename"]

        subtitle_download_url = str(
            request.url_for("download_subtitle", filename=subtitle_filename)
        )
        video_download_url = str(
            request.url_for("download_video", filename=output_filename)
        )

        total_elapsed = round(perf_counter() - total_start, 3)

        pipeline_payload = {
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
            "timings": {
                "upload_seconds": upload_elapsed,
                "audio_extraction_seconds": audio_elapsed,
                "transcription_seconds": transcription_elapsed,
                "generate_srt_seconds": subtitle_elapsed,
                "render_video_seconds": render_elapsed,
                "total_seconds": total_elapsed,
            },
        }

        # 6. 메타데이터 저장
        metadata_result = save_pipeline_metadata(
            payload=pipeline_payload,
            base_name=base_name,
        )

        logger.info(
            "pipeline metadata saved | file=%s",
            metadata_result["metadata_filename"],
        )

        logger.info(
            "pipeline success | filename=%s | total_elapsed=%.3fs",
            file.filename,
            total_elapsed,
        )

        return {
            "status": "success",
            "message": "전체 영상 자막 처리 파이프라인이 완료되었습니다.",
            "pipeline_result": {
                **pipeline_payload,
                "metadata": metadata_result,
            },
        }

    except ValueError as e:
        logger.error("pipeline validation failed | filename=%s | error=%s", file.filename, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error("pipeline file error | filename=%s | error=%s", file.filename, str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        logger.error("pipeline runtime error | filename=%s | error=%s", file.filename, str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("pipeline unexpected failure | filename=%s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"통합 처리 중 오류가 발생했습니다: {str(e)}"
        )