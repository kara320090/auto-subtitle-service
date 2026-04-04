from fastapi import APIRouter, File, HTTPException, UploadFile

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