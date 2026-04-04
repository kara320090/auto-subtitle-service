# 시스템 아키텍처

## 1. 프로젝트 개요
본 프로젝트는 사용자가 업로드한 영상에서 음성을 추출하고, Whisper 모델을 통해 한국어 음성 전사를 수행한 뒤, SRT 자막 파일과 자막이 삽입된 최종 영상을 생성하는 자동 자막 서비스이다.

## 2. 전체 처리 흐름
1. 사용자가 영상 파일(mp4 등)을 업로드한다.
2. 백엔드가 업로드된 영상을 `data/input`에 저장한다.
3. FFmpeg를 사용하여 영상에서 오디오(wav)를 추출한다.
4. Whisper 모델을 사용해 오디오를 텍스트로 전사한다.
5. 전사 결과의 segment 정보를 기반으로 SRT 자막 파일을 생성한다.
6. FFmpeg를 사용해 원본 영상에 자막을 입힌 mp4를 생성한다.
7. 최종적으로 자막 파일과 결과 영상을 다운로드할 수 있도록 제공한다.

## 3. 주요 디렉터리 구조
- `backend/`: FastAPI 기반 API 서버
- `frontend/`: 사용자 UI
- `data/input/`: 업로드된 원본 영상
- `data/audio/`: 추출된 오디오 파일
- `data/subtitles/`: 생성된 SRT 파일
- `data/output/`: 자막이 삽입된 최종 영상
- `docs/`: 프로젝트 문서

## 4. 백엔드 주요 구성
### routes
- `health.py`: 서버 상태 확인 API
- `upload.py`: 업로드 및 통합 처리 API
- `subtitle.py`: 오디오 추출, 전사, SRT 생성, 렌더링 API
- `download.py`: 생성 파일 다운로드 API

### services
- `audio_extractor.py`: FFmpeg 기반 오디오 추출
- `whisper_service.py`: Whisper 전사 처리
- `srt_service.py`: SRT 파일 생성
- `render_service.py`: 자막 삽입 영상 생성

### utils
- `file_utils.py`: 업로드 파일 저장 및 검증
- `ffmpeg_utils.py`: FFmpeg 실행 공통 유틸

## 5. 사용 기술
- FastAPI
- FFmpeg
- OpenAI Whisper
- Python 3.11
- PowerShell (Windows 개발 환경)

## 6. 향후 확장 방향
- 도메인 기반 자막 보정
- 처리 상태(Job) 관리
- 결과 메타데이터 저장
- 프론트엔드 업로드 및 결과 뷰어 연동