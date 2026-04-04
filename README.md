# Auto Subtitle Service

Whisper 기반으로 영상에서 음성을 추출하고 자동으로 자막을 생성한 뒤,
SRT 파일과 자막이 삽입된 영상을 반환하는 서비스입니다.

## 주요 기능
- 영상 업로드
- 음성 추출
- 한국어 음성 전사
- SRT 자막 생성
- 자막 삽입 영상 생성

## 프로젝트 구조

```text
auto-subtitle-service/
├─ backend/
├─ frontend/
├─ ai/
├─ data/
├─ docs/
└─ scripts/