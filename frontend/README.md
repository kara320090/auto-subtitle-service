# frontend

백엔드 `auto-subtitle-service`와 연동되는 React(CRA) 프론트엔드입니다.

## 사용 방법

1. 이 폴더를 백엔드 저장소 루트 아래의 `frontend` 폴더로 넣습니다.
2. 필요하면 `.env.example`을 복사해 `.env`를 만들고 API 주소를 수정합니다.
3. 실행:

```bash
cd frontend
npm install
npm start
```

## 기본 연동 API

- `POST /upload/process`

form-data:
- `file`: 영상 파일
- `domain`: 선택 사항 (`social_news`, `ent`, `vacation`, `politics` 등)

## 현재 반영 내용

- 영상 업로드
- 도메인 선택
- 백엔드 `/upload/process` 연동
- 전사 결과/적용 도메인/사용 어댑터 표시
- SRT 다운로드 링크 / 자막 입힌 영상 다운로드 링크 표시
- 세그먼트 편집 UI

## 참고

이 프론트는 기존 목업 저장소의 방향을 유지하되, 현재 FastAPI 백엔드 구조에 맞게 실제 호출형으로 단순화한 버전입니다.
