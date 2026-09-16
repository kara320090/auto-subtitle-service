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
```

## 문서와 실행 안내

- [시스템 아키텍처](docs/architecture.md): 영상 업로드부터 음성 전사, SRT 생성, 결과 영상 출력까지의 처리 흐름과 백엔드 구성
- [프론트엔드 실행 안내](frontend/README.md): React 화면 실행과 API 연동 방법

## 여행 도메인 LoRA 실험

- [학습 코드](ai/scripts/vacation/train_lora_vacation.py): Whisper large-v3에 여행 도메인 LoRA 어댑터를 적용하고 학습·검증용 JSONL을 읽어 학습하는 실험
- [평가 보고서 코드](ai/scripts/vacation/validation_report.py): 참조 전사문과 추론 결과를 비교해 파일별 WER·CER을 CSV로 저장하는 코드

실험 코드는 데이터·추론 결과 경로 등 실행 환경에 맞는 설정이 필요합니다.
위 링크는 실험 구현을 안내하며, 특정 정확도나 성능 개선 수치를 제시하지 않습니다.
