# sample data 정량평가

## 경로

`C:\auto-subtitle-service\ai\scripts`

## 포함 파일

### 1. `mp4_to_wav.py`

#### 역할

`mp4` 파일에서 오디오를 추출해 `wav` 파일로 변환한다.  
전사 전에 영상 파일을 음성 파일로 전처리하는 단계에 사용한다.

#### 입력

`C:\auto-subtitle-service\ai\data\raw\Sample\mp4\*.mp4`

#### 출력

`C:\auto-subtitle-service\ai\data\raw\Sample\wav\*.wav`

#### 주요 기능

- `ffmpeg`를 사용해 `mp4 -> wav` 변환
- 여러 `mp4` 파일 일괄 처리 가능
- Whisper 입력용 `wav` 파일 생성

#### 사용 목적

Whisper 전사 전에 음성 파일을 준비하는 전처리 스크립트다.

---

### 2. `transcribe_whisper.py`

#### 역할

`wav` 파일을 Whisper 모델로 전사한다.  
같은 이름의 정답지 `json` 파일을 함께 읽어서 결과 JSON에 정답 텍스트와 예측 텍스트를 같이 저장한다.

#### 입력

- 음성 파일: `C:\auto-subtitle-service\ai\data\raw\Sample\wav\*.wav`
- 정답지 파일: `C:\auto-subtitle-service\ai\data\raw\Sample\json\*.json`

#### 출력

- 전사 결과: `C:\auto-subtitle-service\ai\data\processed\sample\*.json`

#### 저장 형식

각 결과 JSON에는 현재 핵심적으로 아래 두 텍스트가 들어간다.

json
{
  "ground_truth_text": "정답지 텍스트",
  "whisper_text": "Whisper 전사 결과"
}

#### 주요 기능

- `openai/whisper-large-v3` 모델 사용
- `wav` 파일 전체 순회
- 파일명 기준으로 `wav`와 `json` 매칭
- 정답지 JSON의 `video.term[].transcription`을 이어붙여 `ground_truth_text` 생성
- Whisper 전사 결과를 `whisper_text`로 저장

#### 사용 목적

전사 결과 생성과 동시에, 이후 WER/CER 평가에 바로 사용할 수 있는 비교용 JSON을 만든다.

---

### 3. `evaluate_asr.py`

#### 역할

`transcribe_whisper.py`가 생성한 JSON 파일들을 읽어서 ASR 성능평가를 수행한다.  
`ground_truth_text`와 `whisper_text`를 비교해 WER, CER를 계산한다.

#### 입력

- 평가 대상 JSON: `C:\auto-subtitle-service\ai\data\processed\sample\*.json`

#### 출력

- 평가 결과: `C:\auto-subtitle-service\ai\data\processed\sample\evaluation_results.json`

#### 주요 기능

- 파일별 WER 계산
- 파일별 CER 계산
- 전체 파일 기준 Overall WER 계산
- 전체 파일 기준 Overall CER 계산
- 결과를 JSON으로 저장

#### 평가 기준

- `ground_truth_text`: 정답지
- `whisper_text`: 모델 예측값

#### 사용 목적

Whisper 모델의 전사 성능을 정량 평가한다.


## 전체 실행 흐름

### 1단계. `mp4`를 `wav`로 변환

powershell
python .\scripts\mp4_to_wav.py

### 2단계. Whisper 전사 수행

python .\scripts\transcribe_whisper.py

### 3단계. WER / CER 평가 수행

python .\scripts\evaluate_asr.py