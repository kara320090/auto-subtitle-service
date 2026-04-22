# ============================================
# 파일명: train_lora_news_train_only.py
#
# 역할:
# - Hugging Face에서 whisper-large-v3 base 모델을 불러온다.
# - social_news 도메인용 LoRA adapter를 모델에 붙인다.
# - train.jsonl 하나만 읽어 LoRA 학습을 수행한다.
# - 학습이 끝나면 adapter 파일을 저장한다.
#
# 입력:
# - C:\auto-subtitle-service\ai\data\processed\social_news\train.jsonl
#
# 출력:
# - C:\auto-subtitle-service\ai\data\results\social_news_lora\adapter\adapter_config.json
# - C:\auto-subtitle-service\ai\data\results\social_news_lora\adapter\adapter_model.safetensors
# - C:\auto-subtitle-service\ai\data\results\social_news_lora\logs\*
#
# 목적:
# - train 세트만으로 social_news 도메인 LoRA를 학습한다.
#
# 학습 방식:
# - 각 샘플은 "오디오 구간 1개 + 정답 문장 1개" 쌍으로 학습한다.
# - manifest에 start/end가 있으면 해당 구간만 잘라서 사용한다.
# - manifest에 start/end가 없으면 전체 오디오를 사용한다.
#
# 핵심 변수 역할:
# - audio         : 원본 wav 파일 경로
# - start / end   : 원본 오디오에서 사용할 구간 시간(초)
# - audio_segment : 실제 학습에 사용할 잘린 오디오
# - input_features: Whisper encoder에 들어가는 음성 feature
# - text          : 사람이 보는 정답 전사문
# - labels        : 모델이 맞혀야 하는 정답 토큰 ID
#
# 참고:
# - 이 파일은 "학습 코드"다.
# - validation / test는 여기서 사용하지 않는다.
# - Whisper의 label 최대 길이(max_target_positions)를 넘는 샘플은 학습 전에 제거한다.
# - Windows 환경에서는 multiprocessing 이슈를 피하기 위해
#   if __name__ == "__main__": 구조를 유지한다.
# ============================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import librosa
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)


def main():
    # =========================
    # 경로 설정
    # =========================
    PROJECT_ROOT = Path(r"C:\auto-subtitle-service")

    # train manifest가 저장된 폴더
    MANIFEST_DIR = PROJECT_ROOT / "ai/data/processed/social_news"

    # 학습 결과(adapter, logs)를 저장할 폴더
    RESULT_DIR = PROJECT_ROOT / "ai/data/results/social_news_lora"

    # train-only 학습이므로 train.jsonl만 사용
    TRAIN_JSONL = str(MANIFEST_DIR / "train.jsonl")

    # 최종 adapter 저장 폴더
    ADAPTER_DIR = RESULT_DIR / "adapter"

    # Trainer 로그 / checkpoint 저장 폴더
    LOG_DIR = RESULT_DIR / "logs"

    MODEL_ID = "openai/whisper-large-v3"
    ADAPTER_NAME = "news_adapter"
    LANGUAGE = "korean"
    TASK = "transcribe"

    # =========================
    # 디바이스 / dtype 설정
    # =========================
    # GPU가 있으면 cuda 사용
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # GPU가 bf16을 지원하면 bf16 사용, 아니면 fp16 사용
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print(f"[INFO] device = {device}")
    print(f"[INFO] use_bf16 = {use_bf16}")
    print(f"[INFO] load_dtype = {load_dtype}")

    # =========================
    # processor / model 로드
    # =========================
    # processor:
    # - 오디오를 Whisper 입력 feature(input_features)로 변환
    # - 텍스트를 토큰(label ids)으로 변환
    print("[INFO] loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # base Whisper 모델 로드
    print("[INFO] loading base model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # Whisper 생성 설정
    # - 한국어 전사 모드
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Whisper가 허용하는 label 최대 길이
    # 이 길이를 넘는 샘플은 loss 계산에서 오류가 날 수 있음
    max_label_length = model.config.max_target_positions
    print(f"[INFO] max_label_length = {max_label_length}")

    # =========================
    # LoRA 설정
    # =========================
    # q_proj / v_proj attention 층에 LoRA adapter를 붙인다.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    # base 모델에 adapter 추가
    model.add_adapter(lora_config, adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)

    if torch.cuda.is_available():
        model = model.to(device)

    def print_trainable_parameters(model):
        """
        현재 학습되는 파라미터 수를 확인한다.

        목적:
        - base 전체가 아니라 LoRA adapter 위주로 학습되는지 확인
        """
        trainable_params = 0
        all_params = 0

        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        pct = 100 * trainable_params / all_params
        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_params:,d} || "
            f"trainable%: {pct:.4f}"
        )

    print_trainable_parameters(model)

    # =========================
    # 데이터셋 로드 (train만)
    # =========================
    print("[INFO] loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_JSONL,
        },
    )

    def get_label_length(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        정답 text를 tokenizer로 변환해서 label 길이를 계산한다.

        역할:
        - text -> labels 토큰 길이 계산
        - Whisper 허용 최대 길이 초과 여부 확인

        중요:
        - Whisper는 labels 길이가 max_target_positions 이하여야 한다.
        """
        input_ids = processor.tokenizer(batch["text"]).input_ids
        batch["label_length"] = len(input_ids)
        return batch

    print("[INFO] calculating label lengths...")
    dataset = dataset.map(get_label_length)

    train_before = len(dataset["train"])

    print("[INFO] filtering long-label samples...")
    dataset = dataset.filter(lambda x: x["label_length"] <= max_label_length)

    train_after = len(dataset["train"])

    print(f"[INFO] train kept: {train_after}/{train_before}")
    print(f"[INFO] train removed: {train_before - train_after}")

    def prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        각 샘플을 Whisper 학습 입력 형태로 변환한다.

        입력:
        - audio : 원본 wav 파일 경로
        - text  : 정답 문장
        - start : 사용할 구간 시작 시간(초) [있으면 사용]
        - end   : 사용할 구간 종료 시간(초) [있으면 사용]

        처리:
        1. audio 파일 로드
        2. start/end가 있으면 해당 구간만 잘라서 audio_segment 생성
        3. audio_segment를 input_features로 변환
        4. text를 labels(token ids)로 변환

        출력:
        - input_features : Whisper encoder 입력
        - labels         : Whisper decoder 정답
        """

        # -------------------------
        # 원본 오디오 경로
        # -------------------------
        audio_path = str(Path(batch["audio"]))

        # -------------------------
        # 정답 문장
        # -------------------------
        # 사람이 보는 정답 텍스트
        text = batch["text"]

        # -------------------------
        # start / end 구간 정보
        # -------------------------
        # manifest에 있으면 해당 구간만 사용
        start = batch.get("start", None)
        end = batch.get("end", None)

        # -------------------------
        # 오디오 로드
        # -------------------------
        # start/end가 있으면 해당 구간만 부분 로드
        # 없으면 전체 오디오 로드
        if start is not None and end is not None:
            start = float(start)
            end = float(end)

            if end <= start:
                raise ValueError(
                    f"Invalid segment range: start={start}, end={end}, file={audio_path}"
                )

            duration = end - start

            # 원본 전체를 읽지 않고 필요한 구간만 로드
            audio_segment, _ = librosa.load(
                audio_path,
                sr=16000,
                mono=True,
                offset=start,
                duration=duration,
            )
        else:
            # start/end가 없으면 전체 오디오를 사용
            audio_segment, _ = librosa.load(
                audio_path,
                sr=16000,
                mono=True,
            )

        if audio_segment.size == 0:
            raise ValueError(f"Empty audio segment: file={audio_path}")

        # -------------------------
        # Whisper 입력 feature 생성
        # -------------------------
        # audio_segment -> input_features
        # 이 값이 Whisper encoder에 들어간다.
        batch["input_features"] = processor.feature_extractor(
            audio_segment,
            sampling_rate=16000,
        )["input_features"][0]

        # -------------------------
        # 정답 text -> labels
        # -------------------------
        # 사람이 보는 문장을 토큰 ID로 바꾼다.
        # 이 값이 Whisper decoder가 맞혀야 하는 정답이다.
        batch["labels"] = processor.tokenizer(text).input_ids

        return batch

    print("[INFO] preprocessing dataset...")
    dataset = dataset.map(
        prepare_batch,
        remove_columns=dataset["train"].column_names,
    )

    # =========================
    # Data collator
    # =========================
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        배치 단위로 input_features와 labels를 padding한다.

        역할:
        - 길이가 다른 오디오 feature들을 같은 길이로 맞춤
        - 길이가 다른 labels도 같은 길이로 맞춤
        - label padding 위치는 -100으로 바꿔 loss 계산에서 무시
        """
        processor: Any

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # -------------------------
            # input_features padding
            # -------------------------
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features,
                return_tensors="pt",
            )

            # -------------------------
            # labels padding
            # -------------------------
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                return_tensors="pt",
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch["attention_mask"].ne(1), -100
            )

            # BOS 토큰 제거
            bos_token_id = self.processor.tokenizer.bos_token_id
            if bos_token_id is not None and labels.shape[1] > 0:
                if (labels[:, 0] == bos_token_id).all():
                    labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # =========================
    # 학습 설정 (train only)
    # =========================
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(LOG_DIR),

        # 실제 GPU에 들어가는 배치 크기
        per_device_train_batch_size=1,

        # gradient accumulation으로 유효 배치 크기 확보
        gradient_accumulation_steps=4,

        learning_rate=1e-4,
        warmup_steps=100,
        num_train_epochs=3,

        # train-only 학습이므로 eval 없음
        eval_strategy="no",

        # checkpoint / logging
        save_strategy="steps",
        save_steps=200,
        logging_strategy="steps",
        logging_steps=20,

        # precision
        fp16=(torch.cuda.is_available() and not use_bf16),
        bf16=use_bf16,

        # 안정성 / 메모리
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        remove_unused_columns=False,

        report_to=[],
        dataloader_num_workers=0,
        label_names=["labels"],
        predict_with_generate=False,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        processing_class=processor,
    )

    # =========================
    # 학습 실행
    # =========================
    print("[INFO] training start...")
    trainer.train()

    # =========================
    # adapter 저장
    # =========================
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] saving adapter to: {ADAPTER_DIR}")

    model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))

    print("[INFO] done.")
    print("[INFO] expected files:")
    print(ADAPTER_DIR / "adapter_config.json")
    print(ADAPTER_DIR / "adapter_model.safetensors")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()