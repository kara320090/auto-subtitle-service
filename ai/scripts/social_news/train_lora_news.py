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
# -train 세트 학습
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
    # 경로
    # =========================
    PROJECT_ROOT = Path(r"C:\auto-subtitle-service")

    MANIFEST_DIR = PROJECT_ROOT / "ai/data/processed/social_news"
    RESULT_DIR = PROJECT_ROOT / "ai/data/results/social_news_lora"

    TRAIN_JSONL = str(MANIFEST_DIR / "train.jsonl")

    ADAPTER_DIR = RESULT_DIR / "adapter"
    LOG_DIR = RESULT_DIR / "logs"

    MODEL_ID = "openai/whisper-large-v3"
    ADAPTER_NAME = "news_adapter"
    LANGUAGE = "korean"
    TASK = "transcribe"

    # =========================
    # 디바이스 / dtype
    # =========================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print(f"[INFO] device = {device}")
    print(f"[INFO] use_bf16 = {use_bf16}")
    print(f"[INFO] load_dtype = {load_dtype}")

    # =========================
    # processor / model
    # =========================
    print("[INFO] loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("[INFO] loading base model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # Whisper 설정
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Whisper 허용 최대 label 길이
    max_label_length = model.config.max_target_positions
    print(f"[INFO] max_label_length = {max_label_length}")

    # =========================
    # LoRA 설정
    # =========================
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model.add_adapter(lora_config, adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)

    if torch.cuda.is_available():
        model = model.to(device)

    def print_trainable_parameters(model):
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
        audio_path = str(Path(batch["audio"]))
        audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)

        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
        )["input_features"][0]

        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
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
        processor: Any

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features,
                return_tensors="pt",
            )

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                return_tensors="pt",
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch["attention_mask"].ne(1), -100
            )

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

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=1e-4,
        warmup_steps=100,
        num_train_epochs=3,

        # validation/eval 없음
        eval_strategy="no",
        save_strategy="steps",
        save_steps=200,
        logging_strategy="steps",
        logging_steps=20,

        fp16=(torch.cuda.is_available() and not use_bf16),
        bf16=use_bf16,

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