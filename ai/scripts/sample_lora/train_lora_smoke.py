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

# =========================
# 경로
# =========================
MANIFEST_DIR = Path(r"C:\auto-subtitle-service\ai\data\processed\sample_lora")
RESULT_DIR = Path(r"C:\auto-subtitle-service\ai\data\results\sample_lora_test")

TRAIN_JSONL = str(MANIFEST_DIR / "train.jsonl")
VAL_JSONL = str(MANIFEST_DIR / "val.jsonl")

ADAPTER_DIR = RESULT_DIR / "adapter"
LOG_DIR = RESULT_DIR / "logs"

MODEL_ID = "openai/whisper-large-v3"
ADAPTER_NAME = "default"
LANGUAGE = "korean"
TASK = "transcribe"

# =========================
# 디바이스
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# 8GB VRAM 때문에 float32로는 부담이 커서
# bf16 가능하면 bf16, 아니면 fp16 weight로만 유지
load_dtype = torch.bfloat16 if use_bf16 else torch.float16

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

# =========================
# LoRA 추가 (Transformers 최신 PEFT 통합)
# =========================
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=4,                  # 8GB VRAM 고려
    lora_alpha=8,
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
# 데이터셋
# =========================
print("[INFO] loading dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_JSONL,
        "validation": VAL_JSONL,
    },
)

def prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    audio_path = batch["audio"]

    # datasets.Audio() 대신 librosa로 직접 로드
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
# 최신 TrainingArguments 스타일
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir=str(LOG_DIR),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    warmup_steps=0,
    max_steps=1,               # smoke test
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=1,

    fp16=False,
    bf16=use_bf16,

    # smoke test에서는 gradient clipping 끔
    max_grad_norm=0.0,

    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to=[],
    dataloader_num_workers=0,
    label_names=["labels"],
    predict_with_generate=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    processing_class=processor,   # 최신 API
)

print("[INFO] training start...")
trainer.train()

ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] saving adapter to: {ADAPTER_DIR}")
model.save_pretrained(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))

print("[INFO] done.")
print("[INFO] expected files:")
print(ADAPTER_DIR / "adapter_config.json")
print(ADAPTER_DIR / "adapter_model.safetensors")