# ============================================
# 파일명: print_loss_tables.py
#
# 역할:
# - logs/checkpoint-*/trainer_state.json 중 마지막 체크포인트를 읽는다.
# - train loss / validation loss 로그를 분리해서
#   터미널에 표 형태로 출력한다.
# ============================================

import json
from pathlib import Path

LOG_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/logs")

# =========================
# 마지막 trainer_state.json 찾기
# =========================
trainer_states = sorted(LOG_DIR.glob("checkpoint-*/trainer_state.json"))

if not trainer_states:
    raise FileNotFoundError(f"trainer_state.json not found under: {LOG_DIR}")

state_path = trainer_states[-1]
print(f"[INFO] using: {state_path}\n")

data = json.loads(state_path.read_text(encoding="utf-8"))
history = data.get("log_history", [])

train_rows = []
val_rows = []

# =========================
# log_history 분리
# =========================
for row in history:
    # train loss row
    if "loss" in row:
        train_rows.append({
            "step": row.get("step", ""),
            "epoch": row.get("epoch", ""),
            "learning_rate": row.get("learning_rate", ""),
            "loss": row.get("loss", ""),
            "grad_norm": row.get("grad_norm", ""),
        })

    # validation loss row
    if "eval_loss" in row:
        val_rows.append({
            "step": row.get("step", ""),
            "epoch": row.get("epoch", ""),
            "eval_loss": row.get("eval_loss", ""),
            "eval_runtime": row.get("eval_runtime", ""),
            "eval_samples_per_second": row.get("eval_samples_per_second", ""),
            "eval_steps_per_second": row.get("eval_steps_per_second", ""),
        })

# =========================
# 출력 함수
# =========================
def fmt(x, digits=6):
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)

def print_table(title, rows, columns):
    print(f"=== {title} ===")
    if not rows:
        print("(no data)\n")
        return

    # 각 컬럼별 폭 계산
    widths = []
    for col in columns:
        max_len = len(col)
        for row in rows:
            max_len = max(max_len, len(fmt(row.get(col, ""))))
        widths.append(max_len)

    # 헤더
    header = "  ".join(col.ljust(width) for col, width in zip(columns, widths))
    print(header)

    # 행
    for row in rows:
        line = "  ".join(
            fmt(row.get(col, "")).ljust(width)
            for col, width in zip(columns, widths)
        )
        print(line)

    print()

# =========================
# train loss 표 출력
# =========================
print_table(
    "Train Loss Table",
    train_rows,
    ["step", "epoch", "learning_rate", "loss", "grad_norm"]
)

# =========================
# validation loss 표 출력
# =========================
print_table(
    "Validation Loss Table",
    val_rows,
    [
        "step",
        "epoch",
        "eval_loss",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    ]
)