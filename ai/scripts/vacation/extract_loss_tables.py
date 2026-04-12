import json
import pandas as pd

# trainer_state.json 경로
json_path = "/home/user/SWPJ3/auto-subtitle-service/ai/data/results/vacation_lora/logs/checkpoint-324/trainer_state.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

log_history = data["log_history"]

# 1) train loss 표
train_rows = [row for row in log_history if "loss" in row]

train_df = pd.DataFrame(train_rows)[
    ["step", "epoch", "learning_rate", "loss", "grad_norm"]
]

print("=== Train Loss Table ===")
print(train_df.to_string(index=False))

# csv 저장
train_df.to_csv("train_loss_table.csv", index=False, encoding="utf-8-sig")


# 2) validation loss 표
eval_rows = [row for row in log_history if "eval_loss" in row]

eval_df = pd.DataFrame(eval_rows)[
    ["step", "epoch", "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"]
]

print("\n=== Validation Loss Table ===")
print(eval_df.to_string(index=False))

# csv 저장
eval_df.to_csv("validation_loss_table.csv", index=False, encoding="utf-8-sig")