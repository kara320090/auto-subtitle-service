import json
from pathlib import Path
import pandas as pd

LOG_DIR = Path("/home/user/SWPJ3/auto-subtitle-service/ai/data/results/game_lora/logs")

trainer_state_files = sorted(LOG_DIR.glob("checkpoint-*/trainer_state.json"))

if not trainer_state_files:
    raise FileNotFoundError(f"trainer_state.json not found under: {LOG_DIR}")

trainer_state_path = trainer_state_files[-1]
print(f"[INFO] using trainer_state: {trainer_state_path}")

with trainer_state_path.open("r", encoding="utf-8") as f:
    trainer_state = json.load(f)

log_history = trainer_state.get("log_history", [])

train_rows = []
for row in log_history:
    if "loss" in row:
        train_rows.append({
            "step": row.get("step"),
            "epoch": row.get("epoch"),
            "learning_rate": row.get("learning_rate"),
            "loss": row.get("loss"),
            "grad_norm": row.get("grad_norm"),
        })

train_df = pd.DataFrame(train_rows)

print("\n=== Train Loss Table ===")
if train_df.empty:
    print("No train loss logs found.")
else:
    print(train_df.to_string(index=False))