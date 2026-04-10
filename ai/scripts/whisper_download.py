from huggingface_hub import snapshot_download

repo_id = "openai/whisper-large-v3"
local_dir = r"C:\auto-subtitle-service\ai\models\whisper-large-v3"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)

print("다운로드 완료:", local_dir)