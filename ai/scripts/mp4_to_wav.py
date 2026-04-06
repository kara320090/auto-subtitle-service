import subprocess
from pathlib import Path
import shutil
import sys

input_dir = Path(r"C:\auto-subtitle-service\ai\data\raw\Sample\mp4")
output_dir = input_dir / "sample"
output_dir.mkdir(exist_ok=True)

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    print("ffmpeg를 찾을 수 없습니다.")
    print("ffmpeg 설치 후 PATH 등록을 하거나, ffmpeg.exe 절대경로를 직접 넣으세요.")
    sys.exit(1)

mp4_files = list(input_dir.glob("*.mp4"))
if not mp4_files:
    print(f"mp4 파일이 없습니다: {input_dir}")
    sys.exit(1)

for mp4_file in mp4_files:
    wav_file = output_dir / f"{mp4_file.stem}.wav"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(mp4_file),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_file)
    ]

    subprocess.run(cmd, check=True)
    print(f"변환 완료: {wav_file}")

print("전체 작업 완료")