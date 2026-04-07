import shutil
from pathlib import Path

# 원본 / 대상 경로
SRC_DIR = Path('~/017.한국어_텍스트-비디오-사운드_데이터/3.개방데이터/1.데이터/Validation/01.원천데이터').expanduser()
DST_DIR = Path('/home/user/SWPJ3/auto-subtitle-service/ai/data/raw/vacation/validation')

# 옮길 확장자
EXTENSIONS = {
    '.jpg', '.jpeg', '.png',
    '.wav', '.mp3', '.flac', '.m4a',
    '.txt', '.json',
    '.mp4', '.avi', '.mov', '.mkv'
}

if not SRC_DIR.exists():
    print(f'원본 경로가 없습니다: {SRC_DIR}')
    raise SystemExit

DST_DIR.mkdir(parents=True, exist_ok=True)

files = [
    f for f in SRC_DIR.rglob('*')
    if f.is_file() and f.suffix.lower() in EXTENSIONS
]

if not files:
    print('옮길 파일이 없습니다.')
    raise SystemExit

print(f'총 {len(files)}개 파일 이동 시작')

moved = 0
for src in files:
    rel_path = src.relative_to(SRC_DIR)
    dst = DST_DIR / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    moved += 1

print(f'완료: {moved}개 파일 이동')