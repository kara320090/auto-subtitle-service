import json
import re
from pathlib import Path


# =========================
# 경로 설정
# =========================
pred_dir = Path(r"C:\auto-subtitle-service\ai\data\results\sample")
output_path = pred_dir / "evaluation_results.json"


# =========================
# 정규화 옵션
# =========================
REMOVE_PUNCT = False
LOWERCASE = False
KEEP_SPACES_IN_CER = False


# =========================
# 헬퍼 함수
# =========================
def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if LOWERCASE:
        text = text.lower()

    if REMOVE_PUNCT:
        text = re.sub(r"[^\w\s가-힣]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

    return text


def levenshtein(ref, hyp):
    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m]


def compute_wer(ref_text: str, hyp_text: str):
    ref_text = normalize_text(ref_text)
    hyp_text = normalize_text(hyp_text)

    ref_words = ref_text.split()
    hyp_words = hyp_text.split()

    if len(ref_words) == 0:
        return {
            "wer": 0.0 if len(hyp_words) == 0 else 1.0,
            "edits": 0 if len(hyp_words) == 0 else len(hyp_words),
            "ref_length": 0,
            "hyp_length": len(hyp_words),
        }

    dist = levenshtein(ref_words, hyp_words)

    return {
        "wer": dist / len(ref_words),
        "edits": dist,
        "ref_length": len(ref_words),
        "hyp_length": len(hyp_words),
    }


def compute_cer(ref_text: str, hyp_text: str):
    ref_text = normalize_text(ref_text)
    hyp_text = normalize_text(hyp_text)

    if KEEP_SPACES_IN_CER:
        ref_chars = list(ref_text)
        hyp_chars = list(hyp_text)
    else:
        ref_chars = list(ref_text.replace(" ", ""))
        hyp_chars = list(hyp_text.replace(" ", ""))

    if len(ref_chars) == 0:
        return {
            "cer": 0.0 if len(hyp_chars) == 0 else 1.0,
            "edits": 0 if len(hyp_chars) == 0 else len(hyp_chars),
            "ref_length": 0,
            "hyp_length": len(hyp_chars),
        }

    dist = levenshtein(ref_chars, hyp_chars)

    return {
        "cer": dist / len(ref_chars),
        "edits": dist,
        "ref_length": len(ref_chars),
        "hyp_length": len(hyp_chars),
    }


# =========================
# 평가 수행
# =========================
json_files = sorted(pred_dir.glob("*.json"))
if not json_files:
    raise FileNotFoundError(f"평가할 JSON 파일이 없습니다: {pred_dir}")

results = []

total_word_edits = 0
total_word_ref_len = 0
total_char_edits = 0
total_char_ref_len = 0

skipped_files = []

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    ground_truth_text = data.get("ground_truth_text", "").strip()
    whisper_text = data.get("whisper_text", "").strip()

    if not ground_truth_text or not whisper_text:
        skipped_files.append(json_file.name)
        continue

    wer_result = compute_wer(ground_truth_text, whisper_text)
    cer_result = compute_cer(ground_truth_text, whisper_text)

    total_word_edits += wer_result["edits"]
    total_word_ref_len += wer_result["ref_length"]
    total_char_edits += cer_result["edits"]
    total_char_ref_len += cer_result["ref_length"]

    results.append({
        "filename": json_file.stem,
        "wer": round(wer_result["wer"], 6),
        "cer": round(cer_result["cer"], 6),
        "wer_detail": wer_result,
        "cer_detail": cer_result,
        "ground_truth_text": ground_truth_text,
        "whisper_text": whisper_text,
    })

overall_wer = total_word_edits / total_word_ref_len if total_word_ref_len > 0 else 0.0
overall_cer = total_char_edits / total_char_ref_len if total_char_ref_len > 0 else 0.0

summary = {
    "evaluated_files": len(results),
    "skipped_files": skipped_files,
    "overall_wer": round(overall_wer, 6),
    "overall_cer": round(overall_cer, 6),
    "total_word_edits": total_word_edits,
    "total_word_reference_length": total_word_ref_len,
    "total_char_edits": total_char_edits,
    "total_char_reference_length": total_char_ref_len,
    "results": results,
}

with open(output_path, "w", encoding="utf-8-sig") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"평가 완료: {output_path}")
print(f"Overall WER: {overall_wer:.4f}")
print(f"Overall CER: {overall_cer:.4f}")
print(f"평가 파일 수: {len(results)}")
if skipped_files:
    print("스킵된 파일:")
    for name in skipped_files:
        print(f"- {name}")