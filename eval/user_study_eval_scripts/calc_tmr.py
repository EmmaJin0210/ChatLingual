import os
import re
import argparse

from src.core.modules.TokenDetectorMatcher import TokenDetectorMatcher
from src.core.utils.language_utils import get_all_levels, load_vocab_file_to_dict
from src.core.utils.misc_utils import read_json_to_dict, write_dict_to_json


SOURCE_FOLDER = "jlpt_anki_with_reading"
TARGET_LANGUAGE = "japanese"

JP_ONLY_RE = re.compile(
    r'^[\u3005\u303B\u309D\u309E\u30FC\u3040-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]+$'
)

all_levels = get_all_levels(TARGET_LANGUAGE)
vocab_dict = load_vocab_file_to_dict(
    language=TARGET_LANGUAGE,
    levels=all_levels,
    vocab_dir=os.path.join("vocab_lists", SOURCE_FOLDER),
)

tdm = TokenDetectorMatcher(
    word_dict=vocab_dict,
    grammar_dict={},
    language=TARGET_LANGUAGE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare user-study logs and compute TMR.")
    parser.add_argument(
        "--input_dir",
        default="eval/user_study_eval_logs",
        help="Directory containing raw user-study json logs.",
    )
    return parser.parse_args()


def get_output_path(input_dir, filename):
    if filename.startswith("e_"):
        return os.path.join(input_dir, filename)
    return os.path.join(input_dir, f"e_{filename}")


def tokenize_japanese(sentence: str):
    tokens = tdm.tokenize(sentence, tokenizer="Sudachi", sudachi_mode="C", strip=True)
    return [t for t in tokens if JP_ONLY_RE.fullmatch(t)]


def calc_round_tmr(round_entry: dict):
    tutor_tokens = round_entry.get("tutor_tokens", []) or []
    hard_tokens = round_entry.get("hard_tokens", []) or []
    denom = max(len(tutor_tokens), 1)
    return float(len(hard_tokens)) / denom


def process_file(input_dir, filename):
    input_path = os.path.join(input_dir, filename)
    data = read_json_to_dict(input_path)

    total_tmr = 0.0
    tmr_rounds_used = 0

    for rnd in data.get("script", []):
        for speaker in ["student", "tutor"]:
            text = (rnd.get(speaker) or "").strip()
            if text:
                rnd[f"{speaker}_tokens"] = tokenize_japanese(text)
            else:
                rnd[f"{speaker}_tokens"] = []

        # preserve existing annotation fields if present
        rnd.setdefault("hard_tokens", [])
        rnd.setdefault("understood", True)
        rnd.setdefault("error", "")

        # compute TMR only for non-major rounds
        if (rnd.get("error") or "").lower() == "major":
            continue

        rnd["tmr"] = calc_round_tmr(rnd)
        total_tmr += rnd["tmr"]
        tmr_rounds_used += 1

    data["mean_tmr_human"] = total_tmr / tmr_rounds_used if tmr_rounds_used else 0.0
    data["tmr_rounds_used"] = tmr_rounds_used

    output_path = get_output_path(input_dir, filename)
    write_dict_to_json(data, output_path)
    print(f"Processed TMR for {input_path} -> {output_path}")


def main():
    args = parse_args()

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".json"):
            continue
        if filename.startswith("e_"):
            continue
        file_path = os.path.join(args.input_dir, filename)
        if os.path.isdir(file_path):
            continue
        process_file(args.input_dir, filename)


if __name__ == "__main__":
    main()