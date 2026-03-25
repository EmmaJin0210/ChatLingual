import os
import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from jreadability import compute_readability
from scipy.stats import spearmanr

from src.core.utils.misc_utils import read_json_to_dict, write_dict_to_json


MODEL_ID = "bennexx/cl-tohoku-bert-base-japanese-v3-jlpt-classifier"
ENGINES = ["baseline", "prompting", "overgen", "fudge"]
JLPT2NUM = {"n5": 1, "n4": 2, "n3": 3, "n2": 4, "n1": 5}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
id2label = model.config.id2label

print("Loaded JLPT difficulty classifier.")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute additional metrics for evaluated user-study logs.")
    parser.add_argument(
        "--input_dir",
        default="eval/user_study_eval_logs",
        help="Directory containing e_*.json user-study logs.",
    )
    return parser.parse_args()


def infer_engine(logs: dict, filename: str):
    engine = (logs.get("engine") or "").lower()
    if engine in ENGINES:
        return engine

    low = filename.lower()
    for key in ENGINES:
        if key in low:
            return key
    return "unknown"


def safe_spearman(x, y):
    if len(x) < 2:
        return None, None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, None
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def calc_neural_errors(text: str, target_level: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    detected_level = id2label[pred_id].lower()
    if detected_level not in JLPT2NUM or target_level not in JLPT2NUM:
        return None, None, None, None, None

    pred_num = JLPT2NUM[detected_level]
    target_num = JLPT2NUM[target_level]

    ce = float((pred_num - target_num) ** 2)
    over = max(0, pred_num - target_num)
    overce = float(over ** 2)
    at_or_below = pred_num <= target_num

    return detected_level, pred_num, ce, overce, at_or_below


def process_file(input_dir, filename, pooled, by_eng, counts):
    file_path = os.path.join(input_dir, filename)
    logs = read_json_to_dict(file_path)

    engine = infer_engine(logs, filename)
    target_level = (logs.get("level") or "").lower()
    if target_level not in JLPT2NUM:
        print(f"[WARN] {filename}: invalid target level '{logs.get('level')}', skipping.")
        return

    counts[engine]["files"] += 1

    jread_sum = 0.0
    jread_rounds = 0
    err_sum = 0.0
    err_over_sum = 0.0
    utt_cnt = 0
    at_or_below_cnt = 0

    for rnd in logs.get("script", []):
        err_flag = (rnd.get("error") or "").lower()
        if err_flag == "major":
            counts[engine]["major"] += 1
            continue
        if err_flag == "minor":
            counts[engine]["minor"] += 1

        tutor_text = (rnd.get("tutor") or "").strip()
        if not tutor_text:
            continue

        # JRead
        jread = float(compute_readability(tutor_text))
        rnd["jread"] = jread
        jread_sum += jread
        jread_rounds += 1

        # Neural CE
        detected_level, detected_num, ce, overce, at_or_below = calc_neural_errors(
            tutor_text, target_level
        )
        if ce is None:
            continue

        rnd["detected_level_neural"] = detected_level
        rnd["detected_level_neural_num"] = detected_num
        rnd["target_level_neural_num"] = JLPT2NUM[target_level]
        rnd["control_error_neural"] = ce
        rnd["control_error_over_neural"] = overce
        rnd["at_or_below_target_neural"] = bool(at_or_below)

        understood = rnd.get("understood", None)
        if understood is not None:
            not_understood = 0 if bool(understood) else 1

            pooled["not_understood"].append(not_understood)
            pooled["tmr"].append(rnd.get("tmr", 0.0))
            pooled["jread"].append(jread)
            pooled["ce"].append(ce)
            pooled["overce"].append(overce)

            if engine in ENGINES:
                by_eng[engine]["not_understood"].append(not_understood)
                by_eng[engine]["tmr"].append(rnd.get("tmr", 0.0))
                by_eng[engine]["jread"].append(jread)
                by_eng[engine]["ce"].append(ce)
                by_eng[engine]["overce"].append(overce)

        err_sum += ce
        err_over_sum += overce
        utt_cnt += 1
        at_or_below_cnt += int(bool(at_or_below))
        counts[engine]["rounds_used"] += 1

    logs["mean_jread_human"] = jread_sum / jread_rounds if jread_rounds else 0.0
    logs["mean_control_error_neural"] = err_sum / utt_cnt if utt_cnt else 0.0
    logs["mean_control_error_over_neural"] = err_over_sum / utt_cnt if utt_cnt else 0.0
    logs["pct_at_or_below_target_neural"] = at_or_below_cnt / utt_cnt if utt_cnt else 0.0

    write_dict_to_json(logs, file_path)
    print(f"Updated metrics for {file_path}")


def main():
    args = parse_args()

    pooled = {
        "not_understood": [],
        "tmr": [],
        "jread": [],
        "ce": [],
        "overce": [],
    }

    by_eng = defaultdict(lambda: {
        "not_understood": [],
        "tmr": [],
        "jread": [],
        "ce": [],
        "overce": [],
    })

    counts = defaultdict(lambda: {
        "major": 0,
        "minor": 0,
        "rounds_used": 0,
        "files": 0,
    })

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".json"):
            continue
        if not filename.startswith("e_"):
            continue

        file_path = os.path.join(args.input_dir, filename)
        if os.path.isdir(file_path):
            continue

        process_file(args.input_dir, filename, pooled, by_eng, counts)

    print("\n=== Spearman rho (Human Eval; per-round; metric vs not_understood) ===")
    print("[ALL ENGINES POOLED]")
    for metric in ["tmr", "jread", "ce", "overce"]:
        rho, p = safe_spearman(pooled[metric], pooled["not_understood"])
        n = len(pooled[metric])
        if rho is None:
            print(f"  {metric}: rho=NA (n={n})")
        else:
            print(f"  {metric}: rho={rho:.3f} (p={p:.3g}, n={n})")

    print("\n[PER-ENGINE]")
    for eng in ENGINES:
        d = by_eng[eng]
        n = len(d["not_understood"])
        print(
            f"\n{eng}: rounds_used={counts[eng]['rounds_used']}, "
            f"major={counts[eng]['major']}, minor={counts[eng]['minor']}, files={counts[eng]['files']}"
        )
        for metric in ["tmr", "jread", "ce", "overce"]:
            rho, p = safe_spearman(d[metric], d["not_understood"])
            if rho is None:
                print(f"  {metric}: rho=NA (n={n})")
            else:
                print(f"  {metric}: rho={rho:.3f} (p={p:.3g}, n={n})")


if __name__ == "__main__":
    main()