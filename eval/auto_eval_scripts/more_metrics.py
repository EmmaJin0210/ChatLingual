import os
import math
import argparse
from datetime import datetime
from itertools import islice

from dotenv import load_dotenv
load_dotenv()

from eval.eval_constants import (
    CONVERSATION_LOGS_FOLDER,
    EvalType,
    PERPLEXITY_MODEL_ID,
    STANZA_RESOURCES_DIR,
)

from src.core.utils.misc_utils import read_json_to_dict, write_dict_to_json
from src.core.utils.language_utils import get_all_levels, load_vocab_file_to_dict
from src.core.modules.TokenDetectorMatcher import TokenDetectorMatcher

METRICS = [
    "ave_tokens",
    "bin_rate",
    "jread",
    "perplexity",
    "control_error_neural",
    "join_utterances",
    "mdd",
    "ngram",
    "readability",
    "all",
]

SOURCE_FOLDER = "jlpt_anki_with_reading"
TARGET_LANGUAGE = "japanese"
ALL_LEVELS = get_all_levels(TARGET_LANGUAGE)

JLPT_TO_INT = {"n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5}

JREAD_TO_JLPT = [
    (float("-inf"), 3.0, "n1"),
    (3.0, 4.0, "n2"),
    (4.0, 5.0, "n3"),
    (5.0, 5.5, "n4"),
    (5.5, float("inf"), "n5"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute extra evaluation metrics on e_*.json logs")

    parser.add_argument(
        "-T",
        choices=[EvalType.BASELINE, EvalType.PROMPTING, EvalType.OVERGEN, EvalType.FUDGE],
        required=True,
        help="Evaluation type",
    )

    parser.add_argument(
        "-M",
        nargs="+",
        choices=METRICS,
        required=True,
        help="Metric(s) to compute",
    )

    return parser.parse_args()


def get_folder_path(eval_type):
    return os.path.join(CONVERSATION_LOGS_FOLDER, eval_type)


def get_evaluated_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            continue
        if filename.startswith("e_"):
            yield file_path


def expand_metrics(metrics):
    if "all" in metrics:
        return [
            "ave_tokens",
            "bin_rate",
            "jread",
            "perplexity",
            "control_error_neural",
            "join_utterances",
            "mdd",
            "ngram",
            "readability",
        ]
    return metrics


# -------------------------
# shared lazy resources
# -------------------------
_tdm = None
_ppl_tokenizer = None
_ppl_model = None
_neural_dc_model = None
_neural_dc_tokenizer = None
_neural_id2label = None
_stanza_nlp = None


def get_tdm():
    global _tdm
    if _tdm is None:
        vocab_dict = load_vocab_file_to_dict(
            language=TARGET_LANGUAGE,
            levels=ALL_LEVELS,
            vocab_dir=os.path.join("vocab_lists", SOURCE_FOLDER),
        )
        _tdm = TokenDetectorMatcher(
            word_dict=vocab_dict,
            grammar_dict={},
            language=TARGET_LANGUAGE,
        )
    return _tdm


def get_perplexity_model():
    global _ppl_tokenizer, _ppl_model
    if _ppl_model is None:
        import torch
        from huggingface_hub import login
        from transformers import AutoTokenizer, AutoModelForCausalLM

        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            raise ValueError("Please set HF_TOKEN in environment for perplexity calculation.")
        login(token=hf_token)

        _ppl_tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL_ID, trust_remote_code=True)
        _ppl_model = AutoModelForCausalLM.from_pretrained(
            PERPLEXITY_MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
        )
        _ppl_model.eval()
    return _ppl_tokenizer, _ppl_model


def get_neural_dc_model():
    global _neural_dc_model, _neural_dc_tokenizer, _neural_id2label
    if _neural_dc_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_id = "bennexx/cl-tohoku-bert-base-japanese-v3-jlpt-classifier"
        _neural_dc_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        _neural_dc_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if (
            _neural_dc_tokenizer.pad_token_id is None
            or _neural_dc_tokenizer.pad_token_id == _neural_dc_tokenizer.eos_token_id
        ):
            _neural_dc_tokenizer.pad_token = _neural_dc_tokenizer.eos_token
        _neural_dc_model.eval()
        _neural_id2label = _neural_dc_model.config.id2label
    return _neural_dc_model, _neural_dc_tokenizer, _neural_id2label


def get_stanza_pipeline():
    global _stanza_nlp
    if _stanza_nlp is None:
        import stanza
        from stanza import DownloadMethod

        _stanza_nlp = stanza.Pipeline(
            lang="ja",
            processors="tokenize,pos,lemma,depparse",
            use_gpu=True,
            dir=STANZA_RESOURCES_DIR,
            download_method=DownloadMethod.REUSE_RESOURCES,
        )
    return _stanza_nlp


# -------------------------
# metric helpers
# -------------------------
def get_jlpt_level_from_jread(score):
    for low, high, level in JREAD_TO_JLPT:
        if low <= score < high:
            return level
    return "unknown"


def calc_perplexity(text, max_length=1024):
    import torch

    tokenizer, model = get_perplexity_model()
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)

    if input_ids.size(1) == 0:
        return 0.0, 0, float("inf")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss_per_token = outputs.loss.item()
    num_tokens = input_ids.size(1)
    total_loss = loss_per_token * num_tokens
    perplexity = math.exp(loss_per_token)
    return total_loss, num_tokens, perplexity


def calc_neural_control_error(text, target_level):
    import torch

    model, tokenizer, id2label = get_neural_dc_model()
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    detected_level = id2label[predicted_class_id].lower()

    pred_num = JLPT_TO_INT[detected_level]
    target_num = JLPT_TO_INT[target_level]
    control_err = (pred_num - target_num) ** 2

    return detected_level, control_err


def calc_mdd(text):
    nlp = get_stanza_pipeline()
    doc = nlp(text)
    distances = []

    for sent in doc.sentences:
        for word in sent.words:
            if word.head == 0:
                continue
            distances.append(abs(word.id - word.head))

    return sum(distances) / len(distances) if distances else 0.0


def calc_ngram_diversity(tokens, n):
    grams = list(zip(*(islice(tokens, i, None) for i in range(n))))
    total = len(grams)
    if total == 0:
        return 0.0
    return len(set(grams)) / total


# -------------------------
# per-file metrics
# -------------------------
def metric_ave_tokens(logs):
    conversations = logs.get("conversations", [])
    total_tokens = 0
    total_utts = 0

    for conversation in conversations:
        for rnd in conversation.get("script", []):
            tokens_cnt = (
                rnd.get(f"{SOURCE_FOLDER}_detected_counts_total", 0)
                + rnd.get(f"{SOURCE_FOLDER}_undetected_counts", 0)
            )
            total_tokens += tokens_cnt
            total_utts += 1

    logs["ave_tokens_per_utterance"] = total_tokens / total_utts if total_utts else 0.0
    return logs


def metric_bin_rate(logs):
    raw_detected_cnts = 0
    raw_undetected_cnts = 0

    for conversation in logs.get("conversations", []):
        for rnd in conversation.get("script", []):
            raw_detected_cnts += rnd.get(f"{SOURCE_FOLDER}_detected_counts_total", 0)
            raw_undetected_cnts += rnd.get(f"{SOURCE_FOLDER}_undetected_counts", 0)

    logs["total_detected_counts"] = raw_detected_cnts
    logs["total_counts"] = raw_detected_cnts + raw_undetected_cnts
    logs["detection_rate"] = (
        raw_detected_cnts / (raw_detected_cnts + raw_undetected_cnts)
        if (raw_detected_cnts + raw_undetected_cnts) > 0
        else 0.0
    )
    return logs


def metric_jread(logs):
    from jreadability import compute_readability
    from scipy.stats import spearmanr

    at_target_counts = 0
    total_utts = 0
    jread_scores = []
    target_levels = []

    for i, conversation in enumerate(logs.get("conversations", [])):
        target_level = conversation.get("tutor_level", "").lower()
        at_target_cnt = 0

        for j, rnd in enumerate(conversation.get("script", [])):
            tutor_utterance = rnd.get("tutor", "").strip()
            score = compute_readability(tutor_utterance)
            level = get_jlpt_level_from_jread(score)
            at_target = (level == target_level)

            logs["conversations"][i]["script"][j]["jread"] = score
            logs["conversations"][i]["script"][j]["jread_at_target"] = at_target

            at_target_cnt += int(at_target)
            at_target_counts += int(at_target)
            total_utts += 1

        tutor_utterances_joined = " ".join(
            [rnd.get("tutor", "").strip() for rnd in conversation.get("script", [])]
        )
        score_conv = compute_readability(tutor_utterances_joined)
        level_conv = get_jlpt_level_from_jread(score_conv)

        logs["conversations"][i]["jread"] = score_conv
        logs["conversations"][i]["jread_at_target_level"] = (level_conv == target_level)
        logs["conversations"][i]["jread_at_target_cnt"] = at_target_cnt

        lvl_int = JLPT_TO_INT.get(target_level)
        if lvl_int is not None:
            jread_scores.append(score_conv)
            target_levels.append(lvl_int)

    logs["jread_at_target_percentage"] = at_target_counts / total_utts if total_utts else 0.0

    if len(jread_scores) > 1 and len(set(target_levels)) > 1:
        rho, p = spearmanr(jread_scores, target_levels)
        logs["jread_spearman_rho"] = float(rho)
        logs["jread_spearman_p"] = float(p)

    return logs


def metric_perplexity(logs):
    start_time = datetime.now()

    total_loss = 0.0
    total_tokens = 0

    for i, conversation in enumerate(logs.get("conversations", [])):
        tutor_text = "".join([rnd.get("tutor", "") for rnd in conversation.get("script", [])])
        conv_loss, conv_tokens, conv_perplexity = calc_perplexity(tutor_text)

        logs["conversations"][i]["perplexity"] = conv_perplexity
        total_loss += conv_loss
        total_tokens += conv_tokens

    logs["overall_perplexity"] = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    logs["runtime_perplexity"] = str(datetime.now() - start_time)
    return logs


def metric_control_error_neural(logs):
    total_err = 0.0
    total_utts = 0

    for i, conversation in enumerate(logs.get("conversations", [])):
        target = conversation.get("tutor_level", "").lower()

        for j, rnd in enumerate(conversation.get("script", [])):
            utt = rnd.get("tutor", "").strip()
            detected_level, err = calc_neural_control_error(utt, target)

            logs["conversations"][i]["script"][j]["detected_level_neural"] = detected_level
            logs["conversations"][i]["script"][j]["control_error_neural"] = err

            total_err += err
            total_utts += 1

    logs["mean_control_error_neural"] = total_err / max(total_utts, 1)
    return logs


def metric_join_utterances(logs):
    tdm = get_tdm()
    joined_utterances = ""
    joined_tokens = []

    for conversation in logs.get("conversations", []):
        tutor_text = "".join([rnd.get("tutor", "") for rnd in conversation.get("script", [])])
        joined_utterances += tutor_text
        tokens = tdm.tokenize(sentence=tutor_text, strip=True, jpn_only=True)
        joined_tokens += tokens

    logs["joined_utterances"] = joined_utterances
    logs["joined_tokens"] = joined_tokens
    return logs


def metric_mdd(logs):
    if "joined_utterances" not in logs:
        logs = metric_join_utterances(logs)
    logs["mdd"] = calc_mdd(logs["joined_utterances"])
    return logs


def metric_ngram(logs):
    if "joined_tokens" not in logs:
        logs = metric_join_utterances(logs)

    for n in [1, 2, 3]:
        logs[f"{n}_gram_diversity"] = calc_ngram_diversity(logs["joined_tokens"], n)
    return logs


def metric_readability(logs):
    from jreadability import compute_readability

    if "joined_utterances" not in logs:
        logs = metric_join_utterances(logs)
    logs["jreadability"] = compute_readability(logs["joined_utterances"])
    return logs


METRIC_TO_FN = {
    "ave_tokens": metric_ave_tokens,
    "bin_rate": metric_bin_rate,
    "jread": metric_jread,
    "perplexity": metric_perplexity,
    "control_error_neural": metric_control_error_neural,
    "join_utterances": metric_join_utterances,
    "mdd": metric_mdd,
    "ngram": metric_ngram,
    "readability": metric_readability,
}


def process_file(file_path, metrics):
    print(f"Processing {file_path} ...")
    logs = read_json_to_dict(file_path)

    for metric in metrics:
        print(f"  -> {metric}")
        logs = METRIC_TO_FN[metric](logs)

    write_dict_to_json(logs, file_path)
    print(f"Completed {file_path}")


def main():
    args = parse_args()
    folder_path = get_folder_path(args.T)
    metrics = expand_metrics(args.M)

    for file_path in get_evaluated_files(folder_path):
        process_file(file_path, metrics)


if __name__ == "__main__":
    main()