import os
import argparse
from datetime import datetime
from src.core.modules.TokenDetectorMatcher import TokenDetectorMatcher
from src.core.modules.DifficultyCalculator import DifficultyCalculator
from src.core.utils.language_utils import get_all_levels, \
    get_levels_above_exclusive, get_levels_below_inclusive
from src.core.utils.language_utils import load_vocab_file_to_dict
from src.core.utils.misc_utils import read_json_to_dict, write_dict_to_json
from eval.eval_constants import CONVERSATION_LOGS_FOLDER, EvalType

source_folder = "jlpt_anki_with_reading"


target_language = "japanese"
all_levels = get_all_levels(target_language)

vocab_dict = load_vocab_file_to_dict(
    language = target_language,
    levels = all_levels,
    vocab_dir = os.path.join("vocab_lists", source_folder)
)

tdm = TokenDetectorMatcher(word_dict = vocab_dict,
                           grammar_dict = {},
                           language = target_language)
dc = DifficultyCalculator()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tutor Difficulty Options")

    parser.add_argument(
        "-T", 
        choices = [EvalType.BASELINE, EvalType.PROMPTING, EvalType.OVERGEN, EvalType.FUDGE], 
        required = True,
        help = "Evaluation Type: baseline or prompting or overgen or fudge"
    )

    return parser.parse_args()


def get_folder_path(eval_type):
    return os.path.join(CONVERSATION_LOGS_FOLDER, eval_type)


def evaluate_tutor_msgs_for_all_files(folder_path):
    filenames = os.listdir(folder_path)
    for filename in filenames:
        if f"e_{filename}" in filenames:
            continue
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            continue
        evaluate_tutor_msgs_for_file(folder_path, filename)
        print(f"Completed difficulty evaluation for {file_path}")


def evaluate_tutor_msgs_for_file(folder_path, filename):
    num_valid_rounds = 0
    total_score      = 0.0
    raw_above_cnts   = 0
    raw_total_cnts   = 0

    file_path  = os.path.join(folder_path, filename)
    start_time = datetime.now()
    logs       = read_json_to_dict(file_path)

    for idx, conv in enumerate(logs.get("conversations", [])):

        updated_conv, valid_rnds, sum_score_conv, above_cnts, total_cnts = \
            calc_difficulty_for_conversation(conv)
        print(f"For conversation {idx}: ")
        print(f"valid rounds = {valid_rnds}, sum_score = {sum_score_conv}, above cnts = {above_cnts}, total cnts = {total_cnts}")
        logs["conversations"][idx] = updated_conv

        if conv.get("tutor_level", "").lower() == "n1":
            continue

        num_valid_rounds += valid_rnds
        total_score      += sum_score_conv
        raw_above_cnts   += above_cnts
        raw_total_cnts   += total_cnts

    ave_by_utt  = total_score    / num_valid_rounds if num_valid_rounds else 0.0
    ave_overall = raw_above_cnts / raw_total_cnts   if raw_total_cnts   else 0.0

    logs[f"{source_folder}_ave_diff_score_by_utterance_fixed"] = ave_by_utt   #  TMR-U
    logs[f"{source_folder}_ave_diff_score_overall_fixed"]      = ave_overall  #  TMR
    logs[f"{source_folder}_runtime_difficulty_eval"]     = str(datetime.now() - start_time)

    new_file_path = file_path if filename.startswith("e_") \
        else os.path.join(folder_path, f"e_{filename}")
    write_dict_to_json(logs, new_file_path)


def calc_difficulty_for_conversation(conversation):
    at_target_cnt = 0
    valid_rounds = 0
    sum_score = 0.0
    raw_above_cnts, raw_total_cnts = 0, 0
    target_level = conversation.get("tutor_level", "")
    levels_above = get_levels_above_exclusive(target_language, target_level)
    for ind, rnd in enumerate(conversation.get("script", [])):
        tutor_utterance = rnd.get("tutor", "").strip()
        utt_at_target, utt_difficulty, detected, undetected = \
            calc_difficulty_for_text(tutor_utterance, target_level)
        detected_cnts_to_log = {level : len(toks) for level, toks in detected.items()}
        detected_cnts_total = sum(detected_cnts_to_log.values())
        above_cnts_total = sum(len(toks) for level, toks in detected.items() if level in levels_above)
        undetected_cnts = len(undetected)
        conversation["script"][ind][f"{source_folder}_difficulty_score"] = utt_difficulty
        conversation["script"][ind][f"{source_folder}_at_target_level"] = utt_at_target
        conversation["script"][ind][f"{source_folder}_detected_counts"] = detected_cnts_to_log
        conversation["script"][ind][f"{source_folder}_detected_counts_total"] = detected_cnts_total
        conversation["script"][ind][f"{source_folder}_undetected_counts"] = undetected_cnts
        conversation["script"][ind][f"{source_folder}_detected"] = detected
        
        at_target_cnt = (at_target_cnt + 1) if utt_at_target else at_target_cnt

        if utt_difficulty and \
            detected_cnts_total / (detected_cnts_total + undetected_cnts) > 0.25:
            valid_rounds += 1
            sum_score += utt_difficulty
            raw_above_cnts += above_cnts_total
            raw_total_cnts += detected_cnts_total

    tutor_utterances_joined = " ".join([rnd.get("tutor", "").strip() 
                                        for rnd in conversation.get("script", [])])
    conv_at_target, conv_difficulty, detected, undetected = \
        calc_difficulty_for_text(tutor_utterances_joined, target_level)
    conversation[f"{source_folder}_difficulty_score"] = conv_difficulty
    conversation[f"{source_folder}_at_target_level"] = conv_at_target
    conversation[f"{source_folder}_num_rounds_at_target_level"] = at_target_cnt

    return conversation, valid_rounds, sum_score, raw_above_cnts, raw_total_cnts


def calc_difficulty_for_text(text, target_level):
    tokens = tdm.tokenize(text, tokenizer='Sudachi', sudachi_mode='C', strip = True)
    detected, undetected = tdm.detect_tokens_at_levels(tokens, all_levels, scope = ['v'])
    undetected_cnt = len(undetected)
    above_levels = get_levels_above_exclusive(target_language, target_level)
    below_levels = get_levels_below_inclusive(target_language, target_level)
    above = {level: tokens for level, tokens in detected.items() if level in above_levels}
    below = {level: tokens for level, tokens in detected.items() if level in below_levels}
    is_at_target = dc.at_target_level(above, below)
    difficulty_score = dc.calc_difficulty_score(above, below, undetected_cnt)
    return is_at_target, difficulty_score, detected, undetected


def main():
    args = parse_args()
    eval_type = args.T

    folder_path = get_folder_path(eval_type)
    evaluate_tutor_msgs_for_all_files(folder_path)


if __name__ == "__main__":
    main()