import os

ROOT_SENTENCES_JPN = os.path.join("data", "sentences")
ROOT_MODELS = os.path.join("src", "core", "models")
ROOT_HF_CACHE_PERSONAL = "hf_cache"

LIST_LEVELS_LABEL_JPN = ["n5", "n4", "n3", "n2", "n1"]
LIST_FILENAMES_SENTENCES_JPN = ["n5.txt", "n4.txt", "n3.txt", "n2.txt", "n1.txt"]
LIST_FILENAMES_SENTENCES_WJT = ["wjt_n5.txt", "wjt_n4.txt", "wjt_n3.txt", "wjt_n2.txt", "wjt_n1.txt"]
LIST_FILENAMES_PREFIX_JPN = ["prefix_n5.txt", "prefix_n4.txt", "prefix_n3.txt", 
                             "prefix_n2.txt", "prefix_n1.txt"]

FILENAME_TRAIN_DATA = "train.json"
FILENAME_EVAL_DATA = "eval.json"
FILENAME_TEST_DATA = "test.json"


CNT_PREFIX_SAMPLES_MIN = 257013
NUM_BATCHES_FUDGE_BP = 16

MODEL_ID_BP = "answerdotai/ModernBERT-base"

MODEL_ID_LM = "Qwen/Qwen2.5-72B-Instruct"
MODEL_ID_LM_MEDIUM = "Qwen/Qwen2.5-32B-Instruct"
MODEL_ID_LM_SMALL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID_LM_TINY = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_ID_HF_DEFAULT = MODEL_ID_LM
MODEL_ID_OPENAI_DEFAULT = "gpt-4-turbo"

MODEL_ID_PREDICTOR = "emmajin0210/modernbert_output_endtok_readable"


LAMBDA = 0.8

