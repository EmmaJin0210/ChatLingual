import random
from src.core.modules.TokenTokenizer import TokenTokenizer
from src.core.utils.misc_utils import write_dict_to_json
from src.core.models.model_constants import (
    MODEL_ID_LM,
    LIST_FILENAMES_SENTENCES_JPN,
    LIST_FILENAMES_PREFIX_JPN,
    LIST_LEVELS_LABEL_JPN,
    ROOT_SENTENCES_JPN,
    FILENAME_TRAIN_DATA,
    FILENAME_EVAL_DATA,
    FILENAME_TEST_DATA,
)
from src.core.core_constants import JPN


tt = TokenTokenizer(language=JPN, tokenizer_id=MODEL_ID_LM)

#############################################
# Step 1: Split Sentences into Train, Eval, and Test Files
#############################################
def count_sentences(filename: str) -> int:
    filepath = f"{ROOT_SENTENCES_JPN}{filename}"
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

# Determine the minimum number of sentences across all sentence files.
min_count = min(count_sentences(filename) for filename in LIST_FILENAMES_SENTENCES_JPN)
print("Minimum sentence count across levels:", min_count)

def split_sentences_train_eval_test(filename: str, train_ratio: float = 0.8, eval_ratio: float = 0.1) -> None:
    """
    Reads sentences from the input file, randomly samples exactly min_count sentences to balance the levels,
    splits them into training (80%), evaluation (10%), and testing (10%) sets, and writes out three files:
      - train_{filename}
      - eval_{filename}
      - test_{filename}
    """
    infilepath = f"{ROOT_SENTENCES_JPN}{filename}"
    with open(infilepath, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    # Randomly sample exactly min_count sentences for balancing.
    sampled_sentences = random.sample(sentences, k=min_count)
    train_end = int(min_count * train_ratio)
    eval_end = train_end + int(min_count * eval_ratio)
    train_sentences = sampled_sentences[:train_end]
    eval_sentences = sampled_sentences[train_end:eval_end]
    test_sentences = sampled_sentences[eval_end:]
    
    train_filepath = f"{ROOT_SENTENCES_JPN}train_{filename}"
    eval_filepath = f"{ROOT_SENTENCES_JPN}eval_{filename}"
    test_filepath = f"{ROOT_SENTENCES_JPN}test_{filename}"
    
    with open(train_filepath, "w", encoding="utf-8") as f:
        for sentence in train_sentences:
            f.write(sentence + "\n")
    with open(eval_filepath, "w", encoding="utf-8") as f:
        for sentence in eval_sentences:
            f.write(sentence + "\n")
    with open(test_filepath, "w", encoding="utf-8") as f:
        for sentence in test_sentences:
            f.write(sentence + "\n")

# Apply train-eval-test split on each original sentence file.
for filename in LIST_FILENAMES_SENTENCES_JPN:
    split_sentences_train_eval_test(filename, train_ratio=0.8, eval_ratio=0.1)

#############################################
# Step 2: Generate Prefixes from the Split Sentence Files
#############################################
def generate_and_store_prefixes(filename: str, n_filename: str) -> None:
    """
    Reads each sentence from the input file, appends the <|im_end|> token to the token list (as one token),
    then generates every possible prefix from that token list and writes them to a new file.
    """
    infilepath = f"{ROOT_SENTENCES_JPN}{filename}"
    outfile_path = f"{ROOT_SENTENCES_JPN}{n_filename}"
    with open(infilepath, "r", encoding="utf-8") as infile, \
         open(outfile_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            sentence = line.strip()
            # 1) Convert to token IDs
            token_ids = tt.tokenizer.encode(sentence, add_special_tokens=False)

            # 2) Append the actual Qwen EOS token
            token_ids.append(tt.tokenizer.eos_token_id)

            # 3) For each partial prefix
            for i in range(1, len(token_ids) + 1):
                prefix_ids = token_ids[:i]
                # Now decode back to text
                prefix_str = tt.tokenizer.decode(prefix_ids, skip_special_tokens=False)
                outfile.write(prefix_str + "\n")

# Generate prefix files from the training sentence files.
for filename in LIST_FILENAMES_SENTENCES_JPN:
    train_filename = f"train_{filename}"
    generate_and_store_prefixes(filename=train_filename, n_filename=f"sampled_prefix_{filename}")

# Optionally, generate prefixes from the evaluation and test files:
for filename in LIST_FILENAMES_SENTENCES_JPN:
    eval_filename = f"eval_{filename}"
    generate_and_store_prefixes(filename=eval_filename, n_filename=f"sampled_evalprefix_{filename}")
    
for filename in LIST_FILENAMES_SENTENCES_JPN:
    test_filename = f"test_{filename}"
    generate_and_store_prefixes(filename=test_filename, n_filename=f"sampled_testprefix_{filename}")

#############################################
# Step 3: Create Train, Eval, and Test Data for FUDGE Training
#############################################
def load_all_prefix_data(prefix_type="sampled_prefix_"):
    """
    Loads all prefix data from files with a given prefix type (e.g. "sampled_prefix_")
    and returns a list of dictionaries with keys "prefix" and "label".
    """
    prefix_label_pairs = []
    for level in LIST_LEVELS_LABEL_JPN:
        filepath = f"{ROOT_SENTENCES_JPN}{prefix_type}{level}.txt"
        with open(filepath, "r", encoding="utf-8") as f:
            prefixes = [line.strip() for line in f if line.strip()]
            prefix_label_pairs.extend([{"prefix": p, "label": level} for p in prefixes])
    return prefix_label_pairs

def split_balanced_train_eval_test(prefix_label_pairs, train_ratio=0.8, eval_ratio=0.1):
    # Group prefixes by label.
    label_to_prefixes = {}
    for item in prefix_label_pairs:
        label_to_prefixes.setdefault(item["label"], []).append(item["prefix"])
    
    train_data, eval_data, test_data = [], [], []
    # We assume each level has the same number of prefixes (since sentences were balanced).
    for label, prefixes in label_to_prefixes.items():
        random.shuffle(prefixes)
        n = len(prefixes)
        train_end = int(n * train_ratio)
        eval_end = train_end + int(n * eval_ratio)
        train_data.extend([{"prefix": p, "label": label} for p in prefixes[:train_end]])
        eval_data.extend([{"prefix": p, "label": label} for p in prefixes[train_end:eval_end]])
        test_data.extend([{"prefix": p, "label": label} for p in prefixes[eval_end:]])
    return train_data, eval_data, test_data

# Load prefix data from the training prefixes.
prefix_label_pairs_train = load_all_prefix_data(prefix_type="sampled_prefix_")
train_data, eval_data, test_data = split_balanced_train_eval_test(prefix_label_pairs_train, train_ratio=0.8, eval_ratio=0.1)

# Define file paths (update FILENAME_TRAIN_DATA etc. as needed)
train_filepath = f"{ROOT_SENTENCES_JPN}{FILENAME_TRAIN_DATA}"
eval_filepath = f"{ROOT_SENTENCES_JPN}{FILENAME_EVAL_DATA}"
test_filepath = f"{ROOT_SENTENCES_JPN}{FILENAME_TEST_DATA}"

write_dict_to_json(train_data, train_filepath)
write_dict_to_json(eval_data, eval_filepath)
write_dict_to_json(test_data, test_filepath)