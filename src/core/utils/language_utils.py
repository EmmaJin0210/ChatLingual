import re
import os
import random
from src.core.utils.misc_utils import read_json_to_dict, read_txt_file_to_string
from src.core.language_constants import MAP_LEVEL_TO_DESC_WORD, LIST_ALL_LEVELS, \
    MAP_LEVEL_TO_NUM_DESIRED_TOKENS, MAP_LEVEL_TO_DIFFICULTY_SCORE, \
    DIRNAME_GRAMMAR, DIRNAME_VOCAB, DIRNAME_FEWSHOTS, DIRNAME_LEVEL_DESCS, \
    DIRNAME_LEVEL_GUIDES, DIRNAME_LEVEL_EX
from web.app_constants import ROOT_STATIC, ROOT_STATIC_DB

###### imports for typing purposes ######
from typing import List, Dict
#########################################



def get_level_desc_word(language: str, level: str) -> str:
    return MAP_LEVEL_TO_DESC_WORD[language][level]


def get_all_levels(language: str) -> List[str]:
    language = language.lower()
    return LIST_ALL_LEVELS[language]


def get_levels_below_inclusive(language: str, target_level: str) \
    -> List[str]:
    index = LIST_ALL_LEVELS[language].index(target_level)
    return LIST_ALL_LEVELS[language][:index + 1]


def get_levels_above_exclusive(language: str, target_level: str) \
    -> List[str]:
    index = LIST_ALL_LEVELS[language].index(target_level)
    return LIST_ALL_LEVELS[language][index + 1:]


def load_grammar_file_to_dict(language: str, levels: List[str], web: bool = False) -> Dict:
    dic = {}
    for level in levels:
        dic[level] = {}
        root = ROOT_STATIC if web else ROOT_STATIC_DB
        file_path = f"{root}{language}/{DIRNAME_GRAMMAR}/{level}.json"
        json_obj = read_json_to_dict(file_path)
        dic[level].update(json_obj)
    return dic


def flatten_dict_for_tokenization(dic: Dict) -> Dict:
    to_return = {}
    for level, words in dic.items():
        to_return[level] = {}
        for word, inner_dict in words.items():
            if "meaning" in inner_dict:
                if word in to_return[level]:
                    to_return[level][word] += "; " + inner_dict["meaning"]
                else:
                    to_return[level][word] = inner_dict["meaning"]
            else:
                to_return[level][word] = ""
    return to_return


def load_vocab_file_to_dict(language: str, levels: List[str], web: bool = False, vocab_dir: str = None) -> Dict:
    to_return = {}
    for level in levels:
        to_return[level] = {}
        root = ROOT_STATIC_DB
        # vocab_dir = vocab_dir if vocab_dir else DIRNAME_VOCAB
        file_path = f"{root}{language}/{vocab_dir}/{level}.json"
        json_obj = read_json_to_dict(file_path)
        to_return[level].update(json_obj)
    return to_return


def get_all_levels_of_language(language: str) -> List[str]:
    return LIST_ALL_LEVELS[language]


def filter_katakana(dic: Dict) -> Dict:
    katakana_pattern = re.compile(r'^[\u30A0-\u30FF]+$')
    for level, words_dict in dic.items():
        dic[level] = {word: details for word, details in words_dict.items() 
                      if not katakana_pattern.match(word)}
    return dic


# def get_level_desc_detailed(language: str, level: str, web: bool = False) -> str:
#     root = ROOT_STATIC if web else ROOT_STATIC_DB
#     path = f"{root}{language}/{DIRNAME_LEVEL_DESCS}/{level}.txt"
#     return read_txt_file_to_string(path)

def get_level_desc_detailed(language: str, level: str, web: bool = False) -> str:
    desc_map = {
        "n5": "absolute beginner student learning Japanese who knows only a few basic expressions",
        "n4": "beginner student learning Japanese who understands simple daily expressions and conversations",
        "n3": "early intermediate student learning Japanese who can handle everyday conversations",
        "n2": "intermediate student learning Japanese who understands most daily and work-related conversations",
        "n1": "advanced student learning Japanese who can discuss a wide range of topics fluently"
    }
    return desc_map[level]

def retrieve_shots(language: str, function: str, web: bool = False) -> Dict:
    root = ROOT_STATIC if web else ROOT_STATIC_DB
    path = f"{root}{language}/{DIRNAME_FEWSHOTS}/{function}.json"
    return read_json_to_dict(path)


def get_level_guidelines(language: str, level: str, web: bool = False) -> str:
    root = ROOT_STATIC if web else ROOT_STATIC_DB
    path = f"{root}{language}/{DIRNAME_LEVEL_GUIDES}/{level}.txt"
    return read_txt_file_to_string(path)


def get_level_example(language: str, level: str, web: bool = False) -> str:
    root = ROOT_STATIC if web else ROOT_STATIC_DB
    path = f"{root}{language}/{DIRNAME_LEVEL_EX}/{level}.txt"
    return read_txt_file_to_string(path)


def get_desired_tokens_count(language: str, level: str) -> int:
    return MAP_LEVEL_TO_NUM_DESIRED_TOKENS[language][level]


def get_level_difficulty_score(language: str, level: str) -> int:
    return MAP_LEVEL_TO_DIFFICULTY_SCORE[language][level]


def get_known_expressions_str(language: str, level: str, num_words: int = 500) -> str:
    # vocab_dict = load_vocab_file_to_dict(
    #     language = language, 
    #     levels = [level], 
    #     vocab_dir = os.path.join("vocab_lists", "jpwac")
    # )
    # words = []
    # for inner_dict in vocab_dict.values():
    #     words += list(inner_dict.keys())
    # return ", ".join(words)
    # read from the bag of words for levels
    vocab_dict = load_vocab_file_to_dict(
        language = language, 
        levels = [level], 
        vocab_dir = os.path.join("vocab_lists", "jpwac")
    )
    # join all words together into one string
    words = []
    for inner_dict in vocab_dict.values():
        words += list(inner_dict.keys())
    words = random.sample(words, num_words)
    return ", ".join(words)
