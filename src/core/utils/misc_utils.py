import json
import ast
###### imports for typing purposes ######
from typing import Dict, List
#########################################


def write_dict_to_json(dic: Dict, file_path: str, indent = 4) -> None:
    with open(file_path, "w", encoding = "utf-8") as outfile:
        json.dump(dic, outfile, indent = indent, ensure_ascii = False)


def read_json_to_dict(file_path: str) -> Dict:
    with open(file_path, "r", encoding = "utf-8") as infile:
        dic = json.load(infile)
    return dic


def print_and_save(to_print: str, save_to: List[str]) -> List[str]:
    print(to_print)
    save_to.append(to_print)
    return save_to


def list_str_to_list(list_string: str) -> List:
    return ast.literal_eval(list_string)


def read_txt_file_to_string(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file: {e}")