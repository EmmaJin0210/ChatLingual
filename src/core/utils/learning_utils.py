from random import sample
from copy import deepcopy


###### imports for typing purposes ######
from typing import Dict, List
#########################################

def pick_words_to_teach():
    pass

def pick_grammars_to_teach(schema: str, 
                           grammar_dict: Dict, 
                           user_profile: Dict) \
    -> List[Dict]:

    to_teach = []
    taught = user_profile["taught-log"]["grammar"]
    to_teach_keys = []
    if schema == "random-sample":
        trimmed = deepcopy(grammar_dict)
        for gp in taught:
            del trimmed[gp]
        to_teach_keys = sample(list(trimmed.keys()), 3)
    for key in to_teach_keys:
        to_teach.append({ key : grammar_dict[key]})
    return to_teach


def generate_practice_example():
    pass