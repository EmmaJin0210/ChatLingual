from src.core.utils.misc_utils import write_dict_to_json
from web.app_constants import ROOT_USER_PROFILES

###### imports for typing purposes ######
from typing import Dict, List, Any
#########################################

def retrieve_profile_path_from_username(username: str) -> str:
    return f"{ROOT_USER_PROFILES}{username}.json"


def retrieve_recent_grammar_learnt(user_profile: Dict, n: int = 5) -> List:
    return list(user_profile["taught-log"]["grammar"].items())[:n]


def write_updated_profile_to_file(user_profile: Dict, filepath: str) -> None:
    write_dict_to_json(user_profile, filepath)


def retrieve_user_name(user_profile: Dict) -> str:
    return user_profile["name"]


def retrieve_user_interests_from_profile(user_profile: Dict) -> str:
    interests = user_profile["interests"]
    if len(interests) != 0:
        return str(interests)
    return "UNKNOWN"


def retrieve_user_info_from_profile(user_profile: Dict) -> str:
    info = user_profile["personal-info"]
    if len(info) != 0:
        return str(info)
    return "UNKNOWN"


def retrieve_past_topics_from_profile(user_profile: Dict) -> str:
    past_topics_list = user_profile["past-topics"]
    if len(past_topics_list) != 0:
        return str(past_topics_list)
    return "NONE"


def update_learning_log_in_profile(grammar_taught: List[Dict], 
                                   user_profile: Dict) \
    -> Dict:

    old_grammar_log = user_profile["taught-log"]["grammar"]
    for dic in grammar_taught:
        updated_grammar_log = {**dic, **old_grammar_log}
        old_grammar_log = updated_grammar_log

    user_profile["taught-log"]["grammar"] = updated_grammar_log
    return user_profile


def update_field_in_profile(user_profile: Dict, 
                            path_to_field: List[str], 
                            new_value: Any) \
    -> Dict:
    for key in path_to_field:
        return

