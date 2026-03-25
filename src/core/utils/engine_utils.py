import os
import torch
from openai import AsyncOpenAI as OpenAIClient
from kani.engines.openai import OpenAIEngine
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.engines.ControlledGenFudgeEngine import ControlledGenFudgeEngine
from src.core.engines.SharedHFModelEngine import SharedHFModelEngine
from src.core.engines.OvergenerationEngine import OvergenerationEngine
from src.core.engines.HFOvergenerationEngine import HFOvergenerationEngine
from src.core.utils.language_utils import get_all_levels
from src.core.sysprompts import get_sysprompt_eval_baseline, get_sysprompt_eval_detailed
from src.core.utils.language_utils import load_vocab_file_to_dict
from src.core.core_constants import ENGINE_ID_OPENAI_OG, ENGINE_ID_DEEPSEEK, ENGINE_ID_OPENAI
from src.training.model_constants import MODEL_ID_HF_DEFAULT, MODEL_ID_OPENAI_DEFAULT, \
    LAMBDA, MODEL_ID_LM_TINY, MODEL_ID_LM_SMALL

###### imports for typing purposes ######
from kani.engines.base import BaseEngine
#########################################


# # engine consts
class EngineTypes:
    FUDGE_ENGINE = "fudge"
    OVERGEN_ENGINE = "overgen"
    HFOVERGEN_ENGINE = "hfovergen"
    SHAREDHFMODEL_ENGINE = "sharedhfmodel" # serves as base hf engine
    OPENAI_ENGINE = "openai"

# # prompt versions
# class PromptVersions:
#     BASELINE = "baseline"
#     DETAILED = "detailed"

class ChatbotLevel:
    N1 = "n1"
    N2 = "n2"
    N3 = "n3"
    N4 = "n4"
    N5 = "n5"


def prep_overgen_engine(model_id, target_level):
    vocab_dict = load_vocab_file_to_dict(
        language = target_language,
        levels = all_levels,
        vocab_dir = os.path.join("vocab_lists", "jpwac")
    )
    client = OpenAIClient(api_key = my_openai_key)
    return OvergenerationEngine(
        language = target_language, 
        target_level = target_level, 
        vocab_dict = vocab_dict,
        grammar_dict = {},
        client = client,
        model = model_id
    )

def prep_hfovergen_engine(model_id, target_level, shared_model, shared_tokenizer):
    vocab_dict = load_vocab_file_to_dict(
        language = target_language,
        levels = all_levels,
        vocab_dir = os.path.join("vocab_lists", "jpwac")
    )
    return HFOvergenerationEngine(
        language = target_language, 
        target_level = target_level, 
        vocab_dict = vocab_dict,
        grammar_dict = {},
        model_id = model_id,
        model = shared_model,
        tokenizer = shared_tokenizer
    )


def set_up_engine(engine_id, model_id, shared_model, shared_tokenizer, 
                  bot_level = None, lamda = LAMBDA, for_student = False):
    SAMPLE_KW = {}
    if for_student:
        if engine_id == EngineTypes.SHAREDHFMODEL_ENGINE:
            SAMPLE_KW = dict(temperature = 0.7, top_p = 1.0, do_sample = True)
        elif engine_id == EngineTypes.OPENAI_ENGINE:
            SAMPLE_KW = dict(temperature = 0.7, top_p = 1.0)
        else:
            SAMPLE_KW = {}

    match engine_id:

        case EngineTypes.FUDGE_ENGINE:
            return ControlledGenFudgeEngine(
                model = shared_model, 
                tokenizer = shared_tokenizer,
                model_id = model_id,
                target_difficulty = bot_level,
                lamda = lamda
            )
        
        case EngineTypes.SHAREDHFMODEL_ENGINE:
            return SharedHFModelEngine(
                model = shared_model,
                tokenizer = shared_tokenizer,
                model_id = model_id,
                **SAMPLE_KW
            )
        
        case EngineTypes.OVERGEN_ENGINE:
            return prep_overgen_engine(
                model_id = model_id,
                target_level = bot_level
            )
        
        case EngineTypes.HFOVERGEN_ENGINE:
            return prep_hfovergen_engine(
                model_id = model_id,
                target_level = bot_level,
                shared_model = shared_model,
                shared_tokenizer = shared_tokenizer
            )
        
        case EngineTypes.OPENAI_ENGINE:
            return OpenAIEngine(
                my_openai_key, 
                model = model_id,
                **SAMPLE_KW
            )
        

def setup_tutor_engine(engine_id, model_id, shared_model, shared_tokenizer, bot_level, lamda):
    return set_up_engine(
        engine_id = engine_id,
        model_id = model_id,
        shared_model = shared_model,
        shared_tokenizer = shared_tokenizer,
        bot_level = bot_level,
        lamda = lamda
    )



# TODO: don't use this anymore
def create_engine(engine_id: str, model_id: str, **kwargs) -> BaseEngine:
    model_id = MODEL_ID_LM_SMALL
    shared_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code = True,
        device_map = "balanced_low_0",
        torch_dtype=torch.float16
    )
    shared_tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code = True
    )

    if shared_tokenizer.pad_token_id is None or shared_tokenizer.pad_token_id == shared_tokenizer.eos_token_id:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token


    return ControlledGenFudgeEngine(
        model=shared_model,
        tokenizer=shared_tokenizer,
        model_id = model_id,
        target_difficulty = "n4",
        lamda = 0.8
    )


async def clean_up(engine: BaseEngine) -> None:
    await engine.close()