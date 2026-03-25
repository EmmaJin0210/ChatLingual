from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import gc
import torch
import argparse
from datetime import datetime
from openai import AsyncOpenAI as OpenAIClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from kani.engines.openai import OpenAIEngine
from src.core.kanis.LogTruncationKani import LogTruncationKani
from src.core.engines.ControlledGenFudgeEngine import ControlledGenFudgeEngine
from src.core.engines.SharedHFModelEngine import SharedHFModelEngine
from src.core.engines.OvergenerationEngine import OvergenerationEngine
from src.core.engines.HFOvergenerationEngine import HFOvergenerationEngine
from src.core.sysprompts import get_sysprompt_student, \
    get_sysprompt_eval_baseline, get_sysprompt_eval_detailed
from src.core.utils.language_utils import get_all_levels
from src.training.model_constants import MODEL_ID_HF_DEFAULT, MODEL_ID_OPENAI_DEFAULT, \
    LAMBDA
from src.core.utils.misc_utils import read_json_to_dict, write_dict_to_json
from src.core.utils.language_utils import load_vocab_file_to_dict


# engine consts
class EngineTypes:
    FUDGE_ENGINE = "fudge"
    OVERGEN_ENGINE = "overgen"
    HFOVERGEN_ENGINE = "hfovergen"
    SHAREDHFMODEL_ENGINE = "sharedhfmodel"
    OPENAI_ENGINE = "openai"

# prompt versions
class PromptVersions:
    BASELINE = "baseline"
    DETAILED = "detailed"

# if model is not specified, fall to default model of the engine
engine_to_default_model = {
    "fudge" : MODEL_ID_HF_DEFAULT,
    "overgen" : MODEL_ID_OPENAI_DEFAULT,
    "hfovergen" : MODEL_ID_HF_DEFAULT,
    "sharedhfmodel" : MODEL_ID_HF_DEFAULT,
    "openai" : MODEL_ID_OPENAI_DEFAULT
}

# Constants
target_language = "japanese"
all_levels = get_all_levels(target_language)
username = "anon"
log_folder = "eval/auto_eval_logs/"
prompts_folder = "eval/auto_eval_prompts/"
descs_file = "level_descs.json"
topics_file = "level_topics.json"
my_openai_key = os.getenv("OPENAI_API_KEY")

def free_cuda():
    torch.cuda.empty_cache()   # releases cached blocks back to the driver
    torch.cuda.ipc_collect()   # cleans stale CUDA IPC handles


def parse_args():
    parser = argparse.ArgumentParser(description="Language Tutor Eval Options")

    parser.add_argument(
        "-P", choices=["baseline", "detailed"], required=True,
        help="System prompt style"
    )

    parser.add_argument(
        "-TE", choices=["fudge", "overgen", "hfovergen", "sharedhfmodel", "openai"], required=True,
        help="Tutor engine (fudge or overgen or hfovergen or sharedhfmodel or openai)"
    )
    parser.add_argument(
        "-SE", choices=["sharedhfmodel", "openai"], required=True,
        help="Student engine (sharedhfmodel or openai)"
    )

    parser.add_argument(
        "-TM", type=str, required=False,
        help="Tutor model. Have paired defaults with Tutor engine. Can specify none-default model here if wanted."
    )

    parser.add_argument(
        "-SM", type=str, required=False,
        help="Student model. Have paired defaults with Student engine. Can specify none-default model here if wanted."
    )

    parser.add_argument(
        "-LAMBDA", type=float, required=False,
        help="Interpolation weight for FUDGE (0.0 = pure LM, 1.0 = pure predictor)"
    )

    return parser.parse_args()


def prompt_token_len(prompt: str, engine) -> int:
    if isinstance(engine, (OpenAIEngine, OvergenerationEngine)):                    # tiktoken
        return len(engine.tokenizer.encode(prompt))
    else:                                                   # Hugging Face
        return len(engine.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ))


def setup_student_system_prompt(student_level, desc, topic):

    system_prompt = get_sysprompt_student(
        language = target_language,
        level = student_level, 
        desc = desc, 
        topic = topic
    )
    return system_prompt


def setup_tutor_system_prompt(bot_level, student_level, version):
    match version:
        case PromptVersions.BASELINE:
            return get_sysprompt_eval_baseline(
                language = target_language,
                level = bot_level
            )
        case PromptVersions.DETAILED:
            return get_sysprompt_eval_detailed(
                language = target_language,
                tutor_level = bot_level,
                student_level = student_level
            )


async def bot_chat(tutor_engine, tutor_prompt, student_engine, student_prompt, 
                   tutor_level, student_level, topic, logfile_path, num_rounds: int = 10):
    
    tutor = LogTruncationKani(
        engine = tutor_engine,
        system_prompt = tutor_prompt,
        desired_response_tokens = 256
    )
    
    student = LogTruncationKani(
        engine = student_engine,
        system_prompt = student_prompt,
        desired_response_tokens= 256
    )
    
    conversation_history = []

    student_msg_text = None
    tutor_msg_text = None
    
    print("Starting conversation...\n")
    for round_num in range(1, num_rounds + 1):
        print(f"--- Round {round_num} ---")

        # ---- student speaks --------------------------------------------
        student_reply = await student.chat_round(tutor_msg_text if tutor_msg_text else "")
        student_msg_text = student_reply.content.strip()
        print("Student:", student_msg_text)

        # ---- tutor replies ---------------------------------------------
        tutor_reply = await tutor.chat_round(student_msg_text)
        tutor_msg_text = tutor_reply.content.replace("<|im_end|>", "").strip()
        print("Tutor:", tutor_msg_text)

        conversation_history.append({
            "student": student_msg_text,
            "tutor": tutor_msg_text
        })

    logs = read_json_to_dict(logfile_path)
    logs["conversations"].append({
        "tutor_level" : tutor_level,
        "student_level" : student_level,
        "topic" : topic,
        "script" : conversation_history
    })
    write_dict_to_json(logs, logfile_path)
    print(f"Logged conversations in {logfile_path}.")


def need_shared_hf_model(t_engine_id, t_model_id, s_engine_id, s_model_id):
    if t_engine_id in [EngineTypes.FUDGE_ENGINE, 
                       EngineTypes.SHAREDHFMODEL_ENGINE, 
                       EngineTypes.HFOVERGEN_ENGINE] \
    and s_engine_id == EngineTypes.SHAREDHFMODEL_ENGINE \
    and t_model_id == s_model_id:
        return True
    return False


def setup_shared_hf_model(model_id):
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


    return shared_model, shared_tokenizer


def setup_tutor_engine(engine_id, model_id, shared_model, shared_tokenizer, bot_level, lamda):
    return set_up_engine(
        engine_id = engine_id,
        model_id = model_id,
        shared_model = shared_model,
        shared_tokenizer = shared_tokenizer,
        bot_level = bot_level,
        lamda = lamda
    )


def setup_student_engine(engine_id, model_id, shared_model, shared_tokenizer):
    return set_up_engine(
        engine_id = engine_id,
        model_id = model_id,
        shared_model = shared_model,
        shared_tokenizer = shared_tokenizer,
        for_student = True
    )


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


def get_logfile_path(t_engine_id, prompt_version, timestamp):
    if t_engine_id == EngineTypes.FUDGE_ENGINE:
        file_folder = f"{log_folder}fudge"
        
    elif t_engine_id == EngineTypes.OVERGEN_ENGINE:
        file_folder = f"{log_folder}overgen"

    elif t_engine_id == EngineTypes.HFOVERGEN_ENGINE:
        file_folder = f"{log_folder}overgen"
    
    elif prompt_version == PromptVersions.DETAILED:
        file_folder = f"{log_folder}prompting"
    
    elif prompt_version == PromptVersions.BASELINE:
        file_folder = f"{log_folder}baseline"

    file_name = f"{timestamp}.json"
    return os.path.join(file_folder, file_name)


def init_logs(t_engine_id, t_model_id, s_engine_id, s_model_id, timestamp, lamda):
    logs = {
        "tutor_engine_id" : t_engine_id,
        "tutor_model_id" : t_model_id,
        "student_engine_id" : s_engine_id,
        "student_model_id" : s_model_id,
        "timestamp" : timestamp,
        "conversations" : []
    }
    if lamda:
        logs["lambda"] = lamda
    return logs


def main():
    start_time = datetime.now()

    shared_model, shared_tokenizer = None, None
    level_descs = read_json_to_dict(f"{prompts_folder}{descs_file}")
    level_topics = read_json_to_dict(f"{prompts_folder}{topics_file}")

    args = parse_args()
    prompt_version = args.P
    t_engine_id, s_engine_id = args.TE, args.SE
    t_model_id = args.TM if args.TM else engine_to_default_model[t_engine_id]
    s_model_id = args.SM if args.SM else engine_to_default_model[s_engine_id]
    lamda = args.LAMBDA

    if need_shared_hf_model(t_engine_id, t_model_id, s_engine_id, s_model_id):
        shared_model, shared_tokenizer = setup_shared_hf_model(t_model_id)

    # set log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_path = get_logfile_path(t_engine_id, prompt_version, timestamp)
    logs = init_logs(t_engine_id, t_model_id, s_engine_id, s_model_id, timestamp, lamda)
    write_dict_to_json(logs, logfile_path)

    student_engine = setup_student_engine(
        engine_id = s_engine_id,
        model_id = s_model_id,
        shared_model = shared_model,
        shared_tokenizer = shared_tokenizer
    )
    conversation_cnt = 0
    for bot_level in all_levels:
        tutor_engine = setup_tutor_engine(
            engine_id = t_engine_id,
            model_id = t_model_id,
            shared_model = shared_model,
            shared_tokenizer = shared_tokenizer,
            bot_level = bot_level,
            lamda = lamda
        )
        for student_level in all_levels:
            tutor_system_prompt = setup_tutor_system_prompt(
                bot_level = bot_level,
                student_level = student_level,
                version = prompt_version
            )
            tutor_sysprompt_length = prompt_token_len(tutor_system_prompt, tutor_engine)
            print(f"TUTOR SYSPROMPT IS {tutor_sysprompt_length} TOKENS")
            for topic in level_topics[student_level]:
                student_system_prompt = setup_student_system_prompt(
                    student_level = student_level,
                    desc = level_descs[student_level], 
                    topic = topic
                )

                asyncio.run(
                    bot_chat(
                        tutor_engine = tutor_engine,
                        tutor_prompt = tutor_system_prompt,
                        student_engine = student_engine,
                        student_prompt = student_system_prompt,
                        tutor_level = bot_level,
                        student_level = student_level,
                        topic = topic,
                        logfile_path = logfile_path
                    )
                )
                gc.collect()
                free_cuda()
                conversation_cnt += 1
                print(f"Conversation {conversation_cnt} complete.")
        del tutor_engine          # drop last strong refs
        gc.collect()              # reclaim cyclic leftovers
        free_cuda() 
    end_time = datetime.now()
    runtime = end_time - start_time
    logs = read_json_to_dict(logfile_path)
    logs["runtime_conversations"] = str(runtime)
    logs["conversation_count"] = conversation_cnt
    write_dict_to_json(logs, logfile_path)


if __name__ == "__main__":
    main()
