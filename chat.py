from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import asyncio
import torch
import subprocess

from transformers import AutoModelForCausalLM, AutoTokenizer
from kani.engines.openai import OpenAIEngine
from openai import AsyncOpenAI as OpenAIClient

from src.core.kanis.LogTruncationKani import LogTruncationKani
from src.core.engines.ControlledGenFudgeEngine import ControlledGenFudgeEngine
from src.core.engines.SharedHFModelEngine import SharedHFModelEngine
from src.core.engines.HFOvergenerationEngine import HFOvergenerationEngine
from src.core.engines.OvergenerationEngine import OvergenerationEngine

from src.core.sysprompts import get_sysprompt_eval_baseline, get_sysprompt_eval_detailed
from src.core.utils.language_utils import get_all_levels, load_vocab_file_to_dict
from src.training.model_constants import MODEL_ID_HF_DEFAULT, MODEL_ID_OPENAI_DEFAULT, LAMBDA, MODEL_ID_LM_TINY

# USE YOUR EXISTING SPEECH UTILS AS-IS
from src.core.utils.speech_utils import speech_to_text, text_to_speech, play_wav, ExitProgram
from web.app_constants import ROOT_TEMP_DATA, FILENAME_AUDIO_OUTPUT  # for playback path


# -----------------------------
# constants / enums
# -----------------------------
class EngineTypes:
    FUDGE_ENGINE = "fudge"
    OVERGEN_ENGINE = "overgen"
    HFOVERGEN_ENGINE = "hfovergen"
    SHAREDHFMODEL_ENGINE = "sharedhfmodel"
    OPENAI_ENGINE = "openai"

class PromptVersions:
    BASELINE = "baseline"
    DETAILED = "detailed"

engine_to_default_model = {
    EngineTypes.FUDGE_ENGINE: MODEL_ID_LM_TINY,
    EngineTypes.OVERGEN_ENGINE: MODEL_ID_OPENAI_DEFAULT,
    EngineTypes.HFOVERGEN_ENGINE: MODEL_ID_HF_DEFAULT,
    EngineTypes.SHAREDHFMODEL_ENGINE: MODEL_ID_HF_DEFAULT,
    EngineTypes.OPENAI_ENGINE: MODEL_ID_OPENAI_DEFAULT,
}

target_language = "japanese"
all_levels = get_all_levels(target_language)
my_openai_key = os.getenv("OPENAI_API_KEY")


def free_cuda():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def parse_args():
    p = argparse.ArgumentParser("Terminal Tutor Chat (same tutor engines, optional voice)")

    # tutor-only flags
    p.add_argument("-P", choices=[PromptVersions.BASELINE, PromptVersions.DETAILED], required=True)
    p.add_argument("-TE", choices=[
        EngineTypes.FUDGE_ENGINE,
        EngineTypes.HFOVERGEN_ENGINE,
        EngineTypes.SHAREDHFMODEL_ENGINE,
        EngineTypes.OPENAI_ENGINE,
        EngineTypes.OVERGEN_ENGINE,
    ], required=True)
    p.add_argument("-LEVEL", choices=["n5", "n4", "n3", "n2", "n1"], required=True)
    p.add_argument("-TM", type=str, required=False)
    p.add_argument("-LAMBDA", type=float, required=False)

    # voice toggles
    p.add_argument("--voice_in", action="store_true", help="Use mic speech-to-text via speech_utils.py")
    p.add_argument("--voice_out", action="store_true", help="Use text-to-speech via speech_utils.py + terminal playback")
    p.add_argument("--tts_voice", type=str, default="onyx")  # speech_utils uses onyx already; kept for future

    p.add_argument("--quit", nargs="*", default=["q", "quit", "exit"])
    return p.parse_args()


def need_shared_hf_model(t_engine_id: str) -> bool:
    return t_engine_id in [
        EngineTypes.FUDGE_ENGINE,
        EngineTypes.SHAREDHFMODEL_ENGINE,
        EngineTypes.HFOVERGEN_ENGINE,
    ]


def setup_shared_hf_model(model_id: str):
    shared_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="balanced_low_0",
        torch_dtype=torch.float16,
    )
    shared_tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if shared_tokenizer.pad_token_id is None or shared_tokenizer.pad_token_id == shared_tokenizer.eos_token_id:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token
    return shared_model, shared_tokenizer


def prep_hfovergen_engine(model_id, target_level, shared_model, shared_tokenizer):
    vocab_dict = load_vocab_file_to_dict(
        language=target_language,
        levels=all_levels,
        vocab_dir=os.path.join("vocab_lists", "jpwac"),
    )
    return HFOvergenerationEngine(
        language=target_language,
        target_level=target_level,
        vocab_dict=vocab_dict,
        grammar_dict={},
        model_id=model_id,
        model=shared_model,
        tokenizer=shared_tokenizer,
    )


def prep_overgen_engine(model_id, target_level):
    vocab_dict = load_vocab_file_to_dict(
        language=target_language,
        levels=all_levels,
        vocab_dir=os.path.join("vocab_lists", "jpwac"),
    )
    client = OpenAIClient(api_key=my_openai_key)
    return OvergenerationEngine(
        language=target_language,
        target_level=target_level,
        vocab_dict=vocab_dict,
        grammar_dict={},
        client=client,
        model=model_id,
    )


def setup_tutor_engine(engine_id, model_id, bot_level, lamda, shared_model, shared_tokenizer):
    if engine_id == EngineTypes.FUDGE_ENGINE:
        return ControlledGenFudgeEngine(
            model=shared_model,
            tokenizer=shared_tokenizer,
            model_id=model_id,
            target_difficulty=bot_level,
            lamda=lamda if lamda is not None else LAMBDA,
        )
    if engine_id == EngineTypes.SHAREDHFMODEL_ENGINE:
        return SharedHFModelEngine(
            model=shared_model,
            tokenizer=shared_tokenizer,
            model_id=model_id,
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
        )
    if engine_id == EngineTypes.HFOVERGEN_ENGINE:
        return prep_hfovergen_engine(model_id, bot_level, shared_model, shared_tokenizer)
    if engine_id == EngineTypes.OPENAI_ENGINE:
        return OpenAIEngine(my_openai_key, model=model_id, temperature=0.7, top_p=1.0)
    if engine_id == EngineTypes.OVERGEN_ENGINE:
        return prep_overgen_engine(model_id, bot_level)
    raise ValueError(f"Unknown engine: {engine_id}")


def setup_tutor_system_prompt(prompt_version: str, bot_level: str):
    if prompt_version == PromptVersions.BASELINE:
        return get_sysprompt_eval_baseline(language=target_language, level=bot_level)
    # detailed needs student_level; for terminal chat, set it equal to bot_level
    return get_sysprompt_eval_detailed(language=target_language, tutor_level=bot_level, student_level=bot_level)


def play_audio_terminal(filepath: str):
    """
    No changes to speech_utils needed.
    Tries ffplay, then mpv. If neither exists, prints where the file is.
    """
    for cmd in (
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
        ["mpv", "--really-quiet", filepath],
    ):
        try:
            subprocess.run(cmd, check=True)
            return
        except Exception:
            pass
    print(f"(Couldn't play audio automatically. Audio saved at: {filepath})")


async def cleanup_engine(engine):
    try:
        if hasattr(engine, "client") and engine.client is not None:
            await engine.client.close()
    except Exception:
        pass
    try:
        await engine.close()
    except Exception:
        pass


async def interactive_chat(tutor: LogTruncationKani, quit_cmds: set[str],
                           voice_in: bool, voice_out: bool):
    print("\n=== Terminal Tutor Chat ===")
    if voice_in:
        print("Voice input ON: press ENTER to start/stop recording; type 'q' + ENTER inside recorder to quit.")
    print("Type 'q' to quit.\n")

    while True:
        if voice_in:
            try:
                user_in = await speech_to_text()  # uses your recorder + transcription
            except ExitProgram:
                print("Exiting.")
                break
        else:
            user_in = input("You: ").strip()

        if user_in.lower() in quit_cmds:
            print("Exiting.")
            break

        reply = await tutor.chat_round(user_in)
        text = (reply.content or "").replace("<|im_end|>", "").strip()
        print(f"Tutor: {text}\n")

        if voice_out and text:
            # generate audio file using your util
            text_to_speech(text)  # returns filename, but it always writes ROOT_TEMP_DATA + FILENAME_AUDIO_OUTPUT
            play_wav(f"{ROOT_TEMP_DATA}{FILENAME_AUDIO_OUTPUT}")


def main():
    args = parse_args()

    t_engine_id = args.TE
    bot_level = args.LEVEL
    t_model_id = args.TM if args.TM else engine_to_default_model[t_engine_id]
    lamda = args.LAMBDA

    shared_model, shared_tokenizer = None, None
    if need_shared_hf_model(t_engine_id):
        shared_model, shared_tokenizer = setup_shared_hf_model(t_model_id)

    tutor_engine = setup_tutor_engine(
        engine_id=t_engine_id,
        model_id=t_model_id,
        bot_level=bot_level,
        lamda=lamda,
        shared_model=shared_model,
        shared_tokenizer=shared_tokenizer,
    )

    tutor_system_prompt = setup_tutor_system_prompt(args.P, bot_level)

    tutor = LogTruncationKani(
        engine=tutor_engine,
        system_prompt=tutor_system_prompt,
        desired_response_tokens=256,
    )

    quit_cmds = set(c.lower() for c in args.quit)

    try:
        asyncio.run(interactive_chat(
            tutor=tutor,
            quit_cmds=quit_cmds,
            voice_in=args.voice_in,
            voice_out=args.voice_out,
        ))
    finally:
        try:
            asyncio.run(cleanup_engine(tutor_engine))
        except RuntimeError:
            pass
        free_cuda()


if __name__ == "__main__":
    main()