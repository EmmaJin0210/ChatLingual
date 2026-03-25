import os
import argparse
from dotenv import load_dotenv
from src.core.engines.ControlledGenFudgeEngine import ControlledGenFudgeEngine
from src.core.engines.SharedHFModelEngine import SharedHFModelEngine
from src.core.engines.HFOvergenerationEngine import HFOvergenerationEngine
import datetime as dt
from web.app_constants import ROOT_TEMP_DATA


class EngineTypes:
    FUDGE_ENGINE         = "fudge"
    HFOVERGEN_ENGINE     = "hfovergen"
    SHAREDHFMODEL_ENGINE = "sharedhfmodel"

from src.training.model_constants import (
    MODEL_ID_HF_DEFAULT,
    MODEL_ID_LM,
    MODEL_ID_LM_MEDIUM,
    MODEL_ID_LM_SMALL,
    MODEL_ID_LM_TINY,
    LAMBDA as DEFAULT_LAMBDA
)

engine_to_default_model = {
    EngineTypes.FUDGE_ENGINE:           MODEL_ID_LM_TINY, # MODEL_ID_HF_DEFAULT,
    EngineTypes.HFOVERGEN_ENGINE:       MODEL_ID_LM_TINY,
    EngineTypes.SHAREDHFMODEL_ENGINE:   MODEL_ID_LM_TINY,
}

MODE_CONFIG = {
    "A": dict(prompt="baseline",
              engine=EngineTypes.SHAREDHFMODEL_ENGINE,
              fudge_lambda=DEFAULT_LAMBDA),
    "B": dict(prompt="detailed",
              engine=EngineTypes.SHAREDHFMODEL_ENGINE,
              fudge_lambda=DEFAULT_LAMBDA),
    "C": dict(prompt="baseline",
              engine=EngineTypes.HFOVERGEN_ENGINE,
              fudge_lambda=DEFAULT_LAMBDA),
    "D": dict(prompt="baseline",
              engine=EngineTypes.FUDGE_ENGINE,
              fudge_lambda=0.8),
}

def apply_mode(mode: str) -> None:
    cfg = MODE_CONFIG.get(mode)
    if not cfg:
        raise ValueError("bad mode")
    g_session_data.current_mode     = mode 
    g_session_data.prompt_version   = cfg["prompt"]
    g_session_data.tutor_engine_id  = cfg["engine"]
    g_session_data.fudge_lambda     = cfg["fudge_lambda"]
    # reset chat state so the next page load starts clean
    g_session_data.engine = None
    g_session_data.tutor  = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Language Tutor Server + Eval Options",
        add_help=False
    )
    parser.add_argument("--host",  default="0.0.0.0", help="Uvicorn host")
    parser.add_argument("--port",  type=int, default=8000,   help="Uvicorn port")
    parser.add_argument("--reload",action="store_true",      help="Enable reload")

    args, _ = parser.parse_known_args()
    return args

def need_shared_hf_model(t_eid, t_mid, *args):
    return t_eid in [
        EngineTypes.FUDGE_ENGINE,
        EngineTypes.SHAREDHFMODEL_ENGINE,
        EngineTypes.HFOVERGEN_ENGINE
    ]

def setup_shared_hf_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    shared_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="balanced_low_0",
        torch_dtype=torch.float16
    )
    shared_tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    print("[app.py] loaded shared hf model")
    if (shared_tokenizer.pad_token_id is None
        or shared_tokenizer.pad_token_id == shared_tokenizer.eos_token_id):
        shared_tokenizer.pad_token = shared_tokenizer.eos_token
    return shared_model, shared_tokenizer


def prep_hfovergen_engine(model_id, target_level, target_language, all_levels, 
                          shared_model, shared_tokenizer):
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
                  target_language, all_levels, bot_level = None, 
                  lamda = DEFAULT_LAMBDA, for_student = False):
    SAMPLE_KW = {}
    if for_student:
        if engine_id == EngineTypes.SHAREDHFMODEL_ENGINE:
            SAMPLE_KW = dict(temperature = 0.7, top_p = 1.0, do_sample = True)
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
        
        case EngineTypes.HFOVERGEN_ENGINE:
            return prep_hfovergen_engine(
                model_id = model_id,
                target_level = bot_level,
                target_language = target_language,
                all_levels = all_levels,
                shared_model = shared_model,
                shared_tokenizer = shared_tokenizer
            )
    
        

def setup_tutor_engine(target_language, all_levels, engine_id, model_id, 
                       shared_model, shared_tokenizer, bot_level, lamda):
    return set_up_engine(
        engine_id = engine_id,
        model_id = model_id,
        shared_model = shared_model,
        shared_tokenizer = shared_tokenizer,
        target_language = target_language,
        all_levels = all_levels,
        bot_level = bot_level,
        lamda = lamda
    )
args = parse_args()

load_dotenv()



from fastapi import FastAPI, Form, File, Request, Response, status, UploadFile, Body, \
    WebSocket, WebSocketDisconnect, \
    HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
from src.core.kanis.LearningKani import LearningKani
from src.core.kanis.LogTruncationKani import LogTruncationKani
from src.core.utils.language_utils import get_all_levels
from src.core.utils.chat_utils import summarize_chat_history
from src.core.utils.engine_utils import clean_up
from src.core.utils.misc_utils import *


from web.web_utils.data_classes import GlobalSessionData
from web.web_utils.app_utils import *

from src.core.core_constants import CHAT_ENGINE, CHAT_MODEL, LEARNING_ENGINE, LEARNING_MODEL
from web.app_constants import FILENAME_USERS_DB, \
    ROOT_TEMP_DATA, ROOT_STATIC, ROOT_TEMPLATES, \
    MOUNT_TEMP_DATA, MOUNT_STATIC, \
    APP_IP, APP_PORT

my_openai_key = os.getenv("OPENAI_API_KEY")

# initialize app
app = FastAPI()
app.mount(MOUNT_TEMP_DATA, StaticFiles(directory=ROOT_TEMP_DATA), name="temp-data")
app.mount(MOUNT_STATIC,    StaticFiles(directory=ROOT_STATIC),   name="static")
templates = Jinja2Templates(directory=ROOT_TEMPLATES)

# preload model
sm, st = setup_shared_hf_model(MODEL_ID_LM_TINY)
app.state.shared_model     = sm
app.state.shared_tokenizer = st

# initialize global session data
g_session_data = GlobalSessionData()
g_session_data.shared_model     = app.state.shared_model
g_session_data.shared_tokenizer = app.state.shared_tokenizer
g_session_data.tutor_model_id   = MODEL_ID_LM_TINY   # single HF checkpoint
apply_mode("A")    

g_session_data.users = read_json_to_dict(
    os.path.join(ROOT_USER_PROFILES, FILENAME_USERS_DB)
)


@app.get("/")
def get_root(request: Request) -> RedirectResponse:
    if g_session_data.username:
        return RedirectResponse(url = "/home", status_code = status.HTTP_303_SEE_OTHER)
    else:
        return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.get("/home", response_class=HTMLResponse)
async def get_home(request: Request) -> Response:
    if g_session_data.username:
        language = g_session_data.language
        return templates.TemplateResponse("home.html", {"request": request, "selected": language})
    else:
        return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.get("/signup")
async def get_signup(request: Request) -> Response:
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup")
def sign_up(request: Request, username: str = Form(...), first_name: str = Form(...)) \
    -> RedirectResponse:
    global g_session_data
    g_session_data = handle_user_signup(session_data = g_session_data,
                                        username = username,
                                        firstname = first_name)
    return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.get("/login")
async def get_login(request: Request) -> Response:
    if g_session_data.username:
        return RedirectResponse(url="/home", status_code=status.HTTP_303_SEE_OTHER)
    else:
        return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login(request: Request, username: str = Form(...)) -> Response:
    global g_session_data
    if username in g_session_data.users:
        g_session_data = handle_user_login(session_data = g_session_data, username = username)
        return RedirectResponse(url = "/home", status_code = status.HTTP_303_SEE_OTHER)
    else:
        content = "<p>Invalid username. Try again or <a href='/signup'>sign up</a>.</p>"
        return HTMLResponse(content = content, status_code = 400)
    

@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    global g_session_data
    g_session_data.reset()
    return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.post("/set-language")
def set_language(request: Request, language: str = Form(...)) -> RedirectResponse:
    global g_session_data
    if g_session_data.username:
        print(f"/set-language: setting language to {language}")
        g_session_data.language = language.lower()
        g_session_data.all_levels = get_all_levels(language)
        return RedirectResponse(url = "/home", status_code = status.HTTP_303_SEE_OTHER)
    else:
        return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.get("/choose-instruction-lang")
async def get_instruction_lang(request: Request) -> Response:
    if g_session_data.username:
        return templates.TemplateResponse("instruction-lang.html", 
                                          {"request": request, "levels": g_session_data.all_levels})
    else:
        return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.post("/set-instruction-lang")
def set_instruction_lang(request: Request, instructionlang: str = Form(...), 
                         targetlevel: str = Form(...)) -> RedirectResponse:
    global g_session_data
    g_session_data.instruction_language = instructionlang.lower()
    g_session_data.target_level = targetlevel.lower()
    return RedirectResponse(url = "/learning", status_code = status.HTTP_303_SEE_OTHER)


@app.get("/choose-backup-lang")
async def get_backup_lang(request: Request) -> Response:
    if g_session_data.username:
        return templates.TemplateResponse("backup-lang.html", 
                                          {"request": request, "levels": g_session_data.all_levels})
    else:
        return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.post("/set-backup-lang")
def set_backup_lang(request: Request, backuplang: str = Form(...), currentlevel: str = Form(...)) \
    -> RedirectResponse:
    global g_session_data
    g_session_data.backup_language = backuplang.lower()
    g_session_data.current_level = currentlevel.lower()
    return RedirectResponse(url = "/conversation", status_code = status.HTTP_303_SEE_OTHER)


@app.get("/learning")
async def get_learning_mode(request: Request) -> Response:
    if g_session_data.username:
        return templates.TemplateResponse("learning.html", 
                                          {"request": request, "language": g_session_data.language})
    else:
        return RedirectResponse(url = "/login", status_code = status.HTTP_303_SEE_OTHER)


@app.post("/save-learning")
def save_learning_mode(request: Request) -> JSONResponse:
    res = save_learning_info_to_profile(g_session_data)
    if res["success"]:
        return JSONResponse(status_code=200, content={"message": res["message"]})
    else:
        return JSONResponse(status_code=500, content={"message": res["message"]})


@app.websocket("/ws-learning")
async def websocket_endpoint_learning(websocket: WebSocket) -> None:
    global g_session_data

    await websocket.accept()
    g_session_data.learning_schema = "random-sample"### dummy

    info = get_learning_endpoint_info(g_session_data)

    engine = get_engine(info = info, engine_id = LEARNING_ENGINE, model_id = LEARNING_MODEL)
    tutor = LearningKani(user_profile = g_session_data.profile, 
                         engine = engine,
                         system_prompt = info.system_prompt)

    g_session_data.grammar_to_teach = info.grammar_to_teach
    g_session_data.engine = engine
    g_session_data.tutor = tutor

    try:
        while True:
            try:
                await websocket.receive()
            except WebSocketDisconnect:
                print("/ws-learning: webSocket disconnected")
                break
            user_input = handle_websocket_input(websocket)
            handle_learning_round(websocket = websocket, 
                                  tutor = tutor,
                                  user_input = user_input)

    except WebSocketDisconnect:
        print("/ws-learning: client disconnected")
    finally:
        await clean_up(engine)


@app.get("/conversation")
async def get_conversation_mode(request: Request) -> Response:
    if not g_session_data.username:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        "conversation.html",
        {
            "request": request,
            "mode": g_session_data.current_mode     # ← now always present
        }
    )


@app.post("/set-mode")
async def set_mode(mode: str = Body(..., embed=True)):
    try:
        apply_mode(mode)
        return {"status": "ok"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Unknown mode")


@app.post("/save-conversation")
async def save_conversation_mode(request: Request) -> JSONResponse:
    global g_session_data
    tutor = g_session_data.tutor

    # 1) Build alternating lines of student/tutor text
    lines = []
    for msg in tutor.chat_history:
        role = msg.role.name.lower()
        prefix = "Student:" if role in ("user", "student") else "Tutor:"
        text = getattr(msg, "text", None) or getattr(msg, "content", "")
        lines.append(f"{prefix} {text}")

    # 2) Write to a .txt file
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"chat_history_{ts}.txt"
    txt_path = os.path.join(ROOT_TEMP_DATA, txt_filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    # 3) Update user profile as before
    res = save_chat_info_to_profile(g_session_data)
    if res["success"]:
        return JSONResponse(
            status_code=200,
            content={
                "message": res["message"],
                "transcript_txt": txt_filename
            }
        )
    else:
        return JSONResponse(status_code=500, content={"message": res["message"]})


@app.websocket("/ws-conversation")
async def websocket_endpoint_conversation(websocket: WebSocket) -> None:
    global g_session_data

    await websocket.accept()


    info = get_conversation_endpoint_info(g_session_data)
    engine = setup_tutor_engine(
        target_language = g_session_data.language,
        all_levels = g_session_data.all_levels,
        engine_id=g_session_data.tutor_engine_id,
        model_id=g_session_data.tutor_model_id,
        shared_model      = app.state.shared_model,        # ← use pre‑loaded
        shared_tokenizer  = app.state.shared_tokenizer,
        bot_level=g_session_data.current_level,
        lamda=g_session_data.fudge_lambda
    )

    tutor = LogTruncationKani(
        engine=engine,
        system_prompt=info.system_prompt,
        desired_response_tokens=info.desired_response_tokens
    )

    g_session_data.engine = engine
    g_session_data.tutor = tutor

    rounds = 0
    g_session_data.user_interests = [g_session_data.profile["interests"]]
    g_session_data.user_info = [g_session_data.profile["personal-info"]]
    try:
        while True:
            if rounds % 16 == 0 and rounds != 0:
                tutor = summarize_chat_history(tutor = tutor, language = info.language)
            
            res = await handle_websocket_input(websocket)
            if not res["success"]:
                return JSONResponse(status_code = 500, 
                                    content = {"error": res["error"]})
            user_input = res["user_input"]
            
            g_session_data = await handle_chat_round(websocket = websocket, 
                                                     session_data = g_session_data,
                                                     tutor = tutor,
                                                     user_input = user_input)
            rounds += 1

    except WebSocketDisconnect:
        print("/ws-conversation: Client disconnected")
    finally:
        await clean_up(engine)
        

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)) -> JSONResponse:
    file_location = f"{ROOT_TEMP_DATA}{file.filename}"
    print(f"Uploaded audio file to {file_location}")
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return JSONResponse(content = {"status": "success"})
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
print("=== reached end of web/app.py import ===")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host   = args.host,
        port   = args.port,
        # reload = args.reload,
    )