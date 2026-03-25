import os

APP_IP = "127.0.0.1"
APP_PORT = 8080
SERVER_ADDR = f"{APP_IP}:{APP_PORT}"

MOUNT_TEMP_DATA = "/temp-data"
MOUNT_STATIC = "/static"

ROOT_TEMP_DATA = os.path.join("web", "web_utils", "tempdata")
ROOT_STATIC = os.path.join("web", "web_utils", "web_ui", "static")
ROOT_USER_PROFILES = os.path.join("web", "user_profiles")
ROOT_TEMPLATES = os.path.join("web", "web_utils", "web_ui", "templates")
ROOT_STATIC_DB = os.path.abspath("data")

FILENAME_USERS_DB = "all_users.json"
FILENAME_AUDIO_INPUT = "input.wav"
FILENAME_AUDIO_OUTPUT = "output.wav"
FILENAME_PROFILE_TEMPLATE = "<username>.json"

MSG_PROGRESS_SAVE_SUCCESS = "Progress saved! Your learning and conversation sessions hereon will \
    take into account what you learned today.\n\nRedirecting to homepage..."
MSG_PROGRESS_SAVE_FAIL = "Failed to save progress..."
MSG_TRANSCRIBE_AUDIO_FAIL = "Failed to transcribe audio..."