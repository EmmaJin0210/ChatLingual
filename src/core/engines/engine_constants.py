import torch

OPENAI_MODELS_CONTEXT_SIZES = [
    ("gpt-3.5-turbo-instruct", 4096),
    ("gpt-3.5-turbo-0613", 4096),
    ("gpt-3.5-turbo", 16385),
    ("o1-", 128000),
    ("gpt-4o", 128000),
    ("gpt-4-turbo", 128000),
    ("gpt-4-32k", 32768),
    ("gpt-4", 8192),
    ("ft:gpt-3.5-turbo", 16385),
    ("ft:gpt-4-32k", 32768),
    ("ft:gpt-4", 8192),
    ("babbage-002", 16384),
    ("davinci-002", 16384),
    ("", 2048),
]

DEFAULT_TOKENIZER_KWARGS = {"trust_remote_code": True}

DEFAULT_MODEL_LOAD_KWARGS = {
    "trust_remote_code": True,
    "device_map": "auto",
    "torch_dtype": torch.float16,
}