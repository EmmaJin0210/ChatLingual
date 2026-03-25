import torch
from kani.engines.huggingface import HuggingEngine
from kani.engines.huggingface.chat_template_pipeline import ChatTemplatePromptPipeline
from src.core.engines.engine_constants import DEFAULT_TOKENIZER_KWARGS, \
    DEFAULT_MODEL_LOAD_KWARGS
from collections import UserDict
from transformers import BatchEncoding
try:
    from transformers import BatchFeature
except ImportError:
    BatchFeature = dict  # fallback

class SharedHFModelEngine(HuggingEngine):
    def __init__(self, model_id, 
                 model = None, tokenizer = None,
                 tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS,
                 model_load_kwargs = DEFAULT_MODEL_LOAD_KWARGS,
                 device = "cuda",
                 prompt_pipeline = None, 
                 max_context_size = None,
                 token_reserve = 0,
                 **kwargs):
        # If model and tokenizer are already provided, bypass parent's loading
        if model is not None and tokenizer is not None:
            self.model = model
            self._processor_or_tokenizer = tokenizer
            self.model_id = model_id
            self.device = str(self.model.get_input_embeddings().weight.device) 

            # Patch pad_token_id
            if self._processor_or_tokenizer.pad_token_id is None or self._processor_or_tokenizer.pad_token_id == self._processor_or_tokenizer.eos_token_id:
                self._processor_or_tokenizer.pad_token = self._processor_or_tokenizer.eos_token

            
            if prompt_pipeline is None:
                prompt_pipeline = ChatTemplatePromptPipeline(self._processor_or_tokenizer)
            self.pipeline = prompt_pipeline

            if max_context_size is None:
                max_context_size = getattr(
                    self.model.config, "model_max_len",
                    getattr(self.model.config, "max_position_embeddings", None)
                )
                if max_context_size is None:
                    raise ValueError(
                        "Could not infer model's max context size from config. Pass `max_context_size`."
                    )
            self.max_context_size = max_context_size

            if self.model.device.type != self.device:
                self.model.to(self.device)

            # If token_reserve isn't given
            if token_reserve == 0 and self.pipeline:
                prompt = self.pipeline.execute([], for_measurement=True)
                if isinstance(prompt, torch.Tensor):
                    token_reserve = len(prompt[0])
                else:
                    tokenized = self._processor_or_tokenizer.encode(prompt, add_special_tokens=False)
                    token_reserve = len(tokenized)
            self._token_reserve = token_reserve
            self.hyperparams = kwargs
        else:
            super().__init__(
                model_id = model_id,
                device = device,
                tokenizer_kwargs = tokenizer_kwargs,
                model_load_kwargs = model_load_kwargs,
                **kwargs
            )

            # Patch pad_token_id
            if self._processor_or_tokenizer.pad_token_id is None or self._processor_or_tokenizer.pad_token_id == self._processor_or_tokenizer.eos_token_id:
                self._processor_or_tokenizer.pad_token = self._processor_or_tokenizer.eos_token



    def _get_generate_args(self, prompt: str | torch.Tensor, **hyperparams):
        if isinstance(prompt, str):
            tokenized = self._processor_or_tokenizer(text=prompt, add_special_tokens=False, return_tensors="pt")
            input_kwargs = tokenized
            input_len = input_kwargs["input_ids"].shape[1]

        elif isinstance(prompt, torch.Tensor):
            input_kwargs = {"input_ids": prompt}
            input_len = prompt.shape[1]

        elif isinstance(prompt, (dict, UserDict, BatchEncoding)) or (
            BatchFeature is not None and isinstance(prompt, BatchFeature)
        ):
            input_kwargs = prompt
            input_len = input_kwargs["input_ids"].shape[1]

        else:
            raise TypeError(
                f"build_prompt returned unsupported type: {type(prompt)}. "
                "Expected str, Tensor, or dict/BatchEncoding."
            )

        if hasattr(input_kwargs, "to"):
            input_kwargs = input_kwargs.to(self.device)
            if BatchFeature is not None and isinstance(input_kwargs, BatchFeature):
                input_kwargs = input_kwargs.to(self.model.dtype)
        else:
            # plain dict
            for k, v in input_kwargs.items():
                if torch.is_tensor(v) and v.device.type != self.device:
                    input_kwargs[k] = v.to(self.device)

        # attention mask: only add if not present
        if "attention_mask" not in input_kwargs:
            input_ids = input_kwargs["input_ids"]
            input_kwargs["attention_mask"] = torch.ones_like(input_ids)

        # merge hyperparams (engine defaults + call-time)
        merged = {**getattr(self, "hyperparams", {}), **hyperparams}
        merged.setdefault("max_length", self.max_context_size)

        try:
            emb_dev = self.model.get_input_embeddings().weight.device
        except Exception:
            emb_dev = None

        ids_dev = input_kwargs["input_ids"].device
        print(f"[DEBUG] engine.device={self.device}  input_ids={ids_dev}  emb={emb_dev}  has_map={hasattr(self.model,'hf_device_map')}")

        return input_kwargs, input_len, merged
    
    def _infer_input_device(self) -> str:
        # For accelerate-sharded models, determine where embeddings live
        dm = getattr(self.model, "hf_device_map", None)
        if isinstance(dm, dict) and len(dm) > 0:
            # common names for token embedding module
            for k in ("model.embed_tokens", "transformer.wte", "model.wte", "embed_tokens"):
                if k in dm:
                    return str(dm[k])
            # fallback: first device in map
            return str(next(iter(dm.values())))
        # non-sharded
        return str(self.model.device)

