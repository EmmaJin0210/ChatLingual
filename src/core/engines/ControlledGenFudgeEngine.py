import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from kani.models import ChatMessage
from kani.engines import Completion
from kani.ai_function import AIFunction

from src.core.engines.SharedHFModelEngine import SharedHFModelEngine
from src.core.engines.FudgeLogitsProcessor import FudgeLogitsProcessor
from src.training.model_constants import MODEL_ID_PREDICTOR, LAMBDA



class ControlledGenFudgeEngine(SharedHFModelEngine):
    def __init__(self, model, tokenizer, model_id: str, target_difficulty: str, lamda: float = LAMBDA, **kwargs):
        super().__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )
        self.target_difficulty = target_difficulty
        self.difficulty_map = {"n1": 0, "n2": 1, "n3": 2, "n4": 3, "n5": 4}
        # Load the predictor
        self.bp_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID_PREDICTOR)
        self.bp_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_PREDICTOR)
        print("!! predictor loaded")
        # Assign predictor to first GPU
        self.bp_device = torch.device("cuda:0") if torch.cuda.device_count() > 1 else torch.device("cuda")
        self.bp_model.to(self.bp_device)
        self.lamda = lamda

    async def predict(self, messages: list, functions: list[AIFunction] = None,
                      top_k: int = 50, max_new_tokens: int = 256, **kwargs) -> Completion:

        prompt = self.build_prompt(messages, functions)

        # HuggingEngine.build_prompt usually returns BatchEncoding/dict with input_ids
        if isinstance(prompt, str):
            input_kwargs = self._processor_or_tokenizer(
                text=prompt, add_special_tokens=False, return_tensors="pt"
            )
        else:
            # assume dict-like (BatchEncoding/BatchFeature)
            input_kwargs = prompt

        input_ids = input_kwargs["input_ids"]
        attention_mask = input_kwargs.get("attention_mask", torch.ones_like(input_ids))

        # Put inputs on the same device as the LM embeddings (important for sharded models)
        embed_dev = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(embed_dev)
        attention_mask = attention_mask.to(embed_dev)

        prompt_token_len = input_ids.size(1)
        
        # Create the custom logits processor.
        fudge_processor = FudgeLogitsProcessor(
            bp_model = self.bp_model,
            bp_tokenizer = self.bp_tokenizer,
            bp_device = self.bp_device,
            target_idx = self.difficulty_map[self.target_difficulty],
            lamda = self.lamda,
            base_tokenizer = self._processor_or_tokenizer,
            prompt_token_len = prompt_token_len,
            top_p = 0.7,
            top_k = top_k
        )
        
        logits_processor = [fudge_processor]
        
        
        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            logits_processor = logits_processor,
            max_new_tokens = max_new_tokens,
            do_sample = True
        )
        # Decode
        new_tokens = outputs[0][prompt_token_len:]
        final_text = self._processor_or_tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        final_message = ChatMessage.assistant(final_text)
        return Completion(message = final_message, prompt_tokens = None, completion_tokens = None)
