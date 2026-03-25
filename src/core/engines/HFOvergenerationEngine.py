import threading
from kani.models import ChatMessage, ChatRole
from kani.ai_function import AIFunction
from src.core.engines.SharedHFModelEngine import SharedHFModelEngine
from src.core.modules.TokenDetectorMatcher import TokenDetectorMatcher
from src.core.modules.DifficultyCalculator import DifficultyCalculator
from src.core.utils.language_utils import get_levels_below_inclusive, get_levels_above_exclusive
from src.core.engines.engine_constants import DEFAULT_TOKENIZER_KWARGS, \
    DEFAULT_MODEL_LOAD_KWARGS


class HFOvergenerationEngine(SharedHFModelEngine):
    def __init__(self, language, target_level, vocab_dict, grammar_dict, model_id,
                 model=None, tokenizer=None,
                 tokenizer_kwargs=DEFAULT_TOKENIZER_KWARGS, model_load_kwargs=DEFAULT_MODEL_LOAD_KWARGS,
                 device="cuda", prompt_pipeline=None, max_context_size=None,
                 token_reserve=0, **kwargs):
        super().__init__(
            model_id=model_id,
            model=model, 
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            model_load_kwargs=model_load_kwargs,
            device=device,
            prompt_pipeline=prompt_pipeline,
            max_context_size=max_context_size,
            token_reserve=token_reserve,
            **kwargs
        )
        self.language = language
        self.target_level = target_level
        self.vocab_dict = vocab_dict
        self.grammar_dict = grammar_dict

        # Compute levels for difficulty estimation
        self.levels_below = get_levels_below_inclusive(self.language, self.target_level)
        self.levels_above = get_levels_above_exclusive(self.language, self.target_level)

    async def predict(self, messages: list[ChatMessage],
                      functions: list[AIFunction] | None = None,
                      n: int = 5, **kwargs):
        responses = await self.generate_responses(messages, functions, n=n, **kwargs)
        return self.pick_response(responses)

    async def generate_responses(self, messages: list[ChatMessage],
                                 functions: list[AIFunction] | None = None,
                                 n: int = 5, **kwargs):
        prompt = self.build_prompt(messages, functions)
        input_toks, input_len, hyperparams = self._get_generate_args(prompt, **kwargs)
        hyperparams["num_return_sequences"] = n

        output = self.model.generate(input_toks, **hyperparams)

        responses = []
        for i in range(n):
            decoded = self._processor_or_tokenizer.decode(output[i][input_len:], skip_special_tokens=True).strip()
            responses.append(decoded)
        return responses

    def pick_response(self, responses: list[str]):
        token_matcher = TokenDetectorMatcher(self.vocab_dict, self.grammar_dict, language=self.language)
        difficulty_calculator = DifficultyCalculator()
        scored_responses = []
        threads = []

        def evaluate_response(response):
            tokens = token_matcher.tokenize(response)
            levels_below_to_matched, undetected = token_matcher.detect_tokens_at_levels(tokens, self.levels_below)
            levels_above_to_matched, _ = token_matcher.detect_tokens_at_levels(undetected, self.levels_above)
            difficulty_score = difficulty_calculator.calc_difficulty_score(levels_above_to_matched, levels_below_to_matched)
            score = difficulty_score if difficulty_score else float('inf')
            scored_responses.append({
                "response": response,
                "score": score,
                "token_count": len(tokens)
            })

        # Evaluate each response in parallel threads.
        for response in responses:
            thread = threading.Thread(target=evaluate_response, args=(response,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if not scored_responses:
            raise ValueError("No valid responses were scored.")

        scored_responses.sort(key=lambda x: (x["score"], x["token_count"]))
        print("SCORED OVERGEN RESPONSES: ", scored_responses)
        best_response = scored_responses[0]["response"]

        best_message = ChatMessage(role=ChatRole.ASSISTANT, content=best_response)
        return type("Completion", (), {"message": best_message, "prompt_tokens": None, "completion_tokens": None})
