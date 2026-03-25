import re
import torch
from transformers import LogitsProcessor, TopPLogitsWarper, TopKLogitsWarper

class FudgeLogitsProcessor(LogitsProcessor):
    def __init__(self, bp_model, bp_tokenizer, bp_device, target_idx, lamda, base_tokenizer, 
                 prompt_token_len: int, top_p: float = 1.0, top_k: int = 50):
        
        self.bp_model = bp_model
        self.bp_tokenizer = bp_tokenizer
        self.bp_device = bp_device
        self.target_idx = target_idx
        self.lamda = lamda
        self.base_tokenizer = base_tokenizer
        self.prompt_token_len = prompt_token_len
        self.top_k = top_k
        self._pre_top_p = TopPLogitsWarper(top_p = top_p, min_tokens_to_keep = 1)
        self._pre_top_k = TopKLogitsWarper(top_k = top_k, min_tokens_to_keep = 1)
        self._splitter = re.compile(r'。+')

    def __call__(self, input_ids, scores):
        if self.lamda == 0:
            return scores
        # Top-p Max-k first
        scores = self._pre_top_p(input_ids, scores)
        scores = self._pre_top_k(input_ids, scores)

        topk_logits, topk_indices = torch.topk(scores, self.top_k, dim = -1)

        decoded = self.base_tokenizer.decode(
            input_ids[0][self.prompt_token_len:], skip_special_tokens = True
        )

        bp_texts = []
        pieces = self._splitter.split(decoded)
        for tok_id in topk_indices[0]:
            tok_str = self.base_tokenizer.decode([tok_id.item()], skip_special_tokens = True)
            bp_text = pieces[-1] + tok_str
            bp_texts.append(bp_text)

        # BP forward pass
        bp_in = self.bp_tokenizer(
            bp_texts, return_tensors = "pt", padding = True, truncation = True
        )
        # move everything to the predictor's device
        bp_in = { k: v.to(self.bp_device) for k, v in bp_in.items() }
        # batch process predictor inputs for logits
        with torch.no_grad():
            bp_logits = self.bp_model(**bp_in).logits # [top_k, 5]
        # only get logits of the target level
        bp_target_logits = bp_logits[:, self.target_idx] # [top_k]]
        bp_target_logits = bp_target_logits.to(topk_logits.device)
        # FUDGE
        new_logits = (1 - self.lamda) * topk_logits[0] + self.lamda * bp_target_logits

        new_scores = scores.clone()
        new_scores[0, topk_indices[0]] = new_logits
        return new_scores