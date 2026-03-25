import regex
import konoha
from typing import List
from transformers import AutoTokenizer
from src.core.core_constants import JPN, TOKENIZER_TYPE_PLAIN, TOKENIZER_TYPE_HF
from src.training.model_constants import ROOT_HF_CACHE_PERSONAL

class TokenTokenizer():
    def __init__(self, language: str, tokenizer_id: str = None) -> None:
        self.language = language.lower()
        self.tokenizer_type = None
        if not tokenizer_id:
            if self.language == JPN:
                self.tokenizer = self.init_sudachipy_tokenizer()
        else:
            self.tokenizer = self.init_hf_tokenizer(tokenizer_id)
    
    def init_sudachipy_tokenizer(self):
        self.tokenizer_type = TOKENIZER_TYPE_PLAIN
        # Use konoha's integration for Sudachi with mode "C"
        konoha_tokenizers = {"Sudachi": {"mode": "C"}}
        return konoha.WordTokenizer("Sudachi", **konoha_tokenizers["Sudachi"])
    
    def init_hf_tokenizer(self, tokenizer_id: str) -> AutoTokenizer:
        self.tokenizer_type = TOKENIZER_TYPE_HF
        return AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=True,
            cache_dir=ROOT_HF_CACHE_PERSONAL
        )
    
    def sentence_to_tokens(self, sentence: str, base_form: bool = False, filter_punctuation: bool = False) -> List[str]:
        tokens = []
        if self.language == JPN:
            if self.tokenizer_type == TOKENIZER_TYPE_PLAIN:
                # For plain tokenization, use konoha's tokenizer. It returns token objects.
                tokens = [token.base_form if base_form else token.surface() for token in self.tokenizer.tokenize(sentence)]
            elif self.tokenizer_type == TOKENIZER_TYPE_HF:
                # raw_tokens = self.tokenizer.tokenize(sentence)
                # # for token in raw_tokens:
                # #     tokens.append(self.tokenizer.convert_tokens_to_string([token]))
                # fixed_sentence = self.tokenizer.convert_tokens_to_string(raw_tokens)
                # # Then, re-tokenize the fixed sentence to get a clean token list:
                tokens = self.tokenizer.tokenize(sentence)
        if filter_punctuation:
            tokens = [token for token in tokens if not regex.match(r'^\p{Punctuation}+$', token)]
        return tokens


# import sudachipy
# import regex
# import konoha
# from typing import List
# from sudachipy.tokenizer import Tokenizer
# from transformers import AutoTokenizer

# from LanguageTutor_v1.core.core_constants import JPN, \
#     TOKENIZER_TYPE_PLAIN, TOKENIZER_TYPE_HF
# from LanguageTutor_v1.core.models.model_constants import ROOT_HF_CACHE_PERSONAL


# class TokenTokenizer():
#     def __init__(self, language: str, tokenizer_id: str = None) -> None:
#         self.language = language.lower()
#         self.tokenizer_type = None
#         if not tokenizer_id:
#             if self.language == JPN:
#                 self.tokenizer = self.init_sudachipy_tokenizer()
#         else:
#             self.tokenizer = self.init_hf_tokenizer(tokenizer_id)

#     def init_sudachipy_tokenizer(self) -> Tokenizer:
#         self.tokenizer_type = TOKENIZER_TYPE_PLAIN
#         return sudachipy.Dictionary().create()
    
#     def init_hf_tokenizer(self, tokenizer_id: str) -> AutoTokenizer:
#         self.tokenizer_type = TOKENIZER_TYPE_HF
#         return AutoTokenizer.from_pretrained(
#             tokenizer_id, 
#             trust_remote_code=True,
#             cache_dir=ROOT_HF_CACHE_PERSONAL
#         )

#     def sentence_to_tokens(self, 
#                           sentence: str, 
#                           base_form: bool = False,
#                           filter_punctuation: bool = False) \
#         -> List[str]:
#         tokens = []
#         if self.language == JPN:

#             if self.tokenizer_type == TOKENIZER_TYPE_PLAIN:
#                 tokens = [m.base_form if base_form else m.surface() for m in 
#                           self.tokenizer.tokenize(sentence)]
                
#             elif self.tokenizer_type == TOKENIZER_TYPE_HF:
#                 tokens = self.tokenizer.tokenize(sentence)
#                 tokens = self.tokenizer.convert_tokens_to_string(tokens)
#         if filter_punctuation:
#             tokens = [token for token in tokens 
#                       if not regex.match(r'^\p{Punctuation}+$', token)]
            
#         return tokens
