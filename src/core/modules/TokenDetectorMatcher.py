import JapaneseTokenizer
import konoha
import subprocess
import regex as re
from collections import Counter
from itertools import chain


MAX_BYTES = 49149          # Sudachi hard limit
CHUNK_SIZE = 3000          # fallback chars per chunk

jp_only_re = re.compile(r'^[\u3005\u303B\u309D\u309E\u30FC\u3040-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]+$')

def is_punctuation(token):
    return re.match(r'^\p{P}+$', token)

class TokenDetectorMatcher:
    def __init__(self, word_dict, grammar_dict, language = "japanese"):
        self.language = language
        self.level_to_words = word_dict
        self.level_to_grammars = grammar_dict

    def _split_long_text(self, text, max_bytes=MAX_BYTES, chunk_size=CHUNK_SIZE):
        """Yield chunks that are safely below the Sudachi byte limit."""
        if len(text.encode("utf-8")) <= max_bytes:
            yield text
            return

        sents = re.split(r'(?<=[。．！？\n])', text)
        buf = ""
        for sent in sents:
            if len((buf + sent).encode("utf-8")) > max_bytes:
                if buf:
                    yield buf
                    buf = ""
            buf += sent
            # still over for a single huge sentence → hard split
            while len(buf.encode("utf-8")) > max_bytes:
                yield buf[:chunk_size]
                buf = buf[chunk_size:]
        if buf:
            yield buf

    def tokenize(self, sentence, tokenizer = 'Sudachi', sudachi_mode = 'C', 
                 strip = False, strip_punc = True, jpn_only = False):
        konoha_tokenizers = {
            'MeCab': {},
            'Janome': {},
            'Sudachi': {"mode": sudachi_mode}
        }
        if tokenizer in konoha_tokenizers:
            self.tokenizer = konoha.WordTokenizer(tokenizer, **konoha_tokenizers[tokenizer])
            # ---- NEW: chunk if necessary -----------------------
            chunks = self._split_long_text(sentence)
            token_lists = ( [tok.base_form for tok in self.tokenizer.tokenize(ch)]
                            for ch in chunks )
            tokens = list(chain.from_iterable(token_lists))
            # ----------------------------------------------------
            # tokens = [token.base_form for token in self.tokenizer.tokenize(sentence)]
            if strip:
                tokens = [tok.strip() for tok in tokens if tok.strip()]
            if jpn_only:
                tokens = [t for t in tokens if jp_only_re.fullmatch(t)]
                
        elif tokenizer == 'juman':
            self.tokenizer = JapaneseTokenizer.JumanppWrapper()
            tokens = self.tokenizer.tokenize(sentence).convert_list_object()
        elif tokenizer == 'kytea':
            result = subprocess.check_output(f"echo '{sentence}' | kytea", shell=True, text=True)
            tokens = [token.split('/')[0] for token in result.strip().split()]
        if strip_punc:
            tokens = [token for token in tokens if token and not is_punctuation(token)]
        return tokens

    def detect_tokens_at_level(self, tokens, level, scope=['v']):
        to_return = []
        for token in tokens:
            if 'v' in scope and token in self.level_to_words[level]:
                to_return.append(token)
            if 'g' in scope and token in self.level_to_grammars[level]:
                to_return.append(token)
        return to_return


    def detect_tokens_at_levels(self, tokens, levels, scope=['v']):  # return a level : [tokens] map
        level_to_detected = {}
        tokens = Counter(tokens)
        # print(levels)
        for level in levels:
            level_to_detected[level] = []
            detected = self.detect_tokens_at_level(tokens, level, scope)
            for word in detected:
                for _ in range(tokens[word]):
                    level_to_detected[level].append(word)
                del tokens[word]
        undetected = list(tokens.keys())
        # for level, detected_set in level_to_detected_unique.items():
        #     level_to_detected_unique[level] = list(detected_set)
        return level_to_detected, undetected

    # def fuzzy_match(self, to_match, levels=[], scope=['v', 'g']):
    #     matched = set()
    #     if not levels:
    #         levels = self.levels
    #     if levels != self.levels:
    #         self.load_db_as_dicts(levels=levels)
    #     for token in to_match:
    #         variations = [token + "る"]
    #         for level in levels:
    #             for variation in variations:
    #                 if 'v' in scope and variation in self.level_to_words[level]:
    #                     matched.add(token)
    #                 if 'g' in scope and variation in self.level_to_grammars[level]:
    #                     matched.add(token)
    #     return matched





def main():
    levels = ["n2", "n3", "n4", "n5"]
    n5_ex1 = "これは本です。"
    n5_ex2 = "一緒に映画を見ませんか。"
    n5_ex3 = "日本へ行きたいです。"
    detector = TokenDetectorMatcher(language="Japanese", levels=levels)
    tokens = detector.tokenize(n5_ex1)
    print(f"Tokens: {tokens}")
    at_level = detector.detect_tokens_at_level(tokens)
    print(f"At Levels {levels}: {at_level}")

if __name__ == "__main__":
    main()