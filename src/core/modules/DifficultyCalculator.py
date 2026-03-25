from src.core.utils.language_utils import get_level_difficulty_score

class DifficultyCalculator:
    
    def at_target_level(self, above, below):
        cnt_above, cnt_below = 0, 0
        for level, tokens in above.items():
            cnt_above += len(tokens)
        for level, tokens in below.items():
            cnt_below += len(tokens)
        cnt_total = cnt_above + cnt_below
        if cnt_total == 0:
            return False
        if cnt_above / cnt_total <= 0.1:
            return True
        return False
    
    def calc_difficulty_score(self, above, below, undetected_cnt):
        cnt_above, cnt_below = 0, 0
        for level, tokens in above.items(): 
            cnt_above += len(tokens)
        for level, tokens in below.items():
            cnt_below += len(tokens)
        cnt_total = cnt_above + cnt_below
        if cnt_total == 0:
            return None
        return cnt_above / (cnt_total + undetected_cnt)

    def calc_difficulty_score_weighted(self, language, above, below):
        score = 0
        for level, tokens in above.items():
            score += get_level_difficulty_score(language, level) * len(tokens)
        for level, tokens in below.items():
            score -= len(tokens) / 5
        return score

    def grammar_difficulty(self, sentence):
        pass
    def vocab_difficulty(self, sentence):
        pass