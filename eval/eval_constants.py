import os

CONVERSATION_LOGS_FOLDER = os.path.join("eval", "auto_eval_logs")
PERPLEXITY_MODEL_ID = "CohereForAI/aya-expanse-8b"

STANZA_RESOURCES_DIR =  "/anon/stanza_resources"

class EvalType:
    BASELINE = "baseline"
    PROMPTING = "prompting"
    OVERGEN = "overgen"
    FUDGE = "fudge"