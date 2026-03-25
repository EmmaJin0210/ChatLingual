DIRNAME_GRAMMAR = "grammar_lists"
DIRNAME_VOCAB = "vocab_lists"
DIRNAME_LEVEL_DESCS = "level_descs"
DIRNAME_FEWSHOTS = "few_shots"
DIRNAME_LEVEL_GUIDES = "level_guidelines"
DIRNAME_LEVEL_EX = "level_examples"


LIST_ALL_LEVELS = {
    "japanese" : ["n5", "n4", "n3", "n2", "n1"]
}

MAP_LEVEL_TO_DESC_WORD = {
    "japanese": {
        "n5" : "absolute beginner",
        "n4" : "beginner",
        "n3" : "early intermediate",
        "n2" : "intermediate",
        "n1" : "advanced"
    }
}

MAP_LEVEL_TO_NUM_DESIRED_TOKENS = {
    "japanese": {
        "n5" : 10,
        "n4" : 15,
        "n3" : 20,
        "n2" : 30,
        "n1" : 50
    }
}

MAP_LEVEL_TO_DIFFICULTY_SCORE = {
    "japanese": {
        "n5" : 1,
        "n4" : 3,
        "n3" : 8,
        "n2" : 15,
        "n1" : 30
    }
}