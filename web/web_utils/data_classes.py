from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ConversationEndpointInfo:
    track_usage: bool = False
    language: str = ""
    target_level: str = ""
    grammar_dict: Dict[str, Any] = field(default_factory=dict)
    vocab_dict: Dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    desired_response_tokens: int = 0


@dataclass
class LearningEndpointInfo:
    grammar_to_teach: List[dict] = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class GlobalSessionData:
    users: Dict[str, dict] = field(default_factory=dict)
    username: str = ""
    firstname: str = ""
    profile_path: str = ""
    profile: Dict[str, Any] = field(default_factory=dict)

    language: str = ""
    all_levels: List[str] = field(default_factory=list)

    engine: Any = None
    tutor: Any = None

    # learning mode
    instruction_language: str = "" 
    target_level: str = ""
    learning_schema: str = ""
    grammar_to_teach: List[dict] = field(default_factory=list)

    # conversation mode
    backup_language: str = ""
    current_level: str = ""
    user_interests: List[str] = field(default_factory=list)
    user_info: List[str] = field(default_factory=list)

    def reset(self):
        self.users.clear()
        self.username = ""
        self.firstname = ""
        self.profile_path = ""
        self.profile.clear()

        self.language = ""
        self.all_levels.clear()

        self.engine = None
        self.tutor = None

        # Reset learning mode
        self.instruction_language = "" 
        self.target_level = ""
        self.learning_schema = ""
        self.grammar_to_teach.clear()

        # Reset conversation mode
        self.backup_language = ""
        self.current_level = ""
        self.user_interests.clear()
        self.user_info.clear()

