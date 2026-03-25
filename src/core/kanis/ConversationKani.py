from kani import Kani, AIParam, ai_function
from typing import Annotated
from src.core.utils.misc_utils import *
from src.core.utils.profile_utils import *

class ConversationKani(Kani):

    def __init__(self, user_profile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_profile = user_profile

    @ai_function()
    def store_user_interest(
        self,
        interest: Annotated[str, AIParam(desc="The interest or hobby the user mentioned, in a sentence.")],
    ):
        """Store the user's interest, hobby, or something they like as a simple sentence when they mention it in a conversation.
        """
        print("Storing user interest: ", interest)
        return interest
    
    @ai_function()
    def store_user_personal_info(
        self,
        info: Annotated[str, AIParam(desc="Personal info of the user that you think helps you know the user better. This could include their school year, which country they live in, etc.")],
    ):
        """Store the user's personal information as a simple sentence when they mention it in a conversation.
        """
        print("Storing personal info: ", info)
        return info