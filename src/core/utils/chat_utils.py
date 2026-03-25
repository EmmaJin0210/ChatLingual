from kani import ChatMessage, ChatRole
from openai import OpenAI
from src.core.utils.language_utils import retrieve_shots

###### imports for typing purposes ######
from kani import Kani
#########################################

def summarize_chat_history(tutor: Kani, 
                           language: str,
                           num_msgs_to_keep: int = 4) \
    -> Kani:
    
    history_to_summarize = format_chat_history_for_summary(
        tutor.chat_history[:len(tutor.chat_history) - num_msgs_to_keep])
    chat_summary = summarize_rounds_history(history = history_to_summarize, 
                                            language = language)
    new_chat_history = [ChatMessage(role = ChatRole.ASSISTANT, content = chat_summary)] + \
                        tutor.chat_history[len(tutor.chat_history) - num_msgs_to_keep:]
    tutor.chat_history = new_chat_history

    return tutor

def format_chat_history_for_summary(history):
    to_return = []
    valid_roles = [ChatRole.ASSISTANT, ChatRole.USER]
    for msg in history:
        if msg.role not in valid_roles or not msg.content:
            continue
        flattened = "user: " if msg.role == ChatRole.USER else "tutor: "
        flattened += msg.content
        to_return.append(flattened)
    return to_return


def summarize_rounds_history(history, language):
    shots = retrieve_shots(language, "summarize_rounds_history")
    sys_prompt = f"""You are a helpful topic summarizer for a language learning chatbot.
The user is having a conversation with the language tutor. Given the list of
conversations they are having, your job is to summarize the conversation into a few very short, beginner-level sentences,
in the language used for each sentence.
Focus on what topics are being covered in the conversation.
"""
    messages = [{"role": "system", "content":  sys_prompt}]
    messages += shots
    messages += [{"role": "user", "content": str(history)}]
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return completion.choices[0].message.content

def summarize_user_interests(interests_list, language):
    sys_prompt = f"""You are a helpful topic summarizer for a language learning chatbot. 
Given a list of the user's interests, your job is to organize and summarize these interests into
concise sentences in their original language.
"""
    shots = retrieve_shots(language, "summarize_user_interests")
    messages = [{"role": "system", "content":  sys_prompt}]
    messages += shots
    messages += [{"role": "user", "content": str(interests_list)}]
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    return completion.choices[0].message.content

def summarize_user_personal_info(info_list, language):
    shots = retrieve_shots(language, "summarize_user_personal_info")
    sys_prompt = f"""You are a helpful topic summarizer for a language learning chatbot. 
Given a list of the user's personal information, your job is to organize and summarize these information into
concise sentences in their original language.
"""
    messages = [{"role": "system", "content":  sys_prompt}]
    messages += shots
    messages += [{"role": "user", "content":  str(info_list)}]
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    return completion.choices[0].message.content