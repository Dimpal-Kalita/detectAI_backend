import os
import Levenshtein
from groq import Groq

from fastapi import HTTPException

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def generate_rewritten_text(input_text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Rewrite the following text (Don't Write anything extra, the answer Should Contain only the rewritten text): \n\n"+input_text,
                }
            ],
            model="llama3-8b-8192",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return chat_completion.choices[0].message.content


def calculate_edit_distance(original_text, rewritten_text):
    # Calculate edit distance using Levenshtein distance algorithm
    edit_distance = Levenshtein.distance(original_text, rewritten_text)

    return edit_distance


def Edit_distance(input_text):

    rewritten_text = generate_rewritten_text(input_text)
    edit_distance = calculate_edit_distance(input_text, rewritten_text)
    # print(f"Original Text: {input_text}")
    # print(f"Rewritten Text: {rewritten_text}")
    # print(f"Edit Distance: {edit_distance}")
    return edit_distance

# text1="The quick brown fox jumps over the lazy dog"
# text2="The quick brown fox jumps over the lazy dog"
# print(Edit_distance(text1))
# print(Edit_distance(text2))