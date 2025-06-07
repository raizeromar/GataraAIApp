from google import genai

client = genai.Client(api_key="")
# chat = client.chats.create(model="gemini-2.0-flash")
# print(chat)
# response = chat.send_message("I have 2 dogs in my house.")
# print(response.text)

# response = chat.send_message("How many paws are in my house?")
# print(response.text)




# response = chat.send_message_stream("I have bringed one other dog in my house.") 
# for chunk in response:
#     print(chunk.text, end="")



# for message in chat.get_history():
#     print(f'role - {message.role}',end=": ")
#     print(message.parts[0].text)    

#docs: https://ai.google.dev/api/caching#Content
# contents = [
#     {
#     "parts": [
#         {
#         "text": "what is ai? 2 words only"
#         }
#     ],
#     "role": "user"
#     },
#     {
#     "parts": [
#         {
#         "text": "intelligence worker"
#         }
#     ],
#     "role": "assistant"
#     },
#     {
#     "parts": [
#         {
#         "text": "what are your system instructions?"
#         }
#     ],
#     "role": "user"
#     },

# ]

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=contents,
# )
# print(response.text)





# openai compataple:
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain to me how AI works, in two words only"
        }
    ]
)

print(response)
