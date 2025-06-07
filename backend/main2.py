import os
import time
import uuid
import json
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union


from google import genai


# uvicorn main:app --reload

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# --- FastAPI Application ---
app = FastAPI(
    title="Gemini OpenAI-Compatible API",
    description="An API that mimics the OpenAI Chat Completions endpoint using Google Gemini.",
    version="1.0.0",
)


# --- Helper Functions ---

def convert_openai_to_gemini(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """
    Converts a list of OpenAI-formatted messages to Gemini-formatted content.
    - The "system" role is prepended to the first "user" message.
    - The "assistant" role is mapped to Gemini's "model" role.
    """
    gemini_contents = []
    system_prompt = ""
    for message in messages:
        # Handle system prompt: Gemini prefers it at the start of the conversation.
        if message.role == "system":
            system_prompt = message.content
            continue
        # Prepend system prompt to the first user message if it exists
        if message.role == "user" and system_prompt:
            message.content = f"{system_prompt}\n\n{message.content}"
            system_prompt = "" # Clear after prepending

        # Map "assistant" to "model" for Gemini
        role = "model" if message.role == "assistant" else message.role
        gemini_contents.append({"role": role, "parts": [{"text": message.content}]})

    return gemini_contents

# --- API Endpoint ---
# In main.py, replace the create_chat_completion function with this one.
# The rest of the file stays the same.

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, authorization: str = Header(None)):
    """
    Handles chat completion requests, compatible with the OpenAI API format.
    This version is optimized for strict clients like the 'com.aallam.openai' Kotlin library.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization header is missing.")
    
    try:
        # Extract the API key from the Authorization header (Bearer token)
        api_key = authorization.split(" ")[1]
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error configuring Gemini client: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure Gemini client.")
    try:
        model_name = request.model

        chat = client.chats.create(model=model_name)

        gemini_contents = convert_openai_to_gemini(request.messages)

        generation_config = genai.types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens
        )
        
        if request.stream:
            # --- Streaming Response ---
            async def stream_generator():
                chat_id = f"chatcmpl-{uuid.uuid4()}"
                created_time = int(time.time())
                
                # Use a flag to send the 'role' only on the first chunk
                is_first_chunk = True
                
                try:
                    stream = model.generate_content(
                        contents=gemini_contents,
                        generation_config=generation_config,
                        stream=True
                    )

                    for chunk in stream:
                        # Skip empty chunks
                        if not chunk.text:
                            continue

                        # --- FIX #1: Delta format based on position in stream ---
                        if is_first_chunk:
                            delta = {"role": "assistant", "content": chunk.text}
                            is_first_chunk = False
                        else:
                            delta = {"content": chunk.text}
                        
                        # --- FIX #2: Map Gemini's finish reason ---
                        finish_reason = None
                        try:
                            # The finish_reason is on the candidate object in the Gemini response
                            reason = chunk.candidates[0].finish_reason
                            # A simple mapping from Gemini's reason enum to OpenAI's string
                            if reason.name.lower() in ["stop", "max_tokens", "length"]:
                                finish_reason = reason.name.lower()
                        except (IndexError, AttributeError):
                            pass # No finish reason in this chunk

                        # --- FIX #3: Every data payload is a valid JSON object ---
                        response_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"

                except Exception as e:
                    # If an error occurs, log it on the server and close the stream.
                    # Do not send a malformed error chunk to the client.
                    print(f"An error occurred during streaming: {e}")
                
                # --- FIX #4: Do NOT send 'data: [DONE]' ---
                # The stream is automatically closed when the generator function ends.
                # The Kotlin client correctly handles this as the end of the stream.
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # --- Non-Streaming Response (no changes needed here) ---
            response = model.generate_content(
                contents=gemini_contents,
                generation_config=generation_config
            )
            openai_response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": { "role": "assistant", "content": response.text },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                },
            }
            return JSONResponse(content=openai_response)

    except Exception as e:
        print(f"An error occurred: {e}")
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{request.model}' not found. Please check the model name."
            )
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
def read_root():
    return {"message": "Welcome to the Gemini OpenAI-Compatible API. Use the /v1/chat/completions endpoint."}



# curl http://127.0.0.1:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "gemini-2.0-flash",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a helpful and concise assistant."
#       },
#       {
#         "role": "user",
#         "content": "Explain how a car engine works in one sentence."
#       }
#     ]
#   }'



# curl http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
#     "model": "gemini-2.0-flash",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a helpful and concise assistant."
#       },
#       {
#         "role": "user",
#         "content": "Explain how a car engine works in one sentence."
#       }
#     ],"stream": true
#   }'