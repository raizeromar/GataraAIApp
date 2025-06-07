import os
import time
import uuid
import json
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI

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
        client = OpenAI(
            api_key= api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    except Exception as e:
        print(f"Error configuring Gemini client: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure Gemini client.")
    try:
        model_name = request.model


        messages= request.messages

        temperature=request.temperature
        top_p=request.top_p
        max_output_tokens=request.max_tokens

        
        if request.stream:
            # --- Streaming Response ---
            async def stream_generator():
                chat_id = f"chatcmpl-{uuid.uuid4()}"
                created_time = int(time.time())
                
                
                try:
                    stream = client.chat.completions.create(
                        model= model_name,
                        messages= messages,
                        temperature= temperature,
                        top_p= top_p,
                        max_tokens= max_output_tokens,
                        stream=True,
                    )

                    for chunk in stream:
                        if not chunk.choices[0].delta.content:
                            continue # Skip empty chunks

                        # --- FIX #3: Every data payload is a valid JSON object ---
                        response_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": chunk.choices[0].delta.model_dump(),
                                    "finish_reason": chunk.choices[0].finish_reason,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"

                    yield "data: [DONE]\n\n"    

                except Exception as e:
                    # If an error occurs, log it on the server and close the stream.
                    # Do not send a malformed error chunk to the client.
                    print(f"An error occurred during streaming: {e}")
                
                # --- FIX #4: Do NOT send 'data: [DONE]' ---
                # The stream is automatically closed when the generator function ends.
                # The Kotlin client correctly handles this as the end of the stream.
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            response = client.chat.completions.create(
                model= model_name,
                messages= messages,
                temperature= temperature,
                top_p= top_p,
                max_tokens= max_output_tokens,
            )
            return response

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

