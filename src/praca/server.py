from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import uuid

from .exam_system import ExamSystem

app = FastAPI(title="System automatyzujący egzaminację")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

exam_system = ExamSystem()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    created: int
    model: str
    choices: List[Dict[str, Any]]

def create_chunk_id() -> str:
    """Generate unique chunk ID"""
    return f"chatcmpl-{uuid.uuid4().hex}"

def create_timestamp() -> int:
    """Get current timestamp"""
    return int(datetime.now().timestamp())

def create_chunk(chunk_id: str, timestamp: int, model: str, content: str = "", finish_reason: Optional[str] = None) -> Dict[str, Any]:
    """Create standardized chunk response"""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason
        }]
    }

@app.get("/v1/models")
async def list_models():
    """List available models - OpenAI compatible"""
    return {
        "object": "list",
        "data": [{
            "id": "Egzaminator",
            "object": "model",
            "created": create_timestamp(),
            "owned_by": "praca"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions - ExaminerAgent talks to user, EvaluatorAgent works behind scenes"""
    try:
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user" and not msg.content.startswith("### Task:"):
                user_message = msg.content
                break
        
        result = exam_system.process_message(user_message)
        response_content = result["response"]
        
        if request.stream:
            return StreamingResponse(_generate_stream(response_content, request.model), media_type="text/plain")
        else:
            return ChatCompletionResponse(
                id=create_chunk_id(),
                created=create_timestamp(),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop"
                }]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_stream(content: str, model: str):
    """Generate streaming response"""
    try:
        chunk_id = create_chunk_id()
        timestamp = create_timestamp()
        
        yield f"data: {json.dumps(create_chunk(chunk_id, timestamp, model, content))}\n\n"
        
        yield f"data: {json.dumps(create_chunk(chunk_id, timestamp, model, finish_reason='stop'))}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = create_chunk(create_chunk_id(), create_timestamp(), model, f"Error: {str(e)}", "stop")
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "two-agent-system"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Exam System API",
        "version": "1.0.0",
        "architecture": "chat-based exam system",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "health": "/health"
        }
    }
