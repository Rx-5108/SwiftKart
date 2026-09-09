import asyncio
import json
import re
import time
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from crew_setup import (
    crew,
    detect_prompt_injection,
    groundedness_check,
    mask_pii,
    parse_crew_output_to_schema,
)
from rag_core import fixed_collection, model

app = FastAPI(title="SwiftKart Customer Support API", version="1.0.0")

MAX_QUERY_TOKENS = 500
GROUNDING_THRESHOLD = 0.30
ORDER_ID_PATTERN = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
SAFE_REFUSAL = "I don't have enough information in the provided policy documents to answer that question."
SECURITY_REFUSAL = "I cannot process this request due to a security concern."
BUDGET_REFUSAL = f"Request exceeds the maximum input budget of {MAX_QUERY_TOKENS} tokens."


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    escalation_score: float | None = None


class AddDocumentRequest(BaseModel):
    text: str = Field(min_length=1)
    topic: str = Field(min_length=1)


class AddDocumentResponse(BaseModel):
    success: bool
    message: str


def estimate_tokens(text: str) -> int:
    """Roughly estimate tokens using the ~4 characters/token heuristic."""
    return max(1, (len(text) + 3) // 4)


def is_within_budget(text: str) -> bool:
    """Return whether a request is within the runtime input-token cap."""
    return estimate_tokens(text) <= MAX_QUERY_TOKENS


def is_in_scope(query: str) -> bool:
    """Apply the grounding guardrail, while allowing order-ID queries to reach the order tool."""
    if groundedness_check(query, fixed_collection, threshold=GROUNDING_THRESHOLD):
        return True
    return bool(ORDER_ID_PATTERN.search(query))


def safe_refusal(trace_id: str, endpoint: str, query: str, start_time: float, status: str) -> AskResponse:
    duration = (time.perf_counter() - start_time) * 1000
    log_request(trace_id, endpoint, query, duration, status)
    return AskResponse(answer=SAFE_REFUSAL, escalation_score=None)


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    if not is_within_budget(request.query):
        duration = (time.perf_counter() - start_time) * 1000
        log_request(trace_id, "/ask", request.query, duration, "blocked_budget")
        return AskResponse(answer=BUDGET_REFUSAL)

    if detect_prompt_injection(request.query):
        duration = (time.perf_counter() - start_time) * 1000
        log_request(trace_id, "/ask", request.query, duration, "blocked_injection")
        return AskResponse(answer=SECURITY_REFUSAL)

    # Grounding is checked before any CrewAI/tool execution.
    if not is_in_scope(request.query):
        return safe_refusal(trace_id, "/ask", request.query, start_time, "blocked_grounding")

    masked_query = mask_pii(request.query)
    result = await crew.kickoff_async(inputs={"query": masked_query})
    validated = parse_crew_output_to_schema(str(result), request.query)

    duration = (time.perf_counter() - start_time) * 1000
    log_request(trace_id, "/ask", request.query, duration, "success")
    return AskResponse(answer=validated.answer, escalation_score=validated.escalation_score)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest):
    """Add one embedded document to the persistent fixed-size Chroma collection."""
    try:
        new_id = f"doc_api_{fixed_collection.count() + 1}"
        embedding = model.encode([request.text], convert_to_numpy=True).tolist()
        fixed_collection.upsert(
            ids=[new_id],
            documents=[request.text],
            metadatas=[{"parent_doc": new_id, "topic": request.topic}],
            embeddings=embedding,
        )
        return AddDocumentResponse(success=True, message=f"Document {new_id} added successfully.")
    except Exception as exc:
        return AddDocumentResponse(success=False, message=str(exc))


@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """Receive complete messages and return complete JSON responses over WebSocket.

    This endpoint is request/response, not token streaming. The project therefore
    does not claim WebSocket token streaming unless a streaming model is configured.
    """
    await websocket.accept()
    session_id = f"ws_session_{id(websocket)}"

    try:
        while True:
            data = await websocket.receive_text()

            if not is_within_budget(data):
                await websocket.send_json({"answer": BUDGET_REFUSAL, "escalation_score": None})
                continue

            if detect_prompt_injection(data):
                await websocket.send_json({"answer": SECURITY_REFUSAL, "escalation_score": None})
                continue

            # Apply the same grounding guardrail as /ask before agent/tool execution.
            if not is_in_scope(data):
                await websocket.send_json({"answer": SAFE_REFUSAL, "escalation_score": None})
                continue

            masked_data = mask_pii(data)
            result = await crew.kickoff_async(inputs={"query": masked_data})
            validated = parse_crew_output_to_schema(str(result), data)
            # Application-level chunk streaming: CrewAI still computes the final answer once,
            # then the WebSocket delivers that answer incrementally to the client.
            chunk_size = 40
            for start in range(0, len(validated.answer), chunk_size):
                await websocket.send_json({"type": "chunk", "content": validated.answer[start:start + chunk_size]})
                await asyncio.sleep(0)
            await websocket.send_json(
                {
                    "type": "done",
                    "answer": validated.answer,
                    "escalation_score": validated.escalation_score,
                }
            )

    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected. Server continues running.")


def log_request(trace_id: str, endpoint: str, query: str, duration_ms: float, status: str):
    """Append a privacy-aware request record to the JSONL audit log."""
    log_entry = {
        "trace_id": trace_id,
        "timestamp": time.time(),
        "endpoint": endpoint,
        "query": mask_pii(query),
        "duration_ms": round(duration_ms, 2),
        "status": status,
    }
    with open("requests.jsonl", "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")
