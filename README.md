# SwiftKart Agentic Customer Support System

A capstone-ready customer-support system combining **RAG, CrewAI agents, tool use, session memory, guardrails, evaluation, caching, governance, a FastAPI API, WebSocket responses, and an AutoGen review stage**.

## 1. Project goals

The system answers SwiftKart policy questions from a controlled knowledge base and can look up synthetic order information when a valid order ID is present. Security and governance controls are applied before agent/tool execution.

### Main capabilities

- Policy retrieval with ChromaDB + Sentence Transformers.
- Fixed-size and sentence-based chunking for retrieval comparison.
- Three CrewAI roles:
  - **Retrieval Specialist** — policy search.
  - **Order Status Specialist** — order lookup only when an `ORD-xxxx` ID is present.
  - **Response Composer** — final customer-facing response.
- In-memory multi-turn conversation memory.
- Guardrails:
  - prompt-injection detection,
  - PII masking,
  - groundedness/out-of-scope refusal,
  - runtime input-token budget.
- Deterministic mock LLMs so the demonstrations can run without an external LLM API key.
- Heuristic mock LLM-as-judge evaluation.
- In-memory response caching.
- FastAPI `/ask` and `/add-document` endpoints.
- WebSocket `/ws` with application-level response chunks and a final `done` message.
- AutoGen policy-compliance review with structured `ReviewVerdict` output.

## 2. Architecture

```text
                    +----------------------+
                    |   FastAPI / Client   |
                    |  /ask or /ws         |
                    +----------+-----------+
                               |
                     Budget + Injection
                         + Grounding
                               |
                    +----------v-----------+
                    |      CrewAI Crew     |
                    +----------+-----------+
                               |
             +-----------------+------------------+
             |                                    |
     +-------v--------+                    +------v-------+
     | Retrieval Agent|                    | Lookup Agent |
     | RAG Search Tool|                    | Order Tool   |
     +-------+--------+                    +------+-------+
             |                                    |
     +-------v------------------------------------v-------+
     |                 Response Composer                  |
     +------------------------+----------------------------+
                              |
                       Structured response
                              |
                     +--------v---------+
                     | API / WebSocket |
                     +------------------+

Policy documents -> chunking -> SentenceTransformer -> ChromaDB

Review stage (separate demonstration):
Draft + retrieved context -> AutoGen compliance reviewer -> Final editor -> ReviewVerdict
```

## 3. File structure

| File | Responsibility |
|---|---|
| `rag_core.py` | Policy corpus, chunking, embeddings, ChromaDB, retrieval and grounded generation |
| `dataset.py` | Deterministic synthetic order dataset |
| `tools.py` | Order-status lookup tool |
| `crew_setup.py` | CrewAI agents/tasks, mock LLM, memory, output parsing and guardrails |
| `api.py` | FastAPI REST API and WebSocket interface |
| `evaluation.py` | 15-query heuristic evaluation |
| `governance.py` | Least-autonomy demonstration, risk classification and token-budget logic |
| `caching.py` | In-memory response-cache demonstration |
| `review_stage.py` | AutoGen compliance-review and structured editor demonstration |
| `requests.jsonl` | Privacy-aware request audit log |
| `requirements.txt` | Python dependencies |

## 4. Setup

Recommended: Python 3.10+ in a virtual environment.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first import of `rag_core.py` downloads/loads the `all-MiniLM-L6-v2` Sentence Transformer model as required by Sentence Transformers. `requirements.txt` pins the direct runtime dependencies to a version snapshot checked against PyPI on 2026-09-09; this environment does not contain those packages, so the pinned set has not been live-installed here.

## 5. Run the demonstrations

Build/load the RAG collections and run the retrieval demonstrations:

```bash
python rag_core.py
```

Run the CrewAI, memory, structured-output and guardrail demonstrations:

```bash
python crew_setup.py
```

Run evaluation:

```bash
python evaluation.py
```

Run governance:

```bash
python governance.py
```

Run caching:

```bash
python caching.py
```

Run the AutoGen review stage:

```bash
python review_stage.py
```

## 6. Run the API

```bash
uvicorn api:app --reload
```

Open the interactive API documentation at the local FastAPI docs endpoint shown by Uvicorn (normally `/docs`).

### REST: `/ask`

Example request body:

```json
{"query":"What is the return window for electronics?"}
```

The endpoint performs, in order:

1. Input-budget check.
2. Prompt-injection check.
3. Groundedness check against the policy collection.
4. Order-ID allowance for legitimate order-status requests.
5. PII masking.
6. CrewAI execution.
7. Structured response parsing.

Out-of-scope questions receive a safe refusal instead of being sent to CrewAI.

### REST: `/add-document`

Accepts:

```json
{"text":"New policy text", "topic":"policy_topic"}
```

and embeds it into the persistent **fixed-size Chroma collection only**. The sentence-based collection is intentionally a separate retrieval-comparison dataset and is not modified by this endpoint.

### WebSocket: `/ws`

The WebSocket applies the same budget, injection and groundedness guardrails as `/ask`. It sends the final answer as application-level chunks followed by a `done` message.

This is **chunked WebSocket delivery**, not provider-level token streaming: the current mock/CrewAI execution computes the answer first and the API then delivers it incrementally.

## 7. Groundedness and out-of-scope behavior

The RAG layer converts Chroma cosine distance into similarity using `1 - distance`. The configured default grounding threshold is `0.30`.

If the best policy retrieval is below the threshold, the system returns:

> I don't have enough information in the provided policy documents to answer that question.

Order-status requests containing a valid `ORD-xxxx` identifier are allowed through the API grounding gate because their source of truth is the order-status tool rather than the policy corpus.

## 8. Memory demonstration

The memory test deliberately uses a dependent follow-up. This memory demonstration is separate from the FastAPI `/ask` and `/ws` endpoints; those endpoints are intentionally request-scoped and do not claim persistent conversational state.

```text
Turn 1: What is the return window for electronics?
Turn 2: What about defective ones?
```

The second turn is run in the same session and receives the previous conversation history before CrewAI processes the new query. Memory is **in-memory only** and is lost when the process restarts.

## 9. Tool-selection demonstration

The order-status specialist follows a least-autonomy rule:

- Query without an order ID → no order-status tool action.
- Query containing `ORD-xxxx` → the order-status tool may be invoked with that exact normalized ID.

The mock LLM uses a strict `ORD-\d{4}` pattern and never invents an order ID.

## 10. Guardrails and governance

### PII masking

Common 10-digit phone numbers and card-ending fragments are masked before agent processing and before audit logging.

### Prompt injection

Common direct injection patterns such as "ignore previous instructions" are blocked before CrewAI execution.

### Token budget

The configured input budget is **500 estimated tokens** using a lightweight character-based estimate. The limit is **actually enforced at runtime by both `/ask` and `/ws`** before CrewAI execution.

The governance script demonstrates the same budget function separately; it is not the enforcement point itself.

### Risk classification

The project demonstration classifies the system as **Medium risk** because it handles customer-support policy and synthetic order-status information while avoiding medical, hiring, and high-stakes financial decision-making.

## 11. Caching

`caching.py` demonstrates response caching using the normalized query plus collection identity, `top_k`, and threshold as the cache key. The FastAPI `/ask` endpoint does **not** use this cache; caching is a standalone demonstration of the optimization pattern.

The cache is **in-memory only**. It does not survive a process restart and is not shared across multiple application processes.

## 12. Evaluation

`evaluation.py` evaluates 15 representative queries, including in-scope policy questions and out-of-scope questions such as:

- `What is the capital of France?`
- `How do I bake a chocolate cake?`

The evaluator reports four dimensions:

| Metric | Meaning |
|---|---|
| Accuracy | Deterministic proxy based on retrieval similarity |
| Grounding | Retrieval similarity proxy |
| Completeness | Answer-length proxy capped at 1.0 |
| Safety | Prompt-injection detection proxy |

**Important:** these are mock/heuristic scores, not human evaluation and not scores from a real LLM judge. The script always evaluates the same 15 queries, but the numeric results depend on the installed embedding/model stack and persisted Chroma state. No fabricated numeric results are recorded here. Run `python evaluation.py` on the target machine and copy its printed averages into the report only after that run has completed.

## 13. Retrieval evaluation

`rag_core.py` also compares fixed-size and sentence-based retrieval on five labeled queries and reports document-level precision and recall.

The evaluation uses the labeled `correct_doc` field and reports whether the expected parent document appears in the retrieved set.

## 14. AutoGen review stage

`review_stage.py` demonstrates two cases:

1. **Compliant draft** → reviewer finds no unsupported claim and the final editor returns `ReviewVerdict(approved=True, ...)`.
2. **Policy-violating draft** → reviewer flags unsupported claims such as a guaranteed refund/free upgrade, and the final editor returns `ReviewVerdict(approved=False, ...)` with a revised answer.

The final verdict is parsed and validated with Pydantic.

## 15. Reproducibility notes

- The order dataset uses `random.seed(8)`, so the generated synthetic records are deterministic.
- ChromaDB persists data under `./chroma_db`.
- The embedding model is `all-MiniLM-L6-v2`.
- Mock LLMs are deterministic and do not require a provider API key.
- Runtime API logs are appended to `requests.jsonl`.

## 16. Verification checklist

After installing dependencies, run:

```bash
python tests_static.py
python -m compileall -q *.py
python dataset.py
python caching.py
python governance.py
python evaluation.py
python review_stage.py
python crew_setup.py
python rag_core.py
```

For the API, start Uvicorn and verify:

- grounded policy query → normal answer;
- out-of-scope query → safe refusal;
- prompt-injection query → security refusal;
- oversized query → budget refusal;
- order query with `ORD-xxxx` → order lookup path;
- WebSocket → chunk messages followed by `done`.

## 17. Known scope limitations

- The included LLM implementations are deterministic mocks for demonstration and testing; they are not production-grade generative models.
- The evaluation judge is heuristic rather than a real LLM judge.
- The WebSocket provides application-level chunk streaming after final generation, not true model-token streaming.
- Session memory is process-local and disappears on restart.
- The synthetic order dataset is not a real database.
- `/add-document` updates only the fixed-size collection by design; sentence-based retrieval remains a comparison-only collection.
- The API endpoints do not use the standalone response cache or the standalone session-memory demonstration.
