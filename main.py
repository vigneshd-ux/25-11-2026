import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from query_pipeline import default_pipeline, QueryResult
from retrieve import retrieve_with_metadata, retrieve_documents

load_dotenv()

app = FastAPI(
    title="HRMS RAG Assistant",
    description="Advanced RAG Query Pipeline with token budgeting, top-k retrieval, and multi-tiered fallbacks.",
    version="2.0.0",
)


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    token_budget: Optional[int] = None
    tenant_id: Optional[str] = "tenant_001"


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "HRMS RAG API",
        "pipeline_stages": [
            "1. Embed Query",
            "2. Retrieve Top-K with Metadata",
            "3. Enforce Context Token Budget",
            "4. Build Grounded Prompt with Citations",
            "5. Call LLM (OpenRouter)",
            "6. Multi-tiered Fallback Strategies (Zero-context, Model, and Offline Extractive)",
        ],
    }


@app.post("/retrieve")
def retrieve_endpoint(request: ChatRequest):
    """Retrieve relevant policy chunks with metadata, similarity scores, and token budget stats."""
    return retrieve_with_metadata(
        question=request.question,
        tenant_id=request.tenant_id or "tenant_001",
        n_results=request.top_k or 3,
        token_budget=request.token_budget,
    )


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Execute full RAG pipeline:
    Embed -> Retrieve Top-K -> Token Budget -> Prompt Build -> LLM Call -> Fallbacks.
    """
    try:
        result = default_pipeline.execute(
            question=request.question,
            top_k=request.top_k,
            token_budget=request.token_budget,
            tenant_id=request.tenant_id or "tenant_001",
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution error: {str(e)}",
        )


def generate_answer(question: str):
    """Helper function for direct programmatic usage."""
    result = default_pipeline.execute(question)
    return result.answer, result.context


if __name__ == "__main__":
    question = "How many casual leaves are available?"
    print("=" * 60)
    print("HRMS RAG QUERY DEMO")
    print("=" * 60)
    print(f"Question: {question}\n")

    result = default_pipeline.execute(question, top_k=3, token_budget=500)

    print(f"Retrieved Chunks : {result.chunks_retrieved}")
    print(f"Chunks Used      : {result.chunks_used}")
    print(f"Tokens Consumed  : {result.context_tokens_used} / {result.context_budget}")
    print(f"Model            : {result.model_used}")
    print(f"Fallback Used    : {result.fallback_used}")
    if result.fallback_reason:
        print(f"Fallback Reason  : {result.fallback_reason}")

    print("\n--- Sources ---")
    for s in result.sources:
        print(f"• {s['source']} (Page {s['page']}) | Similarity: {s['similarity']:.1%} | Tokens: {s['tokens']}")

    print("\n--- Final Answer ---")
    print(result.answer)
