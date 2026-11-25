import os
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from config import (
    collection,
    generate_embeddings,
    count_tokens,
    DEFAULT_TOP_K,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    SCORE_THRESHOLD,
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    page: int
    tenant_id: str
    distance: float
    similarity: float
    token_count: int


@dataclass
class QueryResult:
    question: str
    answer: str
    context: List[str]
    sources: List[Dict[str, Any]]
    context_tokens_used: int
    context_budget: int
    chunks_retrieved: int
    chunks_used: int
    model_used: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "sources": self.sources,
            "context_tokens_used": self.context_tokens_used,
            "context_budget": self.context_budget,
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_used": self.chunks_used,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


class QueryPipeline:
    """
    Complete RAG Query Pipeline:
    1. Embed Query
    2. Retrieve Top-K with Metadata & Scores
    3. Apply Context Token Budget
    4. Build Grounded Prompt
    5. Call LLM
    6. Multi-tiered Fallbacks (zero context, secondary model, extractive offline)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        default_top_k: int = DEFAULT_TOP_K,
        default_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL)
        self.fallback_model = fallback_model or os.getenv("OPENROUTER_FALLBACK_MODEL", OPENROUTER_FALLBACK_MODEL)
        self.default_top_k = default_top_k
        self.default_budget = default_budget
        self.score_threshold = score_threshold

    # -------------------------------------------------------------
    # Step 1: Embed the Query
    # -------------------------------------------------------------
    def embed_query(self, query: str) -> List[float]:
        """Vectorize user query with embedding model."""
        embeddings = generate_embeddings([query])
        return embeddings[0]

    # -------------------------------------------------------------
    # Step 2: Retrieve Top-K with Metadata & Distances
    # -------------------------------------------------------------
    def retrieve_top_k(
        self,
        query: str,
        top_k: Optional[int] = None,
        tenant_id: str = "tenant_001",
    ) -> List[RetrievedChunk]:
        """
        Embeds the query and queries ChromaDB for top-k matching documents
        filtered by tenant_id and relevance score.
        """
        k = top_k or self.default_top_k
        total_docs = collection.count()
        if total_docs == 0:
            return []

        query_embedding = self.embed_query(query)
        effective_k = min(k, total_docs)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            where={"tenant_id": tenant_id},
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        doc_list = results.get("documents", [[]])[0]
        meta_list = results.get("metadatas", [[]])[0]
        id_list = results.get("ids", [[]])[0]
        dist_list = results.get("distances", [[]])[0]

        for i in range(len(doc_list)):
            text = doc_list[i]
            meta = meta_list[i] if i < len(meta_list) else {}
            chunk_id = id_list[i] if i < len(id_list) else f"chunk_{i}"
            dist = dist_list[i] if i < len(dist_list) else 1.0

            # Cosine distance: 0.0 means identical, 1.0 means orthogonal
            similarity = max(0.0, 1.0 - dist)

            # Apply score threshold filter (drop chunks that are too distant)
            if dist > self.score_threshold:
                continue

            token_count = count_tokens(text)

            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    source=meta.get("source", "unknown"),
                    page=meta.get("page", 1),
                    tenant_id=meta.get("tenant_id", tenant_id),
                    distance=dist,
                    similarity=similarity,
                    token_count=token_count,
                )
            )

        # Sort by similarity descending (distance ascending)
        chunks.sort(key=lambda c: c.distance)
        return chunks

    # -------------------------------------------------------------
    # Step 3: Context Token Budgeting
    # -------------------------------------------------------------
    def apply_token_budget(
        self,
        chunks: List[RetrievedChunk],
        token_budget: Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], int, int]:
        """
        Enforce a strict context token budget.
        Greedily adds top chunks until adding another would exceed the budget.
        Returns: (selected_chunks, total_tokens_used, chunks_dropped)
        """
        budget = token_budget if token_budget is not None else self.default_budget
        selected: List[RetrievedChunk] = []
        tokens_used = 0
        dropped_count = 0

        for chunk in chunks:
            if tokens_used + chunk.token_count <= budget:
                selected.append(chunk)
                tokens_used += chunk.token_count
            else:
                dropped_count += 1

        return selected, tokens_used, dropped_count

    # -------------------------------------------------------------
    # Step 4: Build Prompt
    # -------------------------------------------------------------
    def build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Construct grounded prompt containing structured context and citations."""
        formatted_context_sections = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Document {i}: {chunk.source} | Page {chunk.page} | Relevance: {chunk.similarity:.1%}]"
            formatted_context_sections.append(f"{header}\n{chunk.text}")

        context_str = "\n\n".join(formatted_context_sections)

        prompt = f"""You are an HRMS AI assistant.

Answer the user's question using ONLY the provided company policy context below.
Strict rules:
1. Ground your answer strictly in the provided Context.
2. If the answer is not contained in the context, explicitly say that you do not have sufficient information in the company policy. Do not guess or extrapolate.
3. Cite the relevant policy source document and page number in your answer.

Context:
{context_str}

User Question:
{question}

Answer:"""
        return prompt

    # -------------------------------------------------------------
    # Step 5: Call LLM API (OpenRouter)
    # -------------------------------------------------------------
    def _call_openrouter_api(self, prompt: str, model_name: str, timeout: int = 45) -> str:
        """Direct call to OpenRouter with specified model."""
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")

        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=timeout,
        )

        if response.status_code != 200:
            err_msg = "Unknown error"
            try:
                err_data = response.json()
                err_msg = err_data.get("error", {}).get("message", response.text)
            except Exception:
                err_msg = response.text
            raise RuntimeError(f"HTTP {response.status_code}: {err_msg}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def call_llm_with_fallback(
        self,
        prompt: str,
        preferred_model: Optional[str] = None,
    ) -> Tuple[str, str, bool, Optional[str]]:
        """
        Attempts primary model. If it fails, attempts secondary model fallback.
        Returns: (answer, model_used, fallback_used, fallback_reason)
        """
        primary = preferred_model or self.model
        fallback = self.fallback_model

        # 1. Try Primary Model
        try:
            answer = self._call_openrouter_api(prompt, primary)
            return answer, primary, False, None
        except Exception as primary_error:
            primary_reason = str(primary_error)

            # 2. Try Fallback Model if distinct
            if fallback and fallback != primary:
                try:
                    answer = self._call_openrouter_api(prompt, fallback)
                    return (
                        answer,
                        fallback,
                        True,
                        f"Primary model ({primary}) failed: {primary_reason}. Succeeded with fallback model ({fallback}).",
                    )
                except Exception as fallback_error:
                    secondary_reason = str(fallback_error)
                    raise RuntimeError(
                        f"Primary model error: {primary_reason} | Secondary model error: {secondary_reason}"
                    )

            raise RuntimeError(primary_reason)

    # -------------------------------------------------------------
    # Step 6: End-to-End Query Pipeline Execution
    # -------------------------------------------------------------
    def execute(
        self,
        question: str,
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
        tenant_id: str = "tenant_001",
    ) -> QueryResult:
        """
        Executes the entire RAG pipeline:
        1. Embed question
        2. Retrieve top-k
        3. Enforce context token budget
        4. Handle empty/low-similarity context fallback
        5. Build prompt
        6. Call LLM (with model and extractive fallbacks)
        """
        k = top_k or self.default_top_k
        budget = token_budget if token_budget is not None else self.default_budget

        # 1. Retrieve top-k
        all_retrieved = self.retrieve_top_k(question, top_k=k, tenant_id=tenant_id)
        chunks_retrieved_count = len(all_retrieved)

        # 2. Token Budget enforcement
        selected_chunks, tokens_used, dropped = self.apply_token_budget(
            all_retrieved, token_budget=budget
        )

        sources_metadata = [
            {
                "source": c.source,
                "page": c.page,
                "similarity": round(c.similarity, 4),
                "distance": round(c.distance, 4),
                "tokens": c.token_count,
            }
            for c in selected_chunks
        ]
        context_texts = [c.text for c in selected_chunks]

        # 3. Fallback: No relevant context found
        if not selected_chunks:
            return QueryResult(
                question=question,
                answer=(
                    "I do not have sufficient information in the company policy documents "
                    "to answer this question. Please reach out to your HR administrator."
                ),
                context=[],
                sources=[],
                context_tokens_used=0,
                context_budget=budget,
                chunks_retrieved=chunks_retrieved_count,
                chunks_used=0,
                model_used=None,
                fallback_used=True,
                fallback_reason="No relevant policy documents found matching the question.",
            )

        # 4. Build Prompt
        prompt = self.build_prompt(question, selected_chunks)

        # 5. Call LLM with Multi-tiered Fallbacks
        try:
            answer, model_used, fallback_used, fallback_reason = self.call_llm_with_fallback(prompt)
            return QueryResult(
                question=question,
                answer=answer,
                context=context_texts,
                sources=sources_metadata,
                context_tokens_used=tokens_used,
                context_budget=budget,
                chunks_retrieved=chunks_retrieved_count,
                chunks_used=len(selected_chunks),
                model_used=model_used,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                prompt=prompt,
            )

        except Exception as llm_err:
            # Fallback 3: Offline Extractive Fallback
            # When LLM is unreachable (e.g. API key expired or connection error),
            # provide direct relevant excerpts from the verified documents!
            excerpt_lines = []
            for c in selected_chunks:
                excerpt_lines.append(f"• From '{c.source}' (Page {c.page}):\n  {c.text.strip()}")

            excerpts_str = "\n\n".join(excerpt_lines)
            fallback_answer = (
                f"[Offline Fallback - Direct Policy Excerpts]\n"
                f"The LLM service is currently unavailable ({str(llm_err)}).\n"
                f"Here are the relevant excerpts retrieved directly from your company policy documents:\n\n"
                f"{excerpts_str}"
            )

            return QueryResult(
                question=question,
                answer=fallback_answer,
                context=context_texts,
                sources=sources_metadata,
                context_tokens_used=tokens_used,
                context_budget=budget,
                chunks_retrieved=chunks_retrieved_count,
                chunks_used=len(selected_chunks),
                model_used="extractive_fallback",
                fallback_used=True,
                fallback_reason=f"LLM API failure: {str(llm_err)}",
                prompt=prompt,
            )


# Global singleton instance for easy import
default_pipeline = QueryPipeline()


if __name__ == "__main__":
    print("=" * 60)
    print("HRMS RAG QUERY PIPELINE - TEST RUN")
    print("=" * 60)

    sample_questions = [
        "How many casual leaves are available?",
        "What is the procedure for sick leave approval?",
        "Can you tell me about the company quantum computing cluster?",  # Irrelevant query
    ]

    pipeline = QueryPipeline()

    for q in sample_questions:
        print(f"\n>> Question: {q}")
        res = pipeline.execute(q, top_k=3, token_budget=500)
        print(f"Chunks Retrieved: {res.chunks_retrieved} | Chunks Used: {res.chunks_used}")
        print(f"Context Tokens Used: {res.context_tokens_used} / {res.context_budget}")
        print(f"Model: {res.model_used} | Fallback: {res.fallback_used}")
        if res.fallback_reason:
            print(f"Fallback Reason: {res.fallback_reason}")
        print("\nAnswer:\n", res.answer)
        print("-" * 60)
