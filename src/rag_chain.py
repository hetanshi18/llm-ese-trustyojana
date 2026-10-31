"""Basic Retrieve-and-Answer pipeline for TrustYojana.

Implements Experiment 5:
- Retrieves top-k chunks from ChromaDB
- Builds grounded context prompt
- Invokes provider-agnostic LLM
- Returns structured response with answer and cited sources
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_config
from src.llm import get_llm
from src.prompts import BASIC_RAG_SYSTEM_PROMPT, BASIC_RAG_USER_PROMPT
from src.retriever import retrieve

logger = logging.getLogger(__name__)


def format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for LLM prompting."""
    if not chunks:
        return "No relevant documents found."

    formatted_parts = []
    for idx, c in enumerate(chunks, 1):
        source = c.get("source_file", "Doc")
        page = c.get("page", 1)
        title = c.get("title", source)
        text = c.get("text", "").strip()
        formatted_parts.append(f"Passage {idx} [Source: {title}, File: {source}, Page: {page}]:\n{text}")

    return "\n\n".join(formatted_parts)


def format_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format chunk metadata into structured sources list."""
    sources: List[Dict[str, Any]] = []
    seen_keys = set()

    for c in chunks:
        # Deduplicate identical source file and page
        key = (c.get("source_file"), c.get("page"))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        snippet = c.get("text", "")[:250] + "..." if len(c.get("text", "")) > 250 else c.get("text", "")
        sources.append({
            "scheme": c.get("scheme", "Unknown"),
            "title": c.get("title", ""),
            "page": int(c.get("page", 1)),
            "url": c.get("url", ""),
            "snippet": snippet,
            "doc_date": c.get("doc_date", "unknown"),
            "date_accessed": c.get("date_accessed", "unknown"),
        })

    return sources


def answer(question: str, top_k: Optional[int] = None, scheme: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve chunks, prompt LLM, and return {answer, sources[]}."""
    config = get_config()
    k = top_k if top_k is not None else config.rag.top_k

    logger.info(f"RAG Chain answering question: length={len(question)}")
    chunks = retrieve(query=question, k=k, scheme=scheme)

    if not chunks:
        return {
            "answer": "The provided documents do not contain enough information to answer this question.",
            "sources": [],
        }

    context_str = format_context_passages(chunks)
    system_msg = SystemMessage(content=BASIC_RAG_SYSTEM_PROMPT)
    user_msg = HumanMessage(content=BASIC_RAG_USER_PROMPT.format(context=context_str, question=question))

    llm = get_llm()
    response = llm.invoke([system_msg, user_msg])
    answer_text = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer_text,
        "sources": format_sources(chunks),
    }
