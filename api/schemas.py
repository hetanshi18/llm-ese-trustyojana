"""Pydantic request and response models for TrustYojana API.

Implements Experiment 7 schema definitions:
- AskRequest (min length 3, max 500, whitespace stripped)
- Source
- AskResponse
- HealthResponse
- ErrorResponse
- SchemeInfo
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """User query request model with validation."""
    question: str = Field(..., description="The user's question about an Indian government scheme")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = v.strip() if v else ""
        if len(cleaned) < 3:
            raise ValueError("Question must be at least 3 characters long.")
        if len(cleaned) > 500:
            raise ValueError("Question must not exceed 500 characters.")
        return cleaned


class Source(BaseModel):
    """Source passage metadata citation."""
    scheme: str = Field(..., description="Name of the government scheme")
    title: str = Field(..., description="Title of the source document")
    page: int = Field(default=1, description="Page number of the citation")
    url: str = Field(default="", description="Official scheme URL")
    snippet: str = Field(default="", description="Relevant text snippet")
    doc_date: str = Field(default="unknown", description="Published / updated date of the document")
    date_accessed: str = Field(default="unknown", description="Date the document was accessed / indexed")


class AskResponse(BaseModel):
    """Full question response from LangGraph workflow."""
    request_id: str = Field(..., description="Unique UUID for this request")
    answer: str = Field(..., description="Final grounded answer or fallback notice")
    abstained: bool = Field(..., description="Whether system abstained due to lack of evidence or out of scope")
    verdict: str = Field(..., description="Evidence check verdict: SUPPORTED, NOT_ENOUGH_EVIDENCE, CONTRADICTED")
    intent: str = Field(..., description="Classified intent")
    scheme: str = Field(..., description="Target scheme name")
    is_personal: bool = Field(..., description="Whether question is about user's own situation")
    missing_info: List[str] = Field(default_factory=list, description="Information still needed for personal eligibility")
    sources: List[Source] = Field(default_factory=list, description="Retrieved and verified source citations")
    retries: int = Field(default=0, description="Number of query rewrites/retries performed")
    trace: List[Dict[str, Any]] = Field(default_factory=list, description="Step-by-step workflow execution trace")
    latency_s: float = Field(..., description="Total processing latency in seconds")


class HealthResponse(BaseModel):
    """Detailed health check response reporting separate vector store and LLM states."""
    status: str = Field(..., description="'ok' if healthy, 'degraded' otherwise")
    vector_db_ok: bool = Field(..., description="Whether ChromaDB opens and contains chunks")
    indexed_chunks: int = Field(..., description="Total chunks currently indexed in ChromaDB")
    llm_config_ok: bool = Field(..., description="Whether LLM provider, model, and credentials are configured")
    llm_reachable: Optional[bool] = Field(None, description="Reachability status (tested when ?deep=true, else null)")
    version: str = Field(..., description="Application version")


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    request_id: str = Field(..., description="Unique UUID for this request")
    error: str = Field(..., description="High-level error type or message")
    detail: Optional[str] = Field(None, description="Safe details about the error")


class SchemeInfo(BaseModel):
    """Information regarding an indexed government scheme."""
    scheme: str
    title: str
    source_file: str
    url: str
    doc_date: str
    date_accessed: str
    chunks_count: int
