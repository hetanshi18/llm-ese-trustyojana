"""FastAPI application for TrustYojana.

Implements Experiment 7:
- Lifespan startup for pre-loading retriever and LangGraph workflow
- POST /ask: validates input, executes workflow, returns AskResponse
- GET /health: independent checks for vector_db_ok, llm_config_ok, and deep llm_reachable
- GET /schemes: lists indexed schemes and metadata
- Middleware injecting unique X-Request-ID
- Strict privacy-preserving logging (logs question length and hash only, NO question text)
- Rotating log file in logs/api.log
- CORS support for Streamlit frontend
- Exception handlers returning standardized ErrorResponse
"""

import hashlib
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    SchemeInfo,
    Source,
)
from src.config import get_config
from src.llm import check_llm_config, check_llm_reachability
from src.retriever import get_retriever
from src.workflow import get_workflow, run_workflow

# ------------------------------------------------------------------------------
# Logging Setup (Privacy Preserving & Rotating)
# ------------------------------------------------------------------------------

config = get_config()
logs_dir = config.paths.logs_path
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "api.log"

logger = logging.getLogger("trustyojana_api")
logger.setLevel(getattr(logging, config.app.log_level.upper(), logging.INFO))
logger.propagate = False

# Remove existing handlers to avoid duplicate log entries
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler (5MB, 3 backups)
    rotating_handler = RotatingFileHandler(
        str(log_file_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    rotating_handler.setFormatter(formatter)
    logger.addHandler(rotating_handler)


def hash_query(text: str) -> str:
    """Compute short non-reversible SHA256 prefix for privacy-safe logging."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ------------------------------------------------------------------------------
# Lifespan Context Manager
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event: initialize retriever and compile workflow at startup."""
    logger.info("Initializing TrustYojana system during lifespan startup...")
    try:
        retriever = get_retriever()
        total_chunks = retriever.count()
        if total_chunks == 0:
            logger.error("ChromaDB vector store is EMPTY! Run 'python -m src.ingest' to populate it.")
        else:
            logger.info(f"ChromaDB connected successfully. Total chunks indexed: {total_chunks}")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB during startup: {e}")

    try:
        get_workflow()
        logger.info("LangGraph workflow compiled successfully.")
    except Exception as e:
        logger.error(f"Failed to compile LangGraph workflow: {e}")

    yield
    logger.info("TrustYojana API shutting down.")


# ------------------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------------------

app = FastAPI(
    title=config.app.name,
    version=config.app.version,
    description="RAG-based Question-Answering API for Indian Government Schemes",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Request ID Middleware
# ------------------------------------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach unique request ID to request state and response headers."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ------------------------------------------------------------------------------
# Custom Exception Handlers
# ------------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle 422 input validation errors."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    error_msg = "Invalid request input."
    # Extract specific details
    details = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "")
        details.append(f"{loc}: {msg}")
    detail_str = "; ".join(details) if details else str(exc)

    logger.warning(f"[{req_id}] Validation error 422: {detail_str}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            request_id=req_id,
            error=error_msg,
            detail=detail_str,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions with ErrorResponse format."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning(f"[{req_id}] HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            request_id=req_id,
            error=f"HTTP_{exc.status_code}",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(TimeoutError)
async def timeout_exception_handler(request: Request, exc: TimeoutError):
    """Handle 504 on LLM or external service timeout."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"[{req_id}] Service timeout 504: {exc}")
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content=ErrorResponse(
            request_id=req_id,
            error="LLM_TIMEOUT",
            detail="The upstream LLM request timed out. Please try again shortly.",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all 500 handler returning safe message without leaking stack traces or keys."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"[{req_id}] Unhandled internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            request_id=req_id,
            error="INTERNAL_SERVER_ERROR",
            detail="An unexpected error occurred while processing your request. Please try again later.",
        ).model_dump(),
    )


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(payload: AskRequest, request: Request):
    """Process a user question through the LangGraph RAG workflow."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()

    question = payload.question
    q_len = len(question)
    q_hash = hash_query(question)

    retriever = get_retriever()
    total_chunks = retriever.count()
    if total_chunks == 0:
        logger.error(f"[{req_id}] Vector store is empty during /ask.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not initialized or empty. Please ingest scheme documents first.",
        )

    try:
        result = run_workflow(question)
    except TimeoutError as te:
        raise te
    except Exception as e:
        logger.error(f"[{req_id}] Error executing workflow: {e}", exc_info=True)
        # Check if it was an LLM service unavailability
        err_str = str(e).lower()
        if "api_key" in err_str or "authentication" in err_str or "connect" in err_str or "connection" in err_str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service is currently unavailable or improperly configured.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer due to an internal workflow error.",
        )

    latency_s = round(time.time() - start_time, 3)

    # Format response fields
    intent = result.get("intent", "general_info")
    scheme = result.get("scheme", "Unknown")
    verdict = result.get("verdict", "SUPPORTED")
    is_personal = result.get("is_personal", False)
    abstained = result.get("abstained", False)
    retries = result.get("retries", 0)
    final_answer = result.get("final_answer", "")
    missing_info = result.get("missing_info", [])
    raw_sources = result.get("sources", [])

    sources_models = [
        Source(
            scheme=s.get("scheme", "Unknown"),
            title=s.get("title", ""),
            page=int(s.get("page", 1)),
            url=s.get("url", ""),
            snippet=s.get("snippet", ""),
            doc_date=s.get("doc_date", "unknown"),
            date_accessed=s.get("date_accessed", "unknown"),
        )
        for s in raw_sources
    ]

    outcome = "abstained" if abstained else "answered"

    # Strictly log length and hash - NO question text or personal info
    logger.info(
        f"[{req_id}] len={q_len} hash={q_hash} intent={intent} scheme={scheme} "
        f"verdict={verdict} retries={retries} outcome={outcome} latency={latency_s:.3f}s"
    )

    return AskResponse(
        request_id=req_id,
        answer=final_answer,
        abstained=abstained,
        verdict=verdict,
        intent=intent,
        scheme=scheme,
        is_personal=is_personal,
        missing_info=missing_info,
        sources=sources_models,
        retries=retries,
        trace=result.get("trace", []),
        latency_s=latency_s,
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint(deep: bool = Query(False, description="Perform live LLM reachability probe")):
    """Health check endpoint.
    
    Reports separate statuses for:
    - vector_db_ok: ChromaDB collection opens and has chunks
    - llm_config_ok: provider and credentials configured
    - llm_reachable: tested with a minimal prompt when deep=true, else null
    """
    retriever = get_retriever()
    chunk_count = retriever.count()
    vector_db_ok = chunk_count > 0

    llm_config_ok, _ = check_llm_config()

    llm_reachable = None
    if deep:
        reachable, _ = check_llm_reachability(force=False)
        llm_reachable = reachable

    # Determine overall status and HTTP code
    # Healthy if vector_db and llm_config are ok, and deep check did not fail
    is_healthy = vector_db_ok and llm_config_ok and (llm_reachable is not False)
    status_text = "ok" if is_healthy else "degraded"

    response_data = HealthResponse(
        status=status_text,
        vector_db_ok=vector_db_ok,
        indexed_chunks=chunk_count,
        llm_config_ok=llm_config_ok,
        llm_reachable=llm_reachable,
        version=config.app.version,
    )

    if not is_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_data.model_dump(),
        )

    return response_data


@app.get("/schemes", response_model=list[SchemeInfo])
async def list_schemes_endpoint():
    """List all indexed government schemes with metadata and last-accessed dates."""
    retriever = get_retriever()
    schemes = retriever.get_indexed_schemes()
    return [SchemeInfo(**s) for s in schemes]
