"""LangGraph workflow for TrustYojana.

Implements Experiment 6:
- StateGraph with nodes:
  1. classify_intent
  2. retrieve
  3. generate (with stateless single-turn personal eligibility handling)
  4. check_evidence (claim-level LLM grader with 1-retry robust JSON parsing)
  5. rewrite_query (query reformulator incrementing retries)
  6. finalize (grounded answer with sources)
  7. fallback (not enough info message with best matching source links)
- Trace tracking at every node
- Conditional edge routing
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.config import get_config
from src.llm import get_llm
from src.prompts import (
    DRAFT_GENERATION_SYSTEM_PROMPT,
    DRAFT_GENERATION_USER_PROMPT,
    EVIDENCE_CHECK_SYSTEM_PROMPT,
    EVIDENCE_CHECK_USER_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_USER_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
)
from src.rag_chain import format_context_passages, format_sources
from src.retriever import retrieve

logger = logging.getLogger(__name__)


# ==============================================================================
# Graph State
# ==============================================================================

class WorkflowState(TypedDict):
    question: str
    intent: str
    scheme: str
    is_personal: bool
    current_query: str
    chunks: List[Dict[str, Any]]
    draft_answer: str
    verdict: str
    verdict_reason: str
    unsupported_claims: List[str]
    retries: int
    final_answer: str
    sources: List[Dict[str, Any]]
    missing_info: List[str]
    abstained: bool
    trace: List[Dict[str, Any]]


def create_trace_entry(node_name: str, summary: str) -> Dict[str, Any]:
    """Helper to record node transition in trace list."""
    return {
        "node": node_name,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }


def parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract and parse JSON from LLM response."""
    text = raw_text.strip()
    # Check for markdown code blocks
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()
    else:
        # Search for first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1].strip()

    try:
        return json.loads(text)
    except Exception:
        return None


def extract_missing_info(draft: str) -> List[str]:
    """Extract missing information items listed in the draft answer."""
    missing: List[str] = []
    lines = draft.splitlines()
    in_missing_section = False

    for line in lines:
        stripped = line.strip()
        lower_line = stripped.lower()
        if any(h in lower_line for h in ("information needed", "missing info", "details needed")):
            in_missing_section = True
            continue

        if in_missing_section:
            # If line starts another major heading, break out
            if stripped.startswith("#") or (stripped.endswith(":") and not stripped.startswith("-")):
                in_missing_section = False
                continue
            if stripped.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
                clean_item = re.sub(r"^[-*•\d.]+\s*", "", stripped).strip()
                if clean_item:
                    missing.append(clean_item)

    return missing


# ==============================================================================
# Graph Nodes
# ==============================================================================

def classify_intent_node(state: WorkflowState) -> Dict[str, Any]:
    """Node 1: Classify intent, scheme, and personal query flag."""
    question = state["question"]
    llm = get_llm()

    sys_msg = SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT)
    user_msg = HumanMessage(content=INTENT_CLASSIFICATION_USER_PROMPT.format(question=question))

    try:
        response = llm.invoke([sys_msg, user_msg])
        parsed = parse_json_safely(response.content) or {}
    except Exception as e:
        logger.warning(f"Intent classification call failed: {e}. Using defaults.")
        parsed = {}

    intent = parsed.get("intent", "general_info").lower()
    valid_intents = {"eligibility", "benefits", "application_process", "documents_required", "general_info", "out_of_scope"}
    if intent not in valid_intents:
        intent = "general_info"

    scheme = parsed.get("scheme", "Unknown")
    is_personal = bool(parsed.get("is_personal", False))

    trace_summary = f"Classified intent: '{intent}', scheme: '{scheme}', personal: {is_personal}"
    trace = state.get("trace", []) + [create_trace_entry("classify_intent", trace_summary)]

    return {
        "intent": intent,
        "scheme": scheme,
        "is_personal": is_personal,
        "current_query": question,
        "trace": trace,
    }


def retrieve_node(state: WorkflowState) -> Dict[str, Any]:
    """Node 2: Retrieve chunks from ChromaDB with current query and scheme."""
    query = state.get("current_query", state["question"])
    scheme = state.get("scheme")
    config = get_config()

    # If scheme is Unknown/All/Other, pass None so retriever searches across all schemes
    scheme_filter = None if scheme in (None, "Unknown", "All", "Other", "") else scheme

    chunks = retrieve(query=query, k=config.rag.top_k, scheme=scheme_filter)
    sources = format_sources(chunks)

    trace_summary = f"Retrieved {len(chunks)} chunks for query: '{query}' (scheme: {scheme_filter or 'All'})"
    trace = state.get("trace", []) + [create_trace_entry("retrieve", trace_summary)]

    return {
        "chunks": chunks,
        "sources": sources,
        "trace": trace,
    }


def generate_node(state: WorkflowState) -> Dict[str, Any]:
    """Node 3: Draft an answer based solely on retrieved passages."""
    question = state["question"]
    chunks = state.get("chunks", [])
    intent = state.get("intent", "general_info")
    scheme = state.get("scheme", "Unknown")
    is_personal = state.get("is_personal", False)
    verdict = state.get("verdict", "")
    verdict_reason = state.get("verdict_reason", "")

    feedback_section = ""
    if verdict == "CONTRADICTED" and verdict_reason:
        feedback_section = (
            f"\nATTENTION: A previous draft contradicted the source documents for the following reason:\n"
            f"'{verdict_reason}'. You must strictly correct this and adhere only to the context passages below.\n"
        )

    context_str = format_context_passages(chunks)
    llm = get_llm()

    sys_msg = SystemMessage(content=DRAFT_GENERATION_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        content=DRAFT_GENERATION_USER_PROMPT.format(
            context=context_str,
            question=question,
            intent=intent,
            scheme=scheme,
            is_personal=is_personal,
            feedback_section=feedback_section,
        )
    )

    try:
        response = llm.invoke([sys_msg, user_msg])
        draft_answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Error in generate node: {e}")
        draft_answer = "Unable to generate draft answer due to an internal error."

    # Extract missing info for personal eligibility
    missing_info = extract_missing_info(draft_answer) if (intent == "eligibility" and is_personal) else []

    trace_summary = f"Drafted answer ({len(draft_answer)} chars), personal: {is_personal}, missing items: {len(missing_info)}"
    trace = state.get("trace", []) + [create_trace_entry("generate", trace_summary)]

    return {
        "draft_answer": draft_answer,
        "missing_info": missing_info,
        "trace": trace,
    }


def check_evidence_node(state: WorkflowState) -> Dict[str, Any]:
    """Node 4: Claim-level LLM evidence grader with 1 retry on malformed JSON."""
    chunks = state.get("chunks", [])
    draft_answer = state.get("draft_answer", "")
    context_str = format_context_passages(chunks)
    llm = get_llm()

    sys_msg = SystemMessage(content=EVIDENCE_CHECK_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        content=EVIDENCE_CHECK_USER_PROMPT.format(
            context=context_str,
            draft_answer=draft_answer,
        )
    )

    parsed = None
    # First attempt
    try:
        response = llm.invoke([sys_msg, user_msg])
        parsed = parse_json_safely(response.content)
    except Exception as e:
        logger.warning(f"Check evidence first call failed: {e}")

    # One retry if malformed JSON
    if parsed is None or "verdict" not in parsed:
        logger.info("Retrying check_evidence with formatting reminder...")
        retry_prompt = (
            "Your previous output was not valid JSON. Please reply with ONLY a JSON object:\n"
            '{"verdict": "SUPPORTED" | "NOT_ENOUGH_EVIDENCE" | "CONTRADICTED", "unsupported_claims": [], "reason": "..."}'
        )
        try:
            retry_res = llm.invoke([sys_msg, user_msg, HumanMessage(content=retry_prompt)])
            parsed = parse_json_safely(retry_res.content)
        except Exception as e:
            logger.warning(f"Check evidence retry failed: {e}")

    if not parsed or "verdict" not in parsed:
        # Conservative fallback verdict
        verdict = "NOT_ENOUGH_EVIDENCE"
        unsupported = ["Unable to verify claims due to formatting error."]
        reason = "Grader failed to produce valid JSON verification."
    else:
        raw_verdict = str(parsed.get("verdict", "NOT_ENOUGH_EVIDENCE")).upper().strip()
        if raw_verdict in ("SUPPORTED", "NOT_ENOUGH_EVIDENCE", "CONTRADICTED"):
            verdict = raw_verdict
        else:
            verdict = "NOT_ENOUGH_EVIDENCE"
        unsupported = parsed.get("unsupported_claims", [])
        reason = parsed.get("reason", "Evaluated by evidence checker.")

    trace_summary = f"Evidence verdict: {verdict}. Reason: {reason[:120]}..."
    trace = state.get("trace", []) + [create_trace_entry("check_evidence", trace_summary)]

    return {
        "verdict": verdict,
        "verdict_reason": reason,
        "unsupported_claims": unsupported,
        "trace": trace,
    }


def rewrite_query_node(state: WorkflowState) -> Dict[str, Any]:
    """Node 5: Rewrite query based on intent, scheme, verdict, and reason. Increments retries."""
    llm = get_llm()
    retries = state.get("retries", 0) + 1

    sys_msg = SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        content=QUERY_REWRITE_USER_PROMPT.format(
            question=state["question"],
            current_query=state.get("current_query", state["question"]),
            intent=state.get("intent", ""),
            scheme=state.get("scheme", ""),
            verdict=state.get("verdict", ""),
            verdict_reason=state.get("verdict_reason", ""),
            unsupported_claims=", ".join(state.get("unsupported_claims", [])),
        )
    )

    try:
        response = llm.invoke([sys_msg, user_msg])
        new_query = response.content.strip().replace('"', "")
        # Prevent empty query
        if not new_query:
            new_query = f"{state.get('scheme', '')} {state['question']}".strip()
    except Exception as e:
        logger.warning(f"Rewrite query call failed: {e}")
        new_query = f"{state.get('scheme', '')} {state['question']}".strip()

    trace_summary = f"Rewrote query (attempt {retries}): '{new_query}'"
    trace = state.get("trace", []) + [create_trace_entry("rewrite_query", trace_summary)]

    return {
        "current_query": new_query,
        "retries": retries,
        "trace": trace,
    }


def finalize_node(state: WorkflowState) -> Dict[str, Any]:
    """Node 6: Finalize supported draft answer with verified sources."""
    trace_summary = f"Finalized verified answer with {len(state.get('sources', []))} source citations."
    trace = state.get("trace", []) + [create_trace_entry("finalize", trace_summary)]

    return {
        "final_answer": state.get("draft_answer", ""),
        "abstained": False,
        "trace": trace,
    }


def fallback_node(state: WorkflowState) -> Dict[str, Any]:
    """Node 7: Fallback when not enough evidence or out of scope."""
    intent = state.get("intent", "")
    sources = state.get("sources", [])

    if intent == "out_of_scope":
        answer = (
            "I don't have enough information in the available scheme documents to answer this reliably. "
            "TrustYojana is specifically designed for Indian government welfare schemes (such as PM-KISAN, "
            "Ayushman Bharat PM-JAY, NMMSS Scholarship, and PM SVANidhi). "
            "Please check the official government portal for information on other topics."
        )
    else:
        answer = (
            "I don't have enough information in the available scheme documents to answer this reliably. "
            "Please check the official scheme website."
        )

    trace_summary = f"Returned fallback response (intent: {intent}, abstained: True)."
    trace = state.get("trace", []) + [create_trace_entry("fallback", trace_summary)]

    return {
        "final_answer": answer,
        "abstained": True,
        "trace": trace,
    }


# ==============================================================================
# Conditional Edge Routers
# ==============================================================================

def route_after_classify(state: WorkflowState) -> str:
    """Route from classify_intent: out_of_scope goes directly to fallback, otherwise retrieve."""
    intent = state.get("intent", "")
    if intent == "out_of_scope":
        return "fallback"
    return "retrieve"


def route_after_check(state: WorkflowState) -> str:
    """Route from check_evidence:
    - SUPPORTED -> finalize
    - NOT_ENOUGH_EVIDENCE / CONTRADICTED and retries < max_retries -> rewrite_query
    - Otherwise -> fallback
    """
    verdict = state.get("verdict", "NOT_ENOUGH_EVIDENCE")
    retries = state.get("retries", 0)
    config = get_config()
    max_retries = config.rag.max_retries

    if verdict == "SUPPORTED":
        return "finalize"

    if verdict in ("NOT_ENOUGH_EVIDENCE", "CONTRADICTED") and retries < max_retries:
        return "rewrite_query"

    return "fallback"


# ==============================================================================
# Graph Construction
# ==============================================================================

def build_workflow() -> Any:
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(WorkflowState)

    # Add Nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_evidence", check_evidence_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("fallback", fallback_node)

    # Add Edges
    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "check_evidence")

    workflow.add_conditional_edges(
        "check_evidence",
        route_after_check,
        {
            "finalize": "finalize",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("finalize", END)
    workflow.add_edge("fallback", END)

    app = workflow.compile()
    return app


# Cached compiled workflow
_compiled_workflow = None


def get_workflow():
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow


def run_workflow(question: str) -> WorkflowState:
    """Run the complete LangGraph workflow for a question."""
    initial_state: WorkflowState = {
        "question": question.strip(),
        "intent": "",
        "scheme": "",
        "is_personal": False,
        "current_query": question.strip(),
        "chunks": [],
        "draft_answer": "",
        "verdict": "",
        "verdict_reason": "",
        "unsupported_claims": [],
        "retries": 0,
        "final_answer": "",
        "sources": [],
        "missing_info": [],
        "abstained": False,
        "trace": [],
    }

    graph = get_workflow()
    result = graph.invoke(initial_state)
    return result
