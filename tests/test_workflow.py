"""Unit tests for LangGraph workflow, JSON parsing, routing, and trace logging."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from src.workflow import (
    WorkflowState,
    build_workflow,
    classify_intent_node,
    extract_missing_info,
    parse_json_safely,
    route_after_check,
    route_after_classify,
    run_workflow,
)


def test_parse_json_safely_plain_json():
    raw = '{"intent": "eligibility", "scheme": "PM-KISAN", "is_personal": true}'
    parsed = parse_json_safely(raw)
    assert parsed is not None
    assert parsed["intent"] == "eligibility"
    assert parsed["scheme"] == "PM-KISAN"
    assert parsed["is_personal"] is True


@patch("src.workflow.get_llm")
def test_classify_intent_deterministic(mock_get_llm):
    """Verify keyword-based deterministic rules override small LLM classification mistakes."""
    mock_llm = MagicMock()
    # Mock LLM returns general_info/benefits to verify that deterministic rules take precedence
    mock_llm.invoke.return_value = AIMessage(content='{"intent": "general_info", "scheme": "Unknown"}')
    mock_get_llm.return_value = mock_llm

    # 1. NMMSS personal eligibility
    q1 = "Can my daughter apply for the NMMSS scholarship? She scored 70% in Class 7 at a state government school, and our annual family income is Rs. 2.5 lakh."
    r1 = classify_intent_node({"question": q1})  # type: ignore
    assert r1["intent"] == "eligibility"
    assert r1["scheme"] == "NMMSS Scholarship"
    assert r1["is_personal"] is True

    # 2. PM SVANidhi personal eligibility (contains certificate without being documents_required)
    q2 = "Am I eligible for a PM SVANidhi loan? I run a vegetable vending cart in a municipal area and hold a Certificate of Vending."
    r2 = classify_intent_node({"question": q2})  # type: ignore
    assert r2["intent"] == "eligibility"
    assert r2["scheme"] == "PM SVANidhi"
    assert r2["is_personal"] is True

    # 3. PM-KISAN personal eligibility
    q3 = "Am I eligible for PM-KISAN? I own 2 acres of land and my family does farming"
    r3 = classify_intent_node({"question": q3})  # type: ignore
    assert r3["intent"] == "eligibility"
    assert r3["scheme"] == "PM-KISAN"
    assert r3["is_personal"] is True

    # 4. Exclusion general eligibility
    q4 = "Who is excluded from PM-KISAN?"
    r4 = classify_intent_node({"question": q4})  # type: ignore
    assert r4["intent"] == "eligibility"
    assert r4["scheme"] == "PM-KISAN"
    assert r4["is_personal"] is False

    # 5. Documents required
    q5 = "What documents are required to apply for PM-KISAN?"
    r5 = classify_intent_node({"question": q5})  # type: ignore
    assert r5["intent"] == "documents_required"
    assert r5["scheme"] == "PM-KISAN"

    # 6. Application process
    q6 = "How to apply for PM SVANidhi online?"
    r6 = classify_intent_node({"question": q6})  # type: ignore
    assert r6["intent"] == "application_process"
    assert r6["scheme"] == "PM SVANidhi"

    # 7. Out of scope
    q7 = "How to bake a chocolate cake?"
    r7 = classify_intent_node({"question": q7})  # type: ignore
    assert r7["intent"] == "out_of_scope"


def test_parse_json_safely_markdown_wrapped():
    raw = '```json\n{"verdict": "SUPPORTED", "unsupported_claims": [], "reason": "All claims backed"}\n```'
    parsed = parse_json_safely(raw)
    assert parsed is not None
    assert parsed["verdict"] == "SUPPORTED"
    assert parsed["reason"] == "All claims backed"


def test_parse_json_safely_malformed():
    raw = "Not a json response at all {broken"
    parsed = parse_json_safely(raw)
    assert parsed is None


def test_extract_missing_info_bullets():
    draft = (
        "Based on what you've told me, you appear to meet the land requirement.\n\n"
        "### Information needed:\n"
        "- Whether you or any family member pays Income Tax\n"
        "- Whether any member holds a constitutional post or government job\n"
        "- Whether active professional practice is carried out\n\n"
        "This is not an official eligibility decision; confirm with the official scheme website."
    )
    missing = extract_missing_info(draft)
    assert len(missing) == 3
    assert "Whether you or any family member pays Income Tax" in missing
    assert "Whether any member holds a constitutional post or government job" in missing


def test_route_after_classify():
    state_out_of_scope: WorkflowState = {"intent": "out_of_scope"}  # type: ignore
    assert route_after_classify(state_out_of_scope) == "fallback"

    state_eligibility: WorkflowState = {"intent": "eligibility"}  # type: ignore
    assert route_after_classify(state_eligibility) == "retrieve"

    state_benefits: WorkflowState = {"intent": "benefits"}  # type: ignore
    assert route_after_classify(state_benefits) == "retrieve"


def test_route_after_check_supported():
    state: WorkflowState = {  # type: ignore
        "verdict": "SUPPORTED",
        "retries": 0,
    }
    assert route_after_check(state) == "finalize"


def test_route_after_check_not_enough_evidence_retry():
    state: WorkflowState = {  # type: ignore
        "verdict": "NOT_ENOUGH_EVIDENCE",
        "retries": 1,
    }
    assert route_after_check(state) == "rewrite_query"


def test_route_after_check_not_enough_evidence_max_retries():
    state: WorkflowState = {  # type: ignore
        "verdict": "NOT_ENOUGH_EVIDENCE",
        "retries": 2,
    }
    assert route_after_check(state) == "fallback"


def test_route_after_check_contradicted_retry():
    state: WorkflowState = {  # type: ignore
        "verdict": "CONTRADICTED",
        "retries": 0,
    }
    assert route_after_check(state) == "rewrite_query"


def test_route_after_check_contradicted_exhausted():
    state: WorkflowState = {  # type: ignore
        "verdict": "CONTRADICTED",
        "retries": 2,
    }
    assert route_after_check(state) == "fallback"


@patch("src.workflow.get_llm")
def test_full_workflow_supported_flow(mock_get_llm):
    """Test full workflow progression when LLM produces a supported answer."""
    # Setup mock LLM responses for:
    # 1. classify_intent -> eligibility, PM-KISAN, is_personal: False
    # 2. generate -> Draft answer
    # 3. check_evidence -> SUPPORTED
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='{"intent": "benefits", "scheme": "PM-KISAN", "is_personal": false}'),
        AIMessage(content="PM-KISAN provides Rs 6,000 per year in three installments of Rs 2,000."),
        AIMessage(content='{"verdict": "SUPPORTED", "unsupported_claims": [], "reason": "Accurate facts"}'),
    ]
    mock_get_llm.return_value = mock_llm

    result = run_workflow("What is the annual benefit of PM-KISAN?")

    assert result["abstained"] is False
    assert result["verdict"] == "SUPPORTED"
    assert result["intent"] == "benefits"
    assert "PM-KISAN provides Rs 6,000" in result["final_answer"]
    assert len(result["trace"]) >= 4

    # Verify trace steps
    node_names = [t["node"] for t in result["trace"]]
    assert "classify_intent" in node_names
    assert "retrieve" in node_names
    assert "generate" in node_names
    assert "check_evidence" in node_names
    assert "finalize" in node_names
