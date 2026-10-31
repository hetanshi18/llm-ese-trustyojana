"""Unit tests for LangGraph workflow, JSON parsing, routing, and trace logging."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from src.workflow import (
    WorkflowState,
    build_workflow,
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
