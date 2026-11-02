"""Unit tests for FastAPI endpoints, request validation, health checks, and privacy logging."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import get_config

client = TestClient(app)


@patch("api.main.get_retriever")
def test_health_without_deep(mock_get_retriever):
    mock_ret = MagicMock()
    mock_ret.count.return_value = 207
    mock_get_retriever.return_value = mock_ret

    response = client.get("/health")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "vector_db_ok" in data
    assert "indexed_chunks" in data
    assert "llm_config_ok" in data
    assert data["llm_reachable"] is None  # Must be null by default


@patch("api.main.get_retriever")
@patch("api.main.check_llm_config")
@patch("api.main.check_llm_reachability")
def test_health_with_deep_when_configured(mock_reachability, mock_config, mock_get_retriever):
    mock_ret = MagicMock()
    mock_ret.count.return_value = 207
    mock_get_retriever.return_value = mock_ret

    mock_config.return_value = (True, "Valid config")
    mock_reachability.return_value = (True, None)
    response = client.get("/health?deep=true")
    assert response.status_code == 200
    data = response.json()
    assert data["llm_config_ok"] is True
    assert data["llm_reachable"] is True
    assert data["status"] == "ok"


@patch("api.main.get_retriever")
@patch("api.main.check_llm_config")
@patch("api.main.check_llm_reachability")
def test_health_with_deep_when_unreachable(mock_reachability, mock_config, mock_get_retriever):
    mock_ret = MagicMock()
    mock_ret.count.return_value = 207
    mock_get_retriever.return_value = mock_ret

    mock_config.return_value = (True, "Valid config")
    mock_reachability.return_value = (False, "Timeout connecting to provider")
    response = client.get("/health?deep=true")
    assert response.status_code == 503
    data = response.json()
    assert data["llm_reachable"] is False
    assert data["status"] == "degraded"


@patch("api.main.get_retriever")
def test_list_schemes(mock_get_retriever):
    mock_ret = MagicMock()
    mock_ret.get_indexed_schemes.return_value = [
        {
            "scheme": "PM-KISAN",
            "title": "PM-KISAN Operational Guidelines",
            "source_file": "pm_kisan_guidelines.pdf",
            "url": "https://pmkisan.gov.in",
            "doc_date": "2020-03-29",
            "date_accessed": "2026-11-02",
            "chunks_count": 55,
        }
    ]
    mock_get_retriever.return_value = mock_ret

    response = client.get("/schemes")
    assert response.status_code == 200
    schemes = response.json()
    assert isinstance(schemes, list)
    assert len(schemes) >= 1
    sample = schemes[0]
    assert "scheme" in sample
    assert "doc_date" in sample
    assert "date_accessed" in sample


def test_ask_validation_too_short():
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422
    data = response.json()
    assert "request_id" in data
    assert data["error"] == "Invalid request input."
    assert "at least 3 characters" in data["detail"]


def test_ask_validation_too_long():
    long_question = "What is PM KISAN? " * 35  # > 500 chars
    response = client.post("/ask", json={"question": long_question})
    assert response.status_code == 422
    data = response.json()
    assert "not exceed 500 characters" in data["detail"]


@patch("api.main.get_retriever")
@patch("api.main.run_workflow")
def test_ask_valid_request_mocked(mock_run_workflow, mock_get_retriever):
    mock_ret = MagicMock()
    mock_ret.count.return_value = 207
    mock_get_retriever.return_value = mock_ret
    mock_run_workflow.return_value = {
        "question": "What is PM-KISAN?",
        "intent": "benefits",
        "scheme": "PM-KISAN",
        "is_personal": False,
        "current_query": "What is PM-KISAN?",
        "chunks": [],
        "draft_answer": "PM-KISAN grants Rs 6,000 annually.",
        "verdict": "SUPPORTED",
        "verdict_reason": "Passage matches.",
        "unsupported_claims": [],
        "retries": 0,
        "final_answer": "PM-KISAN grants Rs 6,000 annually.",
        "sources": [
            {
                "scheme": "PM-KISAN",
                "title": "PM-KISAN Guidelines",
                "page": 1,
                "url": "https://pmkisan.gov.in",
                "snippet": "Rs 6,000 annually",
                "doc_date": "2024-02-15",
                "date_accessed": "2024-10-15",
            }
        ],
        "missing_info": [],
        "abstained": False,
        "trace": [{"node": "classify_intent", "summary": "benefits", "timestamp": "2024-10-30T12:00:00"}],
    }

    secret_question = "What is PM-KISAN eligibility for a secret person?"
    response = client.post("/ask", json={"question": secret_question})
    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "PM-KISAN grants Rs 6,000 annually."
    assert data["abstained"] is False
    assert data["verdict"] == "SUPPORTED"
    assert data["intent"] == "benefits"
    assert data["scheme"] == "PM-KISAN"
    assert len(data["sources"]) == 1
    assert data["latency_s"] >= 0.0
    assert "X-Request-ID" in response.headers

    # Verify that question text was NOT written to logs/api.log
    config = get_config()
    log_file = config.paths.logs_path / "api.log"
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            assert secret_question not in log_content
            # Verify length was logged
            assert f"len={len(secret_question)}" in log_content
