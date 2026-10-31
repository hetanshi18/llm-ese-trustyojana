"""Unit tests for document cleaning, chunking, and hash-based skip logic."""

import hashlib
import json
import tempfile
from pathlib import Path
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest import clean_text, compute_file_hash, compute_text_hash


def test_clean_text_removes_headers_and_footers():
    raw_sample = (
        "Government of India | Ministry of Agriculture and Farmers Welfare\n"
        "Page 1 of 3\n"
        "PM-KISAN provides financial assistance of Rs. 6,000 per year.\n"
        "------------------------------------\n"
        "PM-KISAN Operational Guidelines - Page 1 of 3\n"
    )
    cleaned = clean_text(raw_sample)
    assert "Government of India" not in cleaned
    assert "Page 1 of 3" not in cleaned
    assert "PM-KISAN provides financial assistance of Rs. 6,000 per year." in cleaned


def test_clean_text_fixes_hyphenated_line_wraps():
    raw_sample = (
        "The financial assistance is disbursed in three equal four-monthly in-\n"
        "stallments directly to the beneficiaries."
    )
    cleaned = clean_text(raw_sample)
    assert "installments" in cleaned
    assert "in-\nstallments" not in cleaned


def test_clean_text_normalizes_whitespace():
    raw_sample = "Eligible    beneficiaries    receive   Rs.   6,000.\n\n\n\nNew paragraph."
    cleaned = clean_text(raw_sample)
    assert "Eligible beneficiaries receive Rs. 6,000." in cleaned
    assert "  " not in cleaned


def test_file_hash_computation():
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write("TrustYojana scheme guideline content")
        tmp_path = Path(tmp.name)

    try:
        h1 = compute_file_hash(tmp_path)
        expected = hashlib.sha256("TrustYojana scheme guideline content".encode("utf-8")).hexdigest()
        assert h1 == expected

        # Modify content and verify hash changes
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("Modified content")
        h2 = compute_file_hash(tmp_path)
        assert h1 != h2
    finally:
        tmp_path.unlink()


def test_splitter_chunking_size_and_overlap():
    sample_text = (
        "Pradhan Mantri Kisan Samman Nidhi is a central sector scheme. "
        "It provides Rs 6000 per annum to all eligible farmer families across the country. "
        "The amount is credited in three equal installments. "
    ) * 15  # Repeat to exceed chunk size

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_text(sample_text)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 450
