"""Streamlit UI for TrustYojana.

Implements Experiment 7:
- Communicates purely via HTTP requests to FastAPI (never imports workflow directly)
- Chat-style interface with in-session history
- Distinct display for grounded answers vs abstained fallbacks
- Badges for Intent, Target Scheme, and Evidence Verdict (SUPPORTED / NOT_ENOUGH_EVIDENCE / CONTRADICTED)
- Specialized Personal Eligibility view:
  * Criteria evaluation (Met / Not met / Unknown)
  * Missing information list
  * Non-official decision caveat notice
- Expandable Sources accordion with scheme, title, page, snippet, URL, doc_date, and date_accessed
- Freshness caption under every response with last-accessed date
- Expandable Workflow Trace with latency and retry count
- Sidebar with:
  * Backend API Health monitor (Vector DB, LLM Config, LLM Reachability)
  * 'Check LLM connection' button (?deep=true)
  * Indexed schemes list with access dates
  * Clear chat button
  * Research prototype disclaimer
- Clickable sample questions (answerable, personal eligibility, unanswerable, out-of-scope)
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests
import streamlit as st
import yaml

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="TrustYojana - Grounded Welfare Schemes QA",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=60)
def load_api_url() -> str:
    """Read API URL from config.yaml or environment variable."""
    env_url = os.getenv("STREAMLIT_API_URL")
    if env_url:
        return env_url.rstrip("/")

    # Try config.yaml in workspace root
    config_file = "config.yaml"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                url = cfg.get("streamlit", {}).get("api_url")
                if url:
                    return url.rstrip("/")
        except Exception:
            pass

    return "http://127.0.0.1:8000"


API_BASE_URL = load_api_url()


# ------------------------------------------------------------------------------
# API Client Helpers
# ------------------------------------------------------------------------------

def fetch_health(deep: bool = False) -> Optional[Dict[str, Any]]:
    """Query GET /health endpoint."""
    url = f"{API_BASE_URL}/health"
    try:
        resp = requests.get(url, params={"deep": "true" if deep else "false"}, timeout=15)
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def fetch_schemes() -> List[Dict[str, Any]]:
    """Query GET /schemes endpoint."""
    url = f"{API_BASE_URL}/schemes"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return []


def ask_question(question: str) -> Dict[str, Any]:
    """Send POST /ask request to backend API."""
    url = f"{API_BASE_URL}/ask"
    try:
        # 180s timeout allows multi-node workflow loops to complete even on CPU inference
        resp = requests.post(url, json={"question": question}, timeout=180)
        data = resp.json()
        return {"status_code": resp.status_code, "data": data}
    except requests.exceptions.Timeout:
        return {
            "status_code": 504,
            "data": {
                "error": "Gateway Timeout",
                "detail": "The request timed out while waiting for LLM/retrieval. Please try again.",
            },
        }
    except requests.exceptions.ConnectionError:
        return {
            "status_code": 503,
            "data": {
                "error": "Backend Unavailable",
                "detail": f"Could not connect to TrustYojana backend at {API_BASE_URL}. Ensure 'uvicorn api.main:app' is running.",
            },
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": 500,
            "data": {
                "error": "Request Error",
                "detail": str(e),
            },
        }


# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "deep_health" not in st.session_state:
    st.session_state.deep_health = None

if "prompt_to_submit" not in st.session_state:
    st.session_state.prompt_to_submit = None


# ------------------------------------------------------------------------------
# Sidebar Component
# ------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏛️ **TrustYojana**")
    st.caption("Verifiable RAG for Indian Government Schemes")
    st.divider()

    # Backend Connection & Health
    st.markdown("### 📡 **Backend Health**")
    st.caption(f"Connecting to: `{API_BASE_URL}`")

    health_info = st.session_state.deep_health or fetch_health(deep=False)

    if health_info is None:
        st.error("🔴 **Backend Offline**")
        st.caption("FastAPI server is not responding.")
    else:
        v_ok = health_info.get("vector_db_ok", False)
        l_cfg_ok = health_info.get("llm_config_ok", False)
        l_reach = health_info.get("llm_reachable")
        chunks = health_info.get("indexed_chunks", 0)

        # Vector DB Status
        if v_ok:
            st.markdown(f"🟢 **Vector Store:** Ready ({chunks} chunks)")
        else:
            st.markdown("🔴 **Vector Store:** Empty / Not initialized")

        # LLM Config Status
        if l_cfg_ok:
            st.markdown("🟢 **LLM Config:** Configured")
        else:
            st.markdown("🔴 **LLM Config:** Missing API Key")

        # LLM Reachability Status
        if l_reach is True:
            st.markdown("🟢 **LLM Connectivity:** Live & Verified")
        elif l_reach is False:
            st.markdown("🔴 **LLM Connectivity:** Unreachable")
        else:
            st.markdown("⚪ **LLM Connectivity:** Not probed")

    if st.button("🔍 Check LLM Connection", help="Sends a lightweight probe to the LLM (cached for 60s)"):
        with st.spinner("Pinging LLM..."):
            st.session_state.deep_health = fetch_health(deep=True)
            st.rerun()

    st.divider()

    # Indexed Schemes Directory
    st.markdown("### 📚 **Indexed Schemes**")
    schemes_data = fetch_schemes()
    if schemes_data:
        for s in schemes_data:
            with st.expander(f"🔹 {s.get('scheme', 'Scheme')}"):
                st.markdown(f"**Title:** {s.get('title')}")
                st.markdown(f"**Doc Date:** `{s.get('doc_date')}`")
                st.markdown(f"**Date Accessed:** `{s.get('date_accessed')}`")
                st.markdown(f"**Chunks:** {s.get('chunks_count')}")
                if s.get("url"):
                    st.markdown(f"[Official Portal]({s.get('url')})")
    else:
        st.caption("No schemes found or backend offline.")

    st.divider()

    # Clear Chat & Prototype Disclaimer
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.deep_health = None
        st.session_state.prompt_to_submit = None
        st.rerun()

    st.markdown("---")
    st.info(
        "⚠️ **Research prototype. Not official advice.** "
        "Verify on the scheme's official website."
    )


# ------------------------------------------------------------------------------
# Main Page Header & Example Buttons
# ------------------------------------------------------------------------------

st.title("🇮🇳 TrustYojana")
st.markdown(
    "Ask factual questions about Indian government welfare schemes (**PM-KISAN**, "
    "**Ayushman Bharat PM-JAY**, **NMMSS Scholarship**, **PM SVANidhi**). "
    "Answers are provided **only when fully backed by verified official documents**; "
    "otherwise, TrustYojana safely abstains."
)

st.markdown("#### 💡 **Example Questions**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📋 PM-KISAN Benefits", use_container_width=True):
        st.session_state.prompt_to_submit = "What is the financial assistance provided under PM-KISAN and how is it disbursed?"

with col2:
    if st.button("👤 Personal Eligibility", use_container_width=True):
        st.session_state.prompt_to_submit = "Am I eligible for PM-KISAN? I own 2 acres of land and my family does farming."

with col3:
    if st.button("🚫 Unanswerable Query", use_container_width=True):
        st.session_state.prompt_to_submit = "Can I purchase a tractor or agricultural vehicle using an Ayushman Bharat card?"

with col4:
    if st.button("🌐 Out of Scope", use_container_width=True):
        st.session_state.prompt_to_submit = "How do I make authentic Hyderabadi biryani?"


# ------------------------------------------------------------------------------
# Chat History Display Function
# ------------------------------------------------------------------------------

def render_assistant_response(msg_data: Dict[str, Any]):
    """Render an assistant message with badges, criteria, sources, freshness, and trace."""
    abstained = msg_data.get("abstained", False)
    answer = msg_data.get("answer", "")
    intent = msg_data.get("intent", "general_info")
    scheme = msg_data.get("scheme", "General")
    verdict = msg_data.get("verdict", "SUPPORTED")
    is_personal = msg_data.get("is_personal", False)
    missing_info = msg_data.get("missing_info", [])
    sources = msg_data.get("sources", [])
    trace = msg_data.get("trace", [])
    retries = msg_data.get("retries", 0)
    latency_s = msg_data.get("latency_s", 0.0)

    # Top Badges
    badge_cols = st.columns([1, 1, 1, 3])
    with badge_cols[0]:
        st.markdown(f"🏷️ **Intent:** `{intent}`")
    with badge_cols[1]:
        st.markdown(f"🏛️ **Scheme:** `{scheme}`")
    with badge_cols[2]:
        verdict_color = "green" if verdict == "SUPPORTED" else ("orange" if verdict == "NOT_ENOUGH_EVIDENCE" else "red")
        st.markdown(f"⚖️ **Evidence:** :{verdict_color}[`{verdict}`]")

    st.markdown("<br>", unsafe_allow_html=True)

    # Primary Answer Rendering
    if abstained:
        st.warning(f"⚠️ **Abstained:**\n\n{answer}")
    else:
        st.markdown(answer)

    # Personal Eligibility Callout (if personal question)
    if is_personal:
        st.markdown("---")
        st.markdown("### 📝 **Personal Eligibility Assessment**")
        st.caption("Stateless preliminary review based strictly on user-stated details:")

        if missing_info:
            st.markdown("##### 🔍 **Information Needed:**")
            for item in missing_info:
                st.markdown(f"- ❓ {item}")

        st.info("ℹ️ **Notice:** This is not an official eligibility decision; confirm with the official scheme website or office.")

    # Freshness Notice
    latest_accessed = "unknown"
    if sources:
        dates = [s.get("date_accessed") for s in sources if s.get("date_accessed") and s.get("date_accessed") != "unknown"]
        if dates:
            latest_accessed = dates[0]

    st.caption(
        f"🕒 *Based on documents last accessed on **{latest_accessed}**. "
        "Scheme rules change, so verify on the official website.*"
    )

    # Expandable Sources Section
    if sources:
        with st.expander(f"📚 Cited Sources ({len(sources)} documents)"):
            for idx, src in enumerate(sources, 1):
                st.markdown(f"**Source {idx}: {src.get('title', 'Document')}** (Page {src.get('page', 1)})")
                st.markdown(f"- **Scheme:** {src.get('scheme')}")
                st.markdown(f"- **Document Published/Updated:** `{src.get('doc_date')}` | **Accessed:** `{src.get('date_accessed')}`")
                if src.get("url"):
                    st.markdown(f"- **URL:** [{src.get('url')}]({src.get('url')})")
                snippet = src.get("snippet", "")
                if snippet:
                    st.markdown(f"> *\"{snippet}\"*")
                st.markdown("---")

    # Expandable Workflow Trace
    with st.expander(f"⚙️ Execution Trace (Latency: {latency_s}s | Retries: {retries})"):
        st.markdown(f"**Retries Performed:** `{retries}` | **Total Latency:** `{latency_s}s`")
        for step in trace:
            node = step.get("node", "step")
            summary = step.get("summary", "")
            ts = step.get("timestamp", "")
            time_part = ts.split("T")[-1][:8] if "T" in ts else ts
            st.markdown(f"- **`{node}`** ({time_part}): {summary}")


# Render previous messages in session
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant_response(msg["data"])


# ------------------------------------------------------------------------------
# Chat Input & Execution
# ------------------------------------------------------------------------------

user_input = st.chat_input("Ask about PM-KISAN, Ayushman Bharat, NMMSS Scholarship, or PM SVANidhi...")

# Check if an example button triggered a submission
active_prompt = None
if st.session_state.prompt_to_submit:
    active_prompt = st.session_state.prompt_to_submit
    st.session_state.prompt_to_submit = None
elif user_input:
    active_prompt = user_input

if active_prompt:
    # 1. Render User Message
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # 2. Invoke Backend API
    with st.chat_message("assistant"):
        with st.spinner("Retrieving official scheme documents and verifying evidence..."):
            response = ask_question(active_prompt)

        status_code = response["status_code"]
        data = response["data"]

        if status_code == 200:
            render_assistant_response(data)
            st.session_state.messages.append({"role": "assistant", "data": data})

        elif status_code == 422:
            detail = data.get("detail", "Input must be between 3 and 500 characters.")
            error_text = f"❌ **Validation Error:** {detail}"
            st.error(error_text)
            st.session_state.messages.append({
                "role": "assistant",
                "data": {
                    "answer": error_text,
                    "abstained": True,
                    "verdict": "ERROR",
                    "intent": "invalid_input",
                    "scheme": "None",
                    "is_personal": False,
                    "sources": [],
                    "trace": [],
                    "retries": 0,
                    "latency_s": 0.0,
                },
            })

        elif status_code == 503:
            detail = data.get("detail", "TrustYojana backend or LLM service is temporarily unavailable.")
            error_text = f"⚠️ **Service Unavailable (503):** {detail}"
            st.error(error_text)
            st.session_state.messages.append({
                "role": "assistant",
                "data": {
                    "answer": error_text,
                    "abstained": True,
                    "verdict": "ERROR",
                    "intent": "service_unavailable",
                    "scheme": "None",
                    "is_personal": False,
                    "sources": [],
                    "trace": [],
                    "retries": 0,
                    "latency_s": 0.0,
                },
            })

        elif status_code == 504:
            error_text = "⏱️ **Request Timeout (504):** Upstream model response timed out. Please retry in a few moments."
            st.error(error_text)
            st.session_state.messages.append({
                "role": "assistant",
                "data": {
                    "answer": error_text,
                    "abstained": True,
                    "verdict": "TIMEOUT",
                    "intent": "timeout",
                    "scheme": "None",
                    "is_personal": False,
                    "sources": [],
                    "trace": [],
                    "retries": 0,
                    "latency_s": 0.0,
                },
            })

        else:
            detail = data.get("detail", "An unexpected error occurred.")
            error_text = f"❌ **Error ({status_code}):** {detail}"
            st.error(error_text)
            st.session_state.messages.append({
                "role": "assistant",
                "data": {
                    "answer": error_text,
                    "abstained": True,
                    "verdict": "ERROR",
                    "intent": "error",
                    "scheme": "None",
                    "is_personal": False,
                    "sources": [],
                    "trace": [],
                    "retries": 0,
                    "latency_s": 0.0,
                },
            })
