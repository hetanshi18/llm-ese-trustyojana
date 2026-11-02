# TrustYojana (v2) 🇮🇳

> **Verifiable, Citation-Backed Question-Answering for Indian Government Schemes**
> *Built with LangChain, LangGraph, ChromaDB, FastAPI, and Streamlit*

**TrustYojana** is an end-to-end RAG system for Indian government welfare schemes (**PM-KISAN**, **Ayushman Bharat PM-JAY**, **National Means-cum-Merit Scholarship Scheme (NMMSS)**, and **PM SVANidhi**).

TrustYojana answers queries **only when retrieved official scheme documents strictly support the facts**. If the evidence is insufficient or contradicted, the system safely falls back with an explicit, helpful message.

---

## 🔒 Privacy, Personal Data & Limitations

### 1. Zero Sensitive PII Interrogation
TrustYojana is **stateless and single-turn**. Under the coursework specification, it explicitly does **not** implement multi-turn "slot-filling" forms and **never asks users for sensitive credentials or personal identifiers** (such as Aadhaar numbers, PAN, or bank account details).

### 2. Personal Eligibility Assessment Handling
When a user voluntarily asks about their personal circumstances (e.g., *"Am I eligible for PM-KISAN? I own 2 acres of land"*):
- The system evaluates the criteria stated in the user's message as **Met** or **Not met**.
- Any unstated criteria are marked **Unknown**.
- Missing items are presented under an **Information Needed** list.
- A mandatory legal notice is always displayed: *"This is not an official eligibility decision; confirm with the official scheme website or office."*

### 3. Cloud Provider vs. Local Privacy (Why Ollama Is Recommended)
- **Local Ollama Execution:** All embedding and LLM reasoning occur 100% locally on your computer. **No personal query text or details are ever transmitted over the internet.**
- **Cloud LLM APIs (OpenAI / Anthropic / Gemini):** If configured with a cloud provider, the user's prompt text is sent via HTTPS to external third-party servers.
- **Server Logging:** The FastAPI backend implements strict privacy-preserving logging. User question text is **never stored or logged**; only question length, a SHA-256 hash prefix, intent, scheme, verdict, retries, and latency are recorded in `logs/api.log`.

---

## 🏗️ Architecture & Integrated Workflow

TrustYojana is developed as **one integrated application** incorporating the three lab experiment components:
- **Experiment 5 (Document-based RAG):** Document ingestion (PDF via PyMuPDF; HTML/TXT), text cleaning, chunking (`RecursiveCharacterTextSplitter`), embeddings (`BAAI/bge-small-en-v1.5`), persistent ChromaDB with file-hash change detection, and basic retrieval.
- **Experiment 6 (LangChain / LangGraph Workflow):** StateGraph workflow coordinating intent classification, scheme detection, personal eligibility analysis, claim-level evidence checking (`SUPPORTED` / `NOT_ENOUGH_EVIDENCE` / `CONTRADICTED`), query rewriting with retry loops, and safe fallback.
- **Experiment 7 (FastAPI & Streamlit):** High-performance asynchronous REST API with validation, middleware-based request tracking (`X-Request-ID`), privacy-preserving logging, health monitoring (independent vector DB and LLM connectivity), and a Streamlit chat UI.

```mermaid
flowchart TD
    User([User Question]) --> API[FastAPI: POST /ask]
    API --> Intent[Node 1: Classify Intent & Scheme]
    
    Intent -->|out_of_scope| Fallback[Node 7: Fallback Abstained Response]
    Intent -->|eligibility / benefits / etc.| Retrieve[Node 2: Retrieve ChromaDB Chunks]
    
    Retrieve --> Generate[Node 3: Draft Answer Grounded in Context]
    Generate --> Check[Node 4: Claim-Level Evidence Grader]
    
    Check -->|SUPPORTED| Finalize[Node 6: Finalize Answer + Citations]
    Check -->|NOT_ENOUGH_EVIDENCE or CONTRADICTED<br>and retries < max_retries| Rewrite[Node 5: Rewrite Query & Increment Retry]
    Rewrite --> Retrieve
    Check -->|retries >= max_retries| Fallback
    
    Finalize --> Response([AskResponse to Streamlit])
    Fallback --> Response
```

---

## 📂 Project Structure

```text
trustyojana/
  ├── LICENSE                 # MIT Open Source License
  ├── README.md               # Complete project documentation and guide
  ├── requirements.txt        # Python 3.11 dependencies
  ├── config.yaml             # Central configuration (models, chunking, paths, ports)
  ├── .env.example            # Template for environment variables and API keys
  ├── data/
  │   ├── raw/                # Official scheme PDFs (PM-KISAN, Ayushman Bharat, NMMSS, PM SVANidhi)
  │   ├── processed/          # Cleaned text files saved for inspection
  │   └── sources.csv         # Metadata mapping (filename, scheme, title, url, doc_date, date_accessed)
  ├── chroma_db/              # Persistent ChromaDB vector database
  ├── logs/
  │   └── api.log             # Rotating API log (privacy-preserving)
  ├── src/
  │   ├── __init__.py
  │   ├── config.py           # Configuration loader (.env and config.yaml)
  │   ├── llm.py              # Provider-agnostic LLM wrapper (Ollama, OpenAI, Gemini, Anthropic)
  │   ├── ingest.py           # Document cleaning, chunking, hashing, and Chroma indexing
  │   ├── retriever.py        # Top-k retrieval with metadata filtering
  │   ├── rag_chain.py        # Basic retrieve-and-answer pipeline (Exp 5)
  │   ├── prompts.py          # Centralized prompts
  │   └── workflow.py         # LangGraph StateGraph workflow (Exp 6)
  ├── api/
  │   ├── __init__.py
  │   ├── main.py             # FastAPI app with health checks, /ask, /schemes (Exp 7)
  │   └── schemas.py          # Pydantic models (AskRequest, AskResponse, HealthResponse, etc.)
  ├── app/
  │   └── streamlit_app.py    # Streamlit chat interface (Exp 7)
  └── tests/
      ├── __init__.py
      ├── test_ingest.py      # Tests for text cleaning, chunking, and hash skip logic
      ├── test_workflow.py    # Tests for JSON parsing, routing, and mock workflow
      └── test_api.py         # FastAPI TestClient tests for validation, health, and privacy logging
```

---

## 📥 Supplying & Ingesting Official Documents

Under Experiment 5, source documents in `data/raw/` are supplied by the user from authentic government sources.

### Official Source Documents Ingested:
1. **PM-KISAN (`pm_kisan_guidelines.pdf`):**
   - Source: Ministry of Agriculture & Farmers Welfare (`https://pmkisan.gov.in`)
   - Document: *PM-KISAN Operational Guidelines (Revised 2020)*
   - Covers: Rs. 6,000/year DBT assistance in 3 installments, land record rules, and the complete institutional/professional/pension exclusion list.
2. **Ayushman Bharat PM-JAY (`ayushman_pmjay_guidelines.pdf`):**
   - Source: National Health Authority (NHA) / MoHFW (`https://pmjay.gov.in`)
   - Document: *Beneficiary Identification Guidelines (AB PM-JAY)*
   - Covers: Rural SECC D1–D7 deprivation criteria, 11 urban occupational categories, Ayushman Golden card printing, and empanelled hospital procedures.
3. **NMMSS Scholarship (`nmmss_guidelines.pdf` & `nmmss_faq.pdf`):**
   - Source: Department of School Education & Literacy, Ministry of Education (`https://scholarships.gov.in`)
   - Documents: *Revised Scheme Guidelines & Frequently Asked Questions*
   - Covers: Class VIII entrance, 55% mark cutoff (50% for SC/ST), Rs. 3,50,000 parental income cap, Rs. 12,000 annual scholarship, and NSP portal submission.
4. **PM SVANidhi (`pm_svanidhi_guidelines.pdf`):**
   - Source: Ministry of Housing and Urban Affairs (MoHUA) (`https://pmsvanidhi.mohua.gov.in`)
   - Document: *PM SVANidhi Scheme Guidelines*
   - Covers: Urban street vendors, 1st tranche (Rs. 10,000), 2nd tranche (Rs. 20,000), 3rd tranche (Rs. 50,000), 7% interest subsidy, and digital transaction cashbacks.

### Running Ingestion
Place your PDFs into `data/raw/` and verify their corresponding rows in `data/sources.csv`. Then run:

```bash
python -m src.ingest
```

- **Incremental Hash Detection:** If files are unchanged, re-running `python -m src.ingest` skips them in seconds.
- **Clean Rebuild:** To completely wipe and rebuild the vector database:
  ```bash
  python -m src.ingest --reset
  ```

---

## ⚙️ Installation & Running

### 1. Virtual Environment & Dependencies
```powershell
# Activate venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
By default, TrustYojana is configured to run with your local **Ollama** instance:
```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:0.5b
OLLAMA_BASE_URL=http://localhost:11434

API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_API_URL=http://127.0.0.1:8000
```

### 3. Launch Applications

**Terminal 1: FastAPI Backend**
```powershell
.\venv\Scripts\activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: Streamlit Frontend**
```powershell
.\venv\Scripts\activate
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔬 Automated Testing

Run the automated test suite with pytest:

```powershell
.\venv\Scripts\pytest -v tests
```
All 23 unit tests verify text cleaning, hash skip logic, evidence JSON parsing, graph edge routing, input validation (422), deep health checks, and privacy-preserving logging.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
