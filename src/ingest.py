"""Document ingestion and vector store indexing for TrustYojana.

Implements Experiment 5:
- Reads PDFs (PyMuPDF), HTML, and TXT files from data/raw/
- Maps with data/sources.csv
- Cleans headers, footers, page numbers, line wraps
- Saves cleaned text in data/processed/
- Chunks with RecursiveCharacterTextSplitter
- Generates embeddings with SentenceTransformers
- Stores in persistent ChromaDB at chroma_db/
- Content-hash based incremental updates (skips unchanged, updates changed, adds new)
- Supports --reset flag to delete and rebuild the collection
- Prints a clear summary
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf  # PyMuPDF

from src.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file for change detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash of text content."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Clean extracted text.
    
    - Removes repeated headers/footers and page number patterns
    - Fixes hyphenated line wraps (e.g. 'inst-\nallments' -> 'installments')
    - Merges soft line breaks within sentences while keeping paragraph breaks
    - Collapses extra whitespace
    """
    if not text:
        return ""

    # Fix broken hyphenated line breaks (word continuation)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Remove standalone page numbers (e.g., "1", "Page 1 of 3", "- 2 -")
        if re.match(r"^(page\s+\d+(\s+of\s+\d+)?|\d+|-\s*\d+\s*-)$", stripped, re.IGNORECASE):
            continue

        # Remove typical repetitive circular headers / footers
        if re.search(r"^(government of india\s*\|.*|.*guidelines\s*-\s*page\s*\d+.*|circular no\..*)$", stripped, re.IGNORECASE):
            continue

        # Skip horizontal dividers
        if re.match(r"^[-=_*~]{3,}$", stripped):
            continue

        cleaned_lines.append(stripped)

    # Reassemble lines
    raw_joined = "\n".join(cleaned_lines)

    # Merge single line breaks (soft wraps) within paragraphs, preserve double line breaks
    paragraphs = raw_joined.split("\n\n")
    processed_paragraphs = []
    for para in paragraphs:
        # Replace single newline inside paragraph with space
        p_clean = " ".join(part.strip() for part in para.split("\n") if part.strip())
        # Collapse multiple spaces
        p_clean = re.sub(r"[ \t]+", " ", p_clean).strip()
        if p_clean:
            processed_paragraphs.append(p_clean)

    return "\n\n".join(processed_paragraphs)


def load_sources_metadata(sources_csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load metadata mapping from data/sources.csv."""
    if not sources_csv_path.exists():
        logger.warning(f"sources.csv not found at {sources_csv_path}")
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    with open(sources_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            if filename:
                mapping[filename] = {
                    "scheme": row.get("scheme", "Unknown Scheme").strip(),
                    "title": row.get("title", filename).strip(),
                    "url": row.get("url", "").strip(),
                    "doc_date": row.get("doc_date", "unknown").strip() or "unknown",
                    "date_accessed": row.get("date_accessed", "unknown").strip() or "unknown",
                }
    return mapping


def extract_pages_from_file(filepath: Path) -> List[Tuple[int, str]]:
    """Extract (page_number, text) tuples from PDF, TXT, or HTML files."""
    suffix = filepath.suffix.lower()
    pages: List[Tuple[int, str]] = []

    if suffix == ".pdf":
        doc = pymupdf.open(str(filepath))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append((page_num + 1, text))
        doc.close()

    elif suffix in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        pages.append((1, content))

    elif suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        pages.append((1, text))

    else:
        logger.warning(f"Unsupported file format for {filepath}")

    return pages


class IngestionManager:
    """Manages document ingestion, hashing, chunking, and ChromaDB persistence."""

    def __init__(self):
        self.config = get_config()
        self.chroma_dir = self.config.paths.chroma_db_path
        self.raw_dir = self.config.paths.raw_data_path
        self.processed_dir = self.config.paths.processed_data_path
        self.sources_csv = self.config.paths.sources_csv_path
        self.manifest_path = self.chroma_dir / "ingest_manifest.json"

        # Ensure directories exist
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Setup Sentence Transformers embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embeddings.model_name,
            device=self.config.embeddings.device,
        )

        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.config.paths.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}. Starting fresh.")
        return {}

    def _save_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def reset_collection(self):
        """Delete collection and manifest, rebuilding clean."""
        logger.info("Resetting ChromaDB collection and manifest...")
        try:
            self.client.delete_collection(self.config.paths.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.config.paths.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.manifest = {}
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        logger.info("ChromaDB collection reset completed.")

    def delete_chunks_for_file(self, filename: str):
        """Delete all chunks associated with a source_file from the collection."""
        try:
            self.collection.delete(where={"source_file": filename})
            logger.info(f"Deleted existing chunks for {filename}")
        except Exception as e:
            logger.debug(f"Note on deleting chunks for {filename}: {e}")

    def ingest_all(self, reset: bool = False) -> Dict[str, int]:
        """Ingest all files from data/raw/ with incremental hash detection."""
        if reset:
            self.reset_collection()

        sources_meta = load_sources_metadata(self.sources_csv)
        raw_files = [
            f for f in self.raw_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md", ".html", ".htm")
        ]

        summary = {
            "indexed": 0,
            "skipped": 0,
            "updated": 0,
            "total_pages": 0,
            "total_chunks": 0,
        }

        for file_path in raw_files:
            filename = file_path.name
            current_hash = compute_file_hash(file_path)
            meta = sources_meta.get(filename, {
                "scheme": "General",
                "title": filename,
                "url": "",
                "doc_date": "unknown",
                "date_accessed": "unknown",
            })

            # Check if unchanged
            prev_info = self.manifest.get(filename)
            if prev_info and prev_info.get("hash") == current_hash and not reset:
                logger.info(f"[SKIP] Unchanged file: {filename}")
                summary["skipped"] += 1
                summary["total_pages"] += prev_info.get("pages", 0)
                summary["total_chunks"] += prev_info.get("chunks", 0)
                continue

            is_update = bool(prev_info)
            if is_update:
                logger.info(f"[UPDATE] Changed file detected: {filename}")
                self.delete_chunks_for_file(filename)
            else:
                logger.info(f"[INDEX] New file detected: {filename}")

            # Extract and clean
            raw_pages = extract_pages_from_file(file_path)
            if not raw_pages:
                logger.warning(f"No text extracted from {filename}")
                continue

            cleaned_full_doc = []
            file_chunks = []
            file_metadatas = []
            file_ids = []

            for page_num, raw_text in raw_pages:
                cleaned_text = clean_text(raw_text)
                if not cleaned_text:
                    continue

                cleaned_full_doc.append(f"--- PAGE {page_num} ---\n{cleaned_text}")

                # Chunk this page
                page_chunks = self.splitter.split_text(cleaned_text)
                for chunk_idx, chunk_text in enumerate(page_chunks):
                    chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"
                    chunk_hash = compute_text_hash(chunk_text)

                    chunk_meta = {
                        "scheme": meta["scheme"],
                        "title": meta["title"],
                        "source_file": filename,
                        "page": int(page_num),
                        "url": meta["url"],
                        "doc_date": meta["doc_date"],
                        "date_accessed": meta["date_accessed"],
                        "chunk_id": chunk_id,
                        "content_hash": chunk_hash,
                    }

                    file_chunks.append(chunk_text)
                    file_metadatas.append(chunk_meta)
                    file_ids.append(chunk_id)

            # Save cleaned document to data/processed/ for inspection
            processed_file = self.processed_dir / f"{file_path.stem}_cleaned.txt"
            with open(processed_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(cleaned_full_doc))

            # Add to Chroma collection
            if file_chunks:
                # Add in batches to avoid any memory or network limits
                batch_size = 64
                for i in range(0, len(file_chunks), batch_size):
                    end_i = i + batch_size
                    self.collection.add(
                        ids=file_ids[i:end_i],
                        documents=file_chunks[i:end_i],
                        metadatas=file_metadatas[i:end_i],
                    )

            # Update manifest
            num_pages = len(raw_pages)
            num_chunks = len(file_chunks)
            self.manifest[filename] = {
                "hash": current_hash,
                "pages": num_pages,
                "chunks": num_chunks,
                "scheme": meta["scheme"],
            }
            self._save_manifest()

            if is_update:
                summary["updated"] += 1
            else:
                summary["indexed"] += 1
            summary["total_pages"] += num_pages
            summary["total_chunks"] += num_chunks

        # Print human-readable summary
        print("\n" + "=" * 55)
        print("          TRUSTYOJANA INGESTION SUMMARY")
        print("=" * 55)
        print(f"  Documents Indexed (New):   {summary['indexed']}")
        print(f"  Documents Updated:         {summary['updated']}")
        print(f"  Documents Skipped:         {summary['skipped']}")
        print(f"  Total Pages Processed:     {summary['total_pages']}")
        print(f"  Total Chunks in DB:        {self.collection.count()}")
        print("=" * 55 + "\n")

        return summary


def main():
    parser = argparse.ArgumentParser(description="TrustYojana Ingestion Tool")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild vector collection from scratch")
    args = parser.parse_args()

    manager = IngestionManager()
    manager.ingest_all(reset=args.reset)


if __name__ == "__main__":
    main()
