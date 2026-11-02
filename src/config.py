"""Configuration loader for TrustYojana.

Loads settings from config.yaml and environment variables (.env).
All parameters (chunk size, overlap, top-k, model names, paths, retries)
are centralized here without hardcoding.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Set offline mode for cached HuggingFace models to eliminate network retry delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Load .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


class AppConfig(BaseModel):
    name: str = "TrustYojana"
    version: str = "2.0.0"
    log_level: str = "INFO"
    log_file: str = "logs/api.log"


class LLMConfig(BaseModel):
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    request_timeout: int = 30
    max_retries: int = 2


class EmbeddingsConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"


class RAGConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    max_retries: int = 2


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    sources_csv: str = "data/sources.csv"
    chroma_db_dir: str = "chroma_db"
    collection_name: str = "trustyojana_schemes"
    logs_dir: str = "logs"

    def get_abs_path(self, relative_path: str) -> Path:
        return BASE_DIR / relative_path

    @property
    def raw_data_path(self) -> Path:
        return self.get_abs_path(self.raw_data_dir)

    @property
    def processed_data_path(self) -> Path:
        return self.get_abs_path(self.processed_data_dir)

    @property
    def sources_csv_path(self) -> Path:
        return self.get_abs_path(self.sources_csv)

    @property
    def chroma_db_path(self) -> Path:
        return self.get_abs_path(self.chroma_db_dir)

    @property
    def logs_path(self) -> Path:
        return self.get_abs_path(self.logs_dir)


class APIConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    cors_origins: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]


class StreamlitConfig(BaseModel):
    api_url: str = Field(default_factory=lambda: os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000"))


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file and apply env overrides."""
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"

    yaml_data: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    config = Config(**yaml_data)

    # Apply environment variable overrides if explicitly present
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER").lower()
    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")
    if os.getenv("EMBEDDING_MODEL"):
        config.embeddings.model_name = os.getenv("EMBEDDING_MODEL")
    if os.getenv("API_HOST"):
        config.api.host = os.getenv("API_HOST")
    if os.getenv("API_PORT"):
        config.api.port = int(os.getenv("API_PORT"))
    if os.getenv("STREAMLIT_API_URL"):
        config.streamlit.api_url = os.getenv("STREAMLIT_API_URL")

    # Ensure required runtime directories exist
    config.paths.raw_data_path.mkdir(parents=True, exist_ok=True)
    config.paths.processed_data_path.mkdir(parents=True, exist_ok=True)
    config.paths.logs_path.mkdir(parents=True, exist_ok=True)

    return config


# Global cached config instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance
