"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from opik import Opik



class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    llm_model: str
    ollama_host: str

    embedding_model: str
    hf_token: str | None = None

    temperature: float
    top_k: int
    top_p: float
    max_tokens: int

    chunk_size: int
    chunk_overlap: int

    opik_api_key: str | None = None
    opik_workspace: str = "akanksha-patil-0279"
    opik_project: str = "Enterprise AI Knowledge Assistant"
    # opik_url_override: str = "https://www.comet.com/opik/api"


settings = Settings()


opik = Opik(
    api_key=settings.opik_api_key,
    workspace=settings.opik_workspace,
    project_name=settings.opik_project,
    batching=False,
)

# Project paths
DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "faiss"

VECTOR_STORE_DIR = INDEX_DIR / "index"
INFO_FILE = INDEX_DIR / "index_info.json"