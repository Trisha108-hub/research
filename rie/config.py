from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    watsonx_api_key: str = ""
    watsonx_project_id: str = ""
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"
    watsonx_model_id: str = "ibm/granite-4-h-small"
    openalex_api_key: str = ""
    crossref_mailto: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_scholar_api_key: str = ""
    faiss_index_path: Path = Path("data/faiss.index")
    faiss_text_path: Path = Path("data/chunks.jsonl")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
