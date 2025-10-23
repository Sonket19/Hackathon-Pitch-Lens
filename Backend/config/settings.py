from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ConfigDict
#from google.cloud import storage
#from typing import Optional
import os

class Settings(BaseSettings):
    # Google Cloud Platform
    GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID")
    GCP_LOCATION: str = os.environ.get("GCP_LOCATION", "us-central1")
    GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME")
    VERTEX_GROUNDED_DATASTORE: str | None = os.environ.get("VERTEX_GROUNDED_DATASTORE")
    VERTEX_ENABLE_GOOGLE_GROUNDING: bool = os.environ.get("VERTEX_ENABLE_GOOGLE_GROUNDING", "true").lower() in {"1", "true", "yes"}
    VERTEX_GROUNDED_MODEL: str = os.environ.get("VERTEX_GROUNDED_MODEL", "gemini-2.5-pro")
    VERTEX_MCS_ENDPOINT_ID: str | None = os.environ.get("VERTEX_MCS_ENDPOINT_ID")

    BIGQUERY_DATASET: str | None = os.environ.get("BIGQUERY_DATASET")
    BIGQUERY_MEMO_TABLE: str | None = os.environ.get("BIGQUERY_MEMO_TABLE")
    BIGQUERY_VECTOR_TABLE: str | None = os.environ.get("BIGQUERY_VECTOR_TABLE")
    BIGQUERY_LOCATION: str = os.environ.get("BIGQUERY_LOCATION", "US")

    VECTOR_EMBED_MODEL: str = os.environ.get("VECTOR_EMBED_MODEL", "text-embedding-004")

    # APIs
    GOOGLE_API_KEY: str
    GOOGLE_SEARCH_ENGINE_ID: str

    # Application
    APP_NAME: str = "AI Investment Memo Generator"
    DEBUG: bool = False
    
    #storage_client=storage.Client()

    #class Config:
    #    env_file = ".env"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
settings = Settings()
