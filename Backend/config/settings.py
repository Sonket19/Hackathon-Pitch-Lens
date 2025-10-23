from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ConfigDict
#from google.cloud import storage
#from typing import Optional
import os

class Settings(BaseSettings):
    # Google Cloud Platform
    GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")
    GCP_LOCATION: str = "us-central1"
    GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME", "")

    # APIs
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GOOGLE_SEARCH_ENGINE_ID: str = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")
    VECTOR_SEARCH_INDEX: str = os.environ.get("VECTOR_SEARCH_INDEX", "")
    VECTOR_SEARCH_INDEX_ENDPOINT: str = os.environ.get("VECTOR_SEARCH_INDEX_ENDPOINT", "")
    VECTOR_SEARCH_DEPLOYED_INDEX_ID: str = os.environ.get("VECTOR_SEARCH_DEPLOYED_INDEX_ID", "")

    # Application
    APP_NAME: str = "AI Investment Memo Generator"
    DEBUG: bool = False
    
    #storage_client=storage.Client()

    #class Config:
    #    env_file = ".env"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
settings = Settings()
