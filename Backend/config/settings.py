from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ConfigDict
#from google.cloud import storage
#from typing import Optional
import os

class Settings(BaseSettings):
    # Google Cloud Platform
    GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID")
    GCP_LOCATION: str = "us-central1"
    GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME")

    DOCAI_PROJECT_ID: str | None = os.environ.get("DOCAI_PROJECT_ID")
    DOCAI_LOCATION: str = os.environ.get("DOCAI_LOCATION", "us")
    DOCAI_PROCESSOR_ID: str | None = os.environ.get("DOCAI_PROCESSOR_ID")

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
