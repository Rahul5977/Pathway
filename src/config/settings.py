"""Configuration management for the Market Anomaly Detector."""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY: Optional[str] = Field(None, env="FINNHUB_API_KEY")
    POLYGON_API_KEY: Optional[str] = Field(None, env="POLYGON_API_KEY")
    NEWS_API_KEY: Optional[str] = Field(None, env="NEWS_API_KEY")
    TWITTER_BEARER_TOKEN: Optional[str] = Field(None, env="TWITTER_BEARER_TOKEN")
    
    # Application Configuration
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    DEBUG: bool = Field(False, env="DEBUG")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    
    # Database
    DATABASE_URL: str = Field("sqlite:///./anomaly_detector.db", env="DATABASE_URL")
    
    # Pathway Configuration
    PATHWAY_PERSISTENCE_DIR: str = Field("./data/persistence", env="PATHWAY_PERSISTENCE_DIR")
    PATHWAY_MONITORING_LEVEL: str = Field("WARN", env="PATHWAY_MONITORING_LEVEL")
    
    # Server Configuration
    FASTAPI_HOST: str = Field("localhost", env="FASTAPI_HOST")
    FASTAPI_PORT: int = Field(8000, env="FASTAPI_PORT")
    STREAMLIT_SERVER_PORT: int = Field(8501, env="STREAMLIT_SERVER_PORT")
    ENABLE_API: bool = Field(True, env="ENABLE_API")
    
    # AI Configuration
    LLM_MODEL: str = Field("gpt-3.5-turbo", env="LLM_MODEL")
    EMBEDDING_MODEL: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")
    MAX_TOKENS: int = Field(1000, env="MAX_TOKENS")
    TEMPERATURE: float = Field(0.1, env="TEMPERATURE")
    
    # Anomaly Detection Parameters
    ANOMALY_THRESHOLD: float = Field(2.0, env="ANOMALY_THRESHOLD")
    LOOKBACK_WINDOW: int = Field(100, env="LOOKBACK_WINDOW")  # Number of data points to analyze
    
    # Market Data Configuration
    SYMBOLS_TO_MONITOR: List[str] = Field(
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "BTC-USD", "ETH-USD"],
        env="SYMBOLS_TO_MONITOR"
    )
    DATA_UPDATE_INTERVAL: int = Field(5, env="DATA_UPDATE_INTERVAL")  # seconds
    
    # News and Sentiment
    NEWS_SOURCES: List[str] = Field(
        ["reuters", "bloomberg", "financial-times", "wall-street-journal"],
        env="NEWS_SOURCES"
    )
    SENTIMENT_UPDATE_INTERVAL: int = Field(60, env="SENTIMENT_UPDATE_INTERVAL")  # seconds
    
    # RAG Configuration
    VECTOR_STORE_TYPE: str = Field("chroma", env="VECTOR_STORE_TYPE")
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env="CHUNK_OVERLAP")
    MAX_SEARCH_RESULTS: int = Field(5, env="MAX_SEARCH_RESULTS")
    
    # Alert Configuration
    ALERT_WEBHOOK_URL: Optional[str] = Field(None, env="ALERT_WEBHOOK_URL")
    ALERT_EMAIL: Optional[str] = Field(None, env="ALERT_EMAIL")
    ENABLE_ALERTS: bool = Field(True, env="ENABLE_ALERTS")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        Path(self.PATHWAY_PERSISTENCE_DIR).mkdir(parents=True, exist_ok=True)
        Path("./data/output").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
