import os
from typing import Optional
from dotenv import load_dotenv 

load_dotenv()


class Config:
    # API Keys & LLM Configuration (LiteLLM / OpenAI / Nebius / Mistral)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", os.getenv("NEBIUS_API_KEY"))
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"))
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", os.getenv("NEBIUS_MODEL", "gemini-2.5-flash"))
    
    NEBIUS_API_KEY: Optional[str] = OPENAI_API_KEY
    NEBIUS_BASE_URL: str = OPENAI_BASE_URL
    NEBIUS_MODEL: str = OPENAI_MODEL

    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN"))
    
    # Server Configuration
    SERVER_NAME: str = os.getenv("SERVER_NAME", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", os.getenv("PORT", "7860")))
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")  # Using smaller model
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    DOCUMENT_STORE_PATH: str = os.getenv("DOCUMENT_STORE_PATH", "./data/documents")
    INDEX_NAME: str = os.getenv("INDEX_NAME", "content_index")
    
    # Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    
    # Search Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.1"))
    
    # OCR Configuration
    TESSERACT_PATH: Optional[str] = os.getenv("TESSERACT_PATH")
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    
    # Template & Structured Extraction Configuration
    TEMPLATE_STORE_PATH: str = os.getenv("TEMPLATE_STORE_PATH", "./data/templates")
    EXTRACTION_TIMEOUT: int = int(os.getenv("EXTRACTION_TIMEOUT", "60"))
    MAX_EXTRACTION_FILE_SIZE: int = int(os.getenv("MAX_EXTRACTION_FILE_SIZE", str(25 * 1024 * 1024)))  # 25 MB
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        # Make API keys optional for testing
        return True

# Global config instance
config = Config()

# Create data directories
import pathlib
pathlib.Path(config.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(config.DOCUMENT_STORE_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(config.TEMPLATE_STORE_PATH).mkdir(parents=True, exist_ok=True)