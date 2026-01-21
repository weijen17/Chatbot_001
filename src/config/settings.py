"""Configuration settings for the two-agent system"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Model Configuration
    MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "openai")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    TOP_P: float = float(os.getenv("TOP_P", "0.1"))

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    RAW_DATA_DIR: Path = BASE_DIR / "input/raw_data"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Embedding and Faiss Configuration
    EMBED_MODEL: str = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    FAISS_DIR = BASE_DIR / "artifact" / "faiss"
    INDEX_DIR = FAISS_DIR / os.getenv("INDEX_NAME", "index.faiss")
    DOCS_DIR = FAISS_DIR / os.getenv("DOCS_NAME", "data.npy")

    # First round and final round retrieve document frequency
    TOP_K = int(os.getenv("TOP_K", "100"))
    TOP_K2 = int(os.getenv("TOP_K2", "10"))

    # Input
    INPUT_FILENAME: str = os.getenv("INPUT_FILENAME", "10k_red_data_v1.2.xlsx")
    INPUT_NAME: Path = RAW_DATA_DIR / INPUT_FILENAME

    # Agent Configuration
    RESEARCH_AGENT_VERBOSE: bool = True
    ANALYST_AGENT_VERBOSE: bool = True

    def __init__(self):
        """Initialize settings and create necessary directories"""
        # self.DF_DIR.mkdir(exist_ok=True)


        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

settings = Settings()

