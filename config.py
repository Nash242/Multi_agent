import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Paths
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# Model configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Retrieval configuration
DEFAULT_SEARCH_TYPE = "similarity"
DEFAULT_K = 4
