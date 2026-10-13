import os
from dotenv import load_dotenv

load_dotenv()

# Folders
PAPERS_DIR = os.getenv("PAPERS_DIR", "papers")
DB_DIR = os.getenv("DB_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION", "papers")

# Embedding model
# bge models need this instruction before the query (not before the chunks), set it to "" for other models
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_INSTRUCTION = os.getenv("QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: ")

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM - works with any OpenAI compatible API, default is local Ollama
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
