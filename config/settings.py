"""
Configuration settings for RAG Code Assistant
"""
import torch


LLM_MODEL = "deepseek-coder:6.7b"
LLM_TEMPERATURE = 0

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
CHUNK_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]


RETRIEVAL_SEARCH_TYPE = "mmr"  # Maximal Marginal Relevance
RETRIEVAL_K = 8  
RETRIEVAL_FETCH_K = 20 


PERSIST_DIRECTORY = "./gradio_db"


CHAT_HISTORY_TTL = 3600  
CHAT_HISTORY_LIMIT = 5 
CHAT_HISTORY_UI_LIMIT = 20  


SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860
SERVER_SHARE = False


DEFAULT_CRAWL_DEPTH = 2
MAX_CRAWL_DEPTH = 5
CRAWLER_TIMEOUT = 10


SITEMAP_PATHS = [
    "/sitemap.xml",
    "/sitemap_index.xml"
]

HTML_REMOVE_TAGS = ['script', 'style', 'nav', 'footer', 'header']