OPENAI_LLM_MODEL = "gpt-5-nano" # your preferred LLM model
CHROMA_COLLECTION_NAME = "Custom_RAG_Agent" # name of your chroma collection
PERSIST_DIRECTORY = r"vectorebase/chroma_langchain_db" # directory to persist the chromaDB
KNOWLEDGE_BASE_PATH = "knowledge_base/" # path to your knowledge base
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # embedding model
LANGSMITH_PROJECT = "RAG_agent" # for langsmith tracking
SYSTEM_PROMPT = "SYSTEM_PROMPT_191025.j2" # system prompt file name from prompts/ folder