"""
Configuration settings for the intent-based search system.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "type": os.environ.get("DB_TYPE", "sqlite"),
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "user": os.environ.get("DB_USER", ""),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "intent_search"),
    "connection_string": ""
}

# Set database connection string based on type
if DB_CONFIG["type"] == "postgres":
    DB_CONFIG["connection_string"] = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
elif DB_CONFIG["type"] == "mysql":
    DB_CONFIG["connection_string"] = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
elif DB_CONFIG["type"] == "sqlite":
    DB_CONFIG["connection_string"] = f"sqlite:///data/intent_search.db"

# Redis configuration (for session management)
REDIS_CONFIG = {
    "host": os.environ.get("REDIS_HOST", "localhost"),
    "port": int(os.environ.get("REDIS_PORT", "6379")),
    "password": os.environ.get("REDIS_PASSWORD", ""),
    "db": int(os.environ.get("REDIS_DB", "0"))
}

# Vector database configuration
VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "faiss")

# Vector store config
VECTOR_STORE_CONFIG = {
    "type": VECTOR_DB_TYPE,
    "index_name": os.environ.get("VECTOR_STORE_INDEX", "products"),
    "dimension": int(os.environ.get("VECTOR_DIMENSION", "384")),
    "metric": os.environ.get("VECTOR_METRIC", "cosine")
}

# Add specific vector DB configurations
if VECTOR_DB_TYPE == "pinecone":
    VECTOR_STORE_CONFIG.update({
        "api_key": os.environ.get("PINECONE_API_KEY", ""),
        "environment": os.environ.get("PINECONE_ENVIRONMENT", "")
    })
elif VECTOR_DB_TYPE == "qdrant":
    VECTOR_STORE_CONFIG.update({
        "url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
        "api_key": os.environ.get("QDRANT_API_KEY", ""),
        "collection_name": os.environ.get("QDRANT_COLLECTION", "products"),
        "dimension": int(os.environ.get("VECTOR_DIMENSION", "384")),
    })
elif VECTOR_DB_TYPE == "milvus":
    VECTOR_STORE_CONFIG.update({
        "host": os.environ.get("MILVUS_HOST", "localhost"),
        "port": os.environ.get("MILVUS_PORT", "19530")
    })

# LLM configuration
LLM_CONFIG = {
    "model": os.environ.get("LLM_MODEL", "gemini-2.0-flash-lite"),
    "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.1")),
    "api_key": os.environ.get("LLM_API_KEY", "")
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "model": os.environ.get("EMBEDDING_MODEL", "models/embedding-001"),
    "api_key": os.environ.get("LLM_API_KEY", "")
}

# Application configuration
APP_CONFIG = {
    "debug": os.environ.get("DEBUG", "False").lower() == "true",
    "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    "cache_ttl": int(os.environ.get("CACHE_TTL", "3600")),  # in seconds
}

# Feature flags
FEATURES = {
    "use_conversation_context": os.environ.get("USE_CONVERSATION", "True").lower() == "true",
    "use_personalization": os.environ.get("USE_PERSONALIZATION", "True").lower() == "true",
    "log_telemetry": os.environ.get("LOG_TELEMETRY", "True").lower() == "true",
}

def get_config() -> Dict[str, Any]:
    """Return the complete configuration dictionary."""
    return {
        "llm": LLM_CONFIG,
        "embedding": EMBEDDING_CONFIG,
        "vector_store": VECTOR_STORE_CONFIG,
        "db": DB_CONFIG,
        "redis": REDIS_CONFIG,
        "app": APP_CONFIG,
        "features": FEATURES
    }
