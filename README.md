# Intent-Based Search System

A comprehensive search system for ecommerce that goes beyond keyword matching to understand the true intent behind user queries.

## Overview

This intent-based search system uses LangGraph to process natural language queries through a pipeline that:

1. Classifies user search intent
2. Extracts structured parameters from queries
3. Enhances queries with relevant context
4. Performs semantic search using vector embeddings
5. Re-ranks results based on user intent
6. Validates result quality
7. Generates natural language responses

The system includes robust conversation context handling, personalization, and comprehensive guardrails for both input and output.

## Key Features

### Intent Recognition
The system recognizes seven core intents:
- **Product Discovery**: General browsing
- **Specific Product Search**: Looking for a specific item
- **Attribute Search**: Finding products with specific characteristics
- **Problem-Solution**: Finding products that solve a problem
- **Comparison**: Comparing multiple products
- **Price-Based**: Focused on price constraints
- **Availability**: Checking if items are in stock

### Advanced Capabilities
- **Conversation Context**: Maintains conversation history and resolves references
- **Personalization**: Applies user preferences to search results
- **Robust Guardrails**: Input validation, quality checks, and error recovery
- **Monitoring**: Collects performance metrics and user feedback

## Installation

### Prerequisites
- Python 3.9+
- Required packages (install with pip):
  ```
  pip install -r requirements.txt
  ```

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/intent-search.git
   cd intent-search
   ```

2. Configure environment variables:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```
   python main.py
   ```

## Project Structure

```
intent_search/
│
├── main.py                # Entry point for the application
├── config.py              # Configuration settings, environment variables
│
├── models/                # Data models and schemas
│   ├── state.py           # SearchState definition
│   └── parameters.py      # Parameter extraction models
│
├── pipeline/              # LangGraph pipeline components
│   ├── graph.py           # Graph structure and compilation
│   ├── input_validation.py
│   ├── intent_classification.py
│   ├── parameter_extraction.py
│   ├── query_enhancement.py
│   ├── vector_search.py
│   ├── results_ranking.py
│   ├── quality_validation.py
│   ├── response_generation.py
│   └── telemetry.py
│
├── services/              # Application services
│   ├── search_service.py
│   ├── conversation_service.py
│   ├── personalization_service.py
│   └── telemetry_service.py
│
├── vectordb/             # Vector database functionality
│   ├── vector_store.py   # Vector storage and retrieval
│   ├── embeddings.py     # Embedding creation
│   └── index.py          # Index management
│
├── data/                 # Data management
│   ├── products.py       # Product data handling
│   ├── users.py          # User data handling
│   └── session.py        # Session management
│
├── utils/                # Utility functions
│   ├── prompts.py        # Prompt templates
│   ├── llm.py            # LLM initialization and wrappers
│   └── monitoring.py     # Monitoring and logging utilities
│
├── api/                  # API implementation
│   └── main.py           # FastAPI implementation
│
└── tests/                # Test suite
    ├── test_pipeline.py
    ├── test_vector_search.py
    ├── test_intent_classification.py
    └── test_integration.py
```

## Usage

### Basic Search

```python
from main import execute_search

# Execute a basic search
result = execute_search("I need running shoes under $100")

# Access the response
print(result["response"])

# View structured results
for product in result["ranked_results"][:3]:
    print(f"{product['name']} - ${product['price']}")
```

### Conversation-Aware Search

```python
from main import execute_conversation_search

# Set up conversation context
conversation_context = {
    "history": [...],  # Previous interactions
    "preferences": {
        "preferred_brands": ["Nike", "Adidas"]
    }
}

# Execute with context
result = execute_conversation_search(
    "do you have these in red?", 
    conversation_context
)
```

### API Usage

Start the API server:
```
uvicorn api.main:app --reload
```

Make a search request:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "User-ID: user123" \
  -H "Session-ID: session456" \
  -d '{"query": "running shoes under $100", "enable_conversation": true}'
```

## Testing

Run tests with pytest:
```
pytest
```

Run specific test files:
```
pytest tests/test_pipeline.py
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.