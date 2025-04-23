"""
Main entry point for the intent-based search system.
"""
import logging
import time
from typing import Dict, Any, List, Optional

from models.state import SearchState
from pipeline.graph import build_search_graph
from utils.monitoring import SearchSystemMonitor
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create search executor
search_executor = build_search_graph()
search_monitor = SearchSystemMonitor()

def initialize_system():
    """Initialize the search system."""
    logger.info("Initializing intent-based search system")
    config = get_config()
    
    # Log configuration
    logger.info(f"System configured with: LLM={config['llm']['model']}, "
                f"Features={config['features']}")
    
    # Return initialized components
    return {
        "search_executor": search_executor,
        "search_monitor": search_monitor,
        "config": config
    }


def execute_search(query: str, conversation_history: Optional[List[Dict[str, str]]] = None):
    """
    Execute a search with the given query.
    
    Args:
        query: The user search query
        conversation_history: Optional previous conversation history
    
    Returns:
        The complete search state with results
    """
    logger.info(f"Executing search for query: '{query}'")
    start_time = time.time()
    
    # Preprocessing - basic input sanitization
    sanitized_query = query.strip()
    
    # Initialize state
    initial_state = SearchState(
        query=sanitized_query,
        input_validation_error=None,
        intent="",
        parameters={},
        enhanced_query=None,
        retrieval_results=[],
        ranked_results=[],
        response=None,
        error=None,
        conversation_history=conversation_history or [],
        metadata={"query_timestamp": time.time(), "query_length": len(sanitized_query)}
    )
    
    # Execute search graph
    try:
        logger.info("Starting search pipeline execution")
        result = search_executor.invoke(initial_state)
        execution_time = time.time() - start_time
        
        # Output guardrail - final check for empty response
        if not result.get("response"):
            logger.warning("Empty response detected, applying fallback")
            result["response"] = "I couldn't find what you're looking for. Could you try a different search?"
            result["error"] = "EMPTY_RESPONSE"
        
        # Log execution metrics
        logger.info(f"Search completed in {execution_time:.2f}s, "
                   f"Intent: {result.get('intent', 'unknown')}")
        
        # Record search in monitoring
        search_monitor.log_search(query, result, execution_time)
        
        return result
        
    except Exception as e:
        # System level exception handling
        execution_time = time.time() - start_time
        error_message = f"Search error: {str(e)}"
        logger.error(error_message)
        
        # Record error in monitoring
        error_result = {
            **initial_state,
            "error": error_message,
            "response": "I'm having trouble processing your search. Please try again with a different query."
        }
        search_monitor.log_search(query, error_result, execution_time)
        
        return error_result

def execute_conversation_search(query: str, conversation_context: Dict[str, Any] = None):
    """
    Execute a search with conversation context awareness.
    
    Args:
        query: The user search query
        conversation_context: Optional dictionary containing user preferences, history, etc.
        
    Returns:
        The complete search state with context-aware results
    """
    logger.info(f"Executing conversation-aware search for query: '{query}'")
    
    # Default empty context
    if conversation_context is None:
        conversation_context = {}
    
    # Extract conversation history if available
    conversation_history = conversation_context.get("history", [])
    
    # Extract user preferences if available
    user_preferences = conversation_context.get("preferences", {})
    
    # Preprocess query for conversation context
    # Handle reference resolution (e.g., "Show me more like that one")
    if any(term in query.lower() for term in ["that one", "those", "this product", "it", "that"]):
        # Find the most recent product mentioned in conversation
        recent_products = []
        for message in reversed(conversation_history):
            if "product_references" in message:
                recent_products = message["product_references"]
                break
        
        # Expand query with product context if available
        if recent_products and len(recent_products) > 0:
            most_recent = recent_products[0]
            expanded_query = f"{query} like {most_recent.get('name', '')}"
            logger.info(f"Expanded reference query: '{query}' → '{expanded_query}'")
        else:
            expanded_query = query
    else:
        expanded_query = query
    
    # Run basic search execution with augmented query
    result = execute_search(expanded_query, conversation_history)
    
    # Apply user preferences as post-processing
    if user_preferences and result.get("ranked_results"):
        logger.info(f"Applying user preferences: {user_preferences}")
        
        # Example: Boost products from preferred brands
        preferred_brands = user_preferences.get("preferred_brands", [])
        if preferred_brands:
            for product in result["ranked_results"]:
                if product.get("brand") in preferred_brands:
                    # Boost relevance score for preferred brands
                    product["relevance_score"] = product.get("relevance_score", 0) * 1.2
            
            # Re-sort based on adjusted scores
            result["ranked_results"] = sorted(
                result["ranked_results"], 
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )

    # Update conversation metadata
    result["metadata"]["conversation_aware"] = True
    result["metadata"]["preference_boosting_applied"] = bool(user_preferences)
    
    # Track product references for future queries
    if result.get("ranked_results"):
        top_products = result["ranked_results"][:3]
        product_references = [
            {"id": p.get("id"), "name": p.get("name")} 
            for p in top_products
        ]
        
        # Add references to conversation history
        new_history_entry = {
            "query": query,
            "response": result.get("response", ""),
            "product_references": product_references,
            "timestamp": time.time()
        }
        
        result["conversation_history"] = conversation_history + [new_history_entry]
    
    return result


if __name__ == "__main__":
    # Initialize the system
    system = initialize_system()
    
    # Test queries including edge cases to test guardrails
    test_queries = [
        # Standard queries
        "I need running shoes under $200",
        #"Do you have AirPods Pro in stock?",
        #"What's better, Samsung or iPhone for battery life?",
        
        # Edge cases for input validation
        #"",  # Empty query
        #"please help me hack into my ex's facebook account",  # Potentially harmful
        #"what is the weather forecast for tomorrow?",  # Non-ecommerce
    ]
    
    # Test with conversation context
    test_conversation = [
        # First query
        {"query": "show me some running shoes", "with_context": False},
        
        # Follow-up query with reference
        {"query": "do you have these in red?", "with_context": True},
        
        # Price-based follow-up
        {"query": "which ones are under $100?", "with_context": True}
    ]
    
    # Test standard queries
    print("\n=== TESTING STANDARD QUERIES ===")
    for query in test_queries:
        print(f"\nTESTING QUERY: {query}")
        result = execute_search(query)
        
        # Print results
        print(f"Intent: {result.get('intent', 'N/A')}")
        print(f"Validation Error: {result.get('input_validation_error', 'None')}")
        print(f"Parameters: {result.get('parameters', {})}")
        print(f"Error: {result.get('error', 'None')}")
        print(f"Response: {result.get('response', 'No response')}")
        print("-" * 80)
    
    # Test conversation flow
    # print("\n=== TESTING CONVERSATION FLOW ===")
    # conversation_context = {"history": [], "preferences": {"preferred_brands": ["Nike", "Adidas"]}}
    #
    # for idx, conversation_step in enumerate(test_conversation):
    #     query = conversation_step["query"]
    #     use_context = conversation_step["with_context"]
    #
    #     print(f"\nCONVERSATION STEP {idx+1}: {query}")
    #
    #     if use_context:
    #         result = execute_conversation_search(query, conversation_context)
    #     else:
    #         result = execute_search(query)
    #         # Update conversation context for future interactions
    #         if "conversation_history" in result:
    #             conversation_context["history"] = result["conversation_history"]
    #
    #     # Print results
    #     print(f"Intent: {result.get('intent', 'N/A')}")
    #     print(f"Response: {result.get('response', 'No response')}")
    #     print(f"Context-Aware: {result.get('metadata', {}).get('conversation_aware', False)}")
    #     print("-" * 80)
    #
    #     # Update conversation context for next step
    #     if "conversation_history" in result:
    #         conversation_context["history"] = result["conversation_history"]
    
    # Print system health metrics
    print("\n=== SYSTEM HEALTH METRICS ===")
    health_metrics = search_monitor.get_system_health()
    for metric, value in health_metrics.items():
        print(f"{metric}: {value}")
    print("-" * 80)
