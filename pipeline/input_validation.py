"""
Input validation components for the search pipeline.
"""
import re
import logging
from typing import Dict, Any

from models.state import SearchState

logger = logging.getLogger(__name__)

# Regular expression patterns for validation
POTENTIALLY_HARMFUL_PATTERN = r'(hack|exploit|steal|crack|illegal|script\s*kiddie)'
NON_ECOMMERCE_PATTERNS = [
    r'(how\s+to\s+make|recipe for|directions to|weather in|news about)',
    r'(what time is|when does|who is the president|sports score)'
]

def validate_input(state: SearchState) -> SearchState:
    """
    Validates the user input query.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with validation results
    """
    query = state["query"]
    validation_error = None
    
    # Log the incoming query
    logger.debug(f"Validating query: {query}")
    
    # Check for empty query
    if not query or query.strip() == "":
        validation_error = "EMPTY_QUERY"
        logger.info("Query validation failed: Empty query")
    
    # Check for query length
    elif len(query) > 500:
        validation_error = "QUERY_TOO_LONG"
        logger.info(f"Query validation failed: Query too long ({len(query)} chars)")
    
    # Check for potentially harmful content
    elif re.search(POTENTIALLY_HARMFUL_PATTERN, query, re.IGNORECASE):
        validation_error = "POTENTIALLY_HARMFUL_CONTENT"
        logger.warning(f"Query validation failed: Potentially harmful content")
        
    # Check for non-ecommerce queries
    else:
        for pattern in NON_ECOMMERCE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                validation_error = "NON_ECOMMERCE_QUERY"
                logger.info(f"Query validation failed: Non-ecommerce query")
                break
    
    # Add validation metadata
    metadata = {
        **(state.get("metadata", {})),
        "input_validation_timestamp": "timestamp_here",
        "query_length": len(query)
    }
    
    # Update state
    return {
        **state, 
        "input_validation_error": validation_error,
        "metadata": metadata
    }

def handle_validation_error(state: SearchState) -> SearchState:
    """
    Handles input validation errors by creating appropriate responses.
    
    Args:
        state: The current search state with validation error
        
    Returns:
        Updated state with error response
    """
    error_type = state.get("input_validation_error")
    
    error_messages = {
        "EMPTY_QUERY": "I noticed your search was empty. What kind of products are you looking for?",
        "QUERY_TOO_LONG": "Your search query is quite detailed. Could you try a more concise search?",
        "POTENTIALLY_HARMFUL_CONTENT": "I'm unable to process that type of search query. Let me help you find products instead.",
        "NON_ECOMMERCE_QUERY": "I'm specialized in finding products. What items are you interested in shopping for?"
    }
    
    response = error_messages.get(error_type, "I couldn't process your search. Could you try rephrasing it?")
    logger.info(f"Generating validation error response for: {error_type}")
    
    return {
        **state, 
        "response": response,
        "error": f"Input validation failed: {error_type}"
    }