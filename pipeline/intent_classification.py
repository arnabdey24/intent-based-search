"""
Intent classification component for the search pipeline.
"""
import logging
from typing import Dict, Any, Union

from models.state import SearchState
from models.parameters import VALID_INTENTS
from utils.prompts import INTENT_CLASSIFICATION_PROMPT
from utils.llm import get_llm, safe_llm_call

logger = logging.getLogger(__name__)

def classify_intent(state: SearchState) -> Union[Dict[str, Any], str]:
    """
    Identifies the user's search intent.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with classified intent or routing instruction
    """
    # Check if input validation passed
    if state.get("input_validation_error"):
        logger.info("Skipping intent classification due to validation error")
        return "handle_validation_error"
    
    query = state["query"]
    logger.info(f"Classifying intent for query: {query}")
    
    # Get intent classification
    chain = INTENT_CLASSIFICATION_PROMPT | get_llm()
    intent = safe_llm_call(
        chain=chain,
        inputs={"query": query},
        default_response="PRODUCT_DISCOVERY"  # Default fallback
    ).strip()
    
    logger.info(f"Classified intent: {intent}")
    
    # Validate intent is in our allowed list
    if intent not in VALID_INTENTS:
        logger.warning(f"Invalid intent received: {intent}, using default")
        intent = "PRODUCT_DISCOVERY"  # Default fallback
    
    # Calculate confidence based on query length and complexity
    # This is a simple heuristic that could be replaced with a more sophisticated approach
    confidence = "high" if len(query) > 5 else "low"
    
    # Update state with intent and metadata
    return {
        **state, 
        "intent": intent,
        "metadata": {
            **(state.get("metadata", {})),
            "intent_classification_confidence": confidence,
            "intent_classification_timestamp": "timestamp_here"
        }
    }