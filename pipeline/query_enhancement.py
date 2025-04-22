"""
Query enhancement component for the search pipeline.
"""
import logging
from typing import Dict, Any

from models.state import SearchState
from utils.prompts import QUERY_ENHANCEMENT_PROMPT
from utils.llm import get_llm, safe_llm_call

logger = logging.getLogger(__name__)

def enhance_query(state: SearchState) -> SearchState:
    """
    Enhances the original query with additional context and synonyms.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with enhanced query
    """
    query = state["query"]
    intent = state["intent"]
    parameters = state["parameters"]
    
    logger.info(f"Enhancing query: '{query}' with intent: {intent}")
    
    # Create the chain
    chain = QUERY_ENHANCEMENT_PROMPT | get_llm()
    
    try:
        # Get enhanced query
        enhanced_query = safe_llm_call(
            chain=chain,
            inputs={
                "query": query, 
                "intent": intent, 
                "parameters": str(parameters)
            },
            default_response=query  # Fall back to original query if enhancement fails
        ).strip()
        
        # Log the enhancement
        logger.info(f"Enhanced query: '{enhanced_query}'")
        
        # Add query expansion to metadata
        expansion_rate = len(enhanced_query) / max(1, len(query))
        
        metadata = {
            **(state.get("metadata", {})),
            "query_enhancement_timestamp": "timestamp_here",
            "query_expansion_ratio": expansion_rate,
            "original_query_length": len(query),
            "enhanced_query_length": len(enhanced_query)
        }
        
        # Update state
        return {
            **state, 
            "enhanced_query": enhanced_query,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Query enhancement failed: {str(e)}")
        
        # Fall back to original query
        return {
            **state,
            "enhanced_query": query,  # Use original query as fallback
            "metadata": {
                **(state.get("metadata", {})),
                "query_enhancement_error": str(e),
                "using_original_query": True
            }
        }

def apply_domain_knowledge(query: str, intent: str) -> str:
    """
    Apply domain-specific knowledge to enhance the query.
    This function can be expanded with domain rules.
    
    Args:
        query: The original query
        intent: The classified intent
        
    Returns:
        Enhanced query with domain knowledge applied
    """
    # This is where you can add domain-specific enhancements
    # Example: For electronics, add technical specifications
    enhanced = query
    
    if intent == "PROBLEM_SOLUTION" and "battery" in query.lower():
        if "long" in query.lower() or "last" in query.lower():
            enhanced += " high capacity extended battery life"
    
    if intent == "SPECIFIC_PRODUCT":
        # Add common model variations
        if "iphone" in query.lower() and not any(x in query.lower() for x in ["pro", "max", "plus"]):
            enhanced += " pro max plus"
            
    # Add more domain-specific rules as needed
    
    return enhanced