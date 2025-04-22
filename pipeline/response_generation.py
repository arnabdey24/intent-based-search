"""
Response generation component for the search pipeline.
"""
import re
import logging
from typing import Dict, Any

from models.state import SearchState
from utils.prompts import RESPONSE_GENERATION_PROMPT, RESPONSE_CLEANING_PROMPT
from utils.llm import get_llm, safe_llm_call

logger = logging.getLogger(__name__)

def build_response(state: SearchState) -> SearchState:
    """
    Builds a natural language response from the ranked results.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with generated response
    """
    query = state["query"]
    intent = state["intent"]
    ranked_results = state["ranked_results"]
    parameters = state["parameters"]
    
    logger.info(f"Building response for query: '{query}' with intent: {intent}")
    
    # Create the chain
    chain = RESPONSE_GENERATION_PROMPT | get_llm()
    
    try:
        # Get response
        response = safe_llm_call(
            chain=chain,
            inputs={
                "query": query,
                "intent": intent,
                "parameters": str(parameters),
                "top_results": str(ranked_results[:3])
            },
            default_response=f"Here are some products that match your search for '{query}'."
        )
        
        logger.debug(f"Raw response generated: {response[:100]}...")
        
        # Output validation - check for prohibited content
        cleaned_response = clean_response(response)
        
        # Update state
        result = {
            **state, 
            "response": cleaned_response,
            "metadata": {
                **(state.get("metadata", {})),
                "response_generation_timestamp": "timestamp_here",
                "response_word_count": len(cleaned_response.split()),
                "response_required_cleaning": cleaned_response != response
            }
        }
        
        logger.info(f"Response generated successfully: {len(cleaned_response.split())} words")
        return result
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        
        # Fallback to simple response
        fallback_response = _generate_fallback_response(query, ranked_results)
        
        return {
            **state,
            "response": fallback_response,
            "error": f"Response generation error: {str(e)}",
            "metadata": {
                **(state.get("metadata", {})),
                "used_fallback_response": True
            }
        }

def clean_response(response: str) -> str:
    """
    Clean response by removing prohibited patterns.
    
    Args:
        response: The raw response
        
    Returns:
        Cleaned response
    """
    # Check for prohibited patterns
    prohibited_patterns = [
        r'(I apologize|I\'m sorry|As an AI|I don\'t have access)',
        r'(cannot|can\'t) (provide|give|offer) (you )?(specific|exact|real)'
    ]
    
    contains_prohibited = any(
        re.search(pattern, response, re.IGNORECASE) 
        for pattern in prohibited_patterns
    )
    
    # If prohibited patterns found, clean the response
    if contains_prohibited:
        logger.info("Prohibited patterns found in response, cleaning")
        
        # Create the chain
        chain = RESPONSE_CLEANING_PROMPT | get_llm()
        
        # Clean the response
        cleaned = safe_llm_call(
            chain=chain,
            inputs={"response": response},
            default_response=response  # Fall back to original if cleaning fails
        )
        
        return cleaned
    
    return response

def _generate_fallback_response(query: str, results: list) -> str:
    """
    Generate a simple fallback response when LLM generation fails.
    
    Args:
        query: The search query
        results: The search results
        
    Returns:
        Simple fallback response
    """
    if not results:
        return f"I found no products matching your search for '{query}'."
    
    response = f"Here are some products that match your search for '{query}':\n\n"
    
    for i, product in enumerate(results[:3], 1):
        name = product.get("name", "Product")
        price = product.get("price", "")
        price_text = f" - ${price}" if price else ""
        response += f"{i}. {name}{price_text}\n"
    
    return response