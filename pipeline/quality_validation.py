"""
Quality validation component for the search pipeline.
"""
import logging
from typing import Dict, Any, Union

from models.state import SearchState
from utils.prompts import NO_RESULTS_PROMPT, QUALITY_ISSUES_PROMPT
from utils.llm import get_llm, safe_llm_call

logger = logging.getLogger(__name__)

def validate_results(state: SearchState) -> Union[Dict[str, Any], str]:
    """
    Validates search results for quality and relevance.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state or routing instruction
    """
    ranked_results = state["ranked_results"]
    intent = state["intent"]
    parameters = state["parameters"]
    
    logger.info(f"Validating results quality for intent: {intent}")
    
    # No results handling
    if not ranked_results:
        logger.warning("No results found")
        return "handle_no_results"
    
    # Check result quality
    quality_issues = []
    
    # For specific product searches, check if we found an exact match
    if intent == "SPECIFIC_PRODUCT" and parameters.get("specific_product"):
        specific_product = parameters.get("specific_product", "").lower()
        exact_match_found = any(
            specific_product in product.get("name", "").lower() 
            for product in ranked_results[:3]
        )
        
        if not exact_match_found:
            quality_issues.append("NO_EXACT_MATCH")
            logger.info(f"Quality issue: No exact match found for '{specific_product}'")
    
    # For price-based searches, check if results match price constraints
    if intent == "PRICE_BASED" and parameters.get("price_range"):
        price_range = parameters.get("price_range", {})
        price_min = price_range.get("min")
        price_max = price_range.get("max")
        
        price_match_found = False
        for product in ranked_results[:5]:
            price = product.get("price", 0)
            if ((price_min is None or price >= price_min) and 
                (price_max is None or price <= price_max)):
                price_match_found = True
                break
                
        if not price_match_found and (price_min is not None or price_max is not None):
            quality_issues.append("NO_PRICE_MATCH")
            price_constraints = []
            if price_min is not None:
                price_constraints.append(f"min={price_min}")
            if price_max is not None:
                price_constraints.append(f"max={price_max}")
            logger.info(f"Quality issue: No products match price constraints {', '.join(price_constraints)}")
    
    # For availability searches, check if products are in stock
    if intent == "AVAILABILITY":
        availability_found = any(
            product.get("in_stock", False) is True
            for product in ranked_results[:5]
        )
        
        if not availability_found:
            quality_issues.append("NO_IN_STOCK")
            logger.info("Quality issue: No in-stock products found")
    
    # For attribute searches, check if attributes match
    if intent == "ATTRIBUTE_SEARCH" and parameters.get("attributes"):
        attributes = parameters.get("attributes", {})
        if attributes:
            attribute_match_found = False
            
            for product in ranked_results[:5]:
                product_attrs = product.get("attributes", {})
                matches_all = True
                
                for attr_name, attr_values in attributes.items():
                    if not product_attrs.get(attr_name):
                        matches_all = False
                        break
                        
                    # Check if any of the requested values match
                    if isinstance(attr_values, list) and isinstance(product_attrs[attr_name], list):
                        if not any(val in product_attrs[attr_name] for val in attr_values):
                            matches_all = False
                            break
                
                if matches_all:
                    attribute_match_found = True
                    break
            
            if not attribute_match_found:
                quality_issues.append("NO_ATTRIBUTE_MATCH")
                logger.info(f"Quality issue: No products match requested attributes: {attributes}")
    
    # Update metadata with quality check results
    state["metadata"] = {
        **(state.get("metadata", {})),
        "result_quality_issues": quality_issues,
        "result_count": len(ranked_results),
        "quality_check_timestamp": "timestamp_here"
    }
    
    # Route to appropriate handler if there are quality issues
    if quality_issues:
        logger.info(f"Quality issues detected: {quality_issues}")
        return "handle_quality_issues"
    
    logger.info("Results quality validation passed")
    return "build_response"

def handle_no_results(state: SearchState) -> SearchState:
    """
    Handles the case where no search results were found.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with no results response
    """
    query = state["query"]
    intent = state["intent"]
    parameters = state["parameters"]
    
    logger.info(f"Handling no results for query: '{query}'")
    
    # Create the chain
    chain = NO_RESULTS_PROMPT | get_llm()
    
    # Get no results response
    response = safe_llm_call(
        chain=chain,
        inputs={
            "query": query,
            "intent": intent,
            "parameters": str(parameters)
        },
        default_response="I couldn't find any products matching your search. Could you try with different terms?"
    )
    
    return {
        **state, 
        "response": response,
        "error": "NO_RESULTS_FOUND",
        "metadata": {
            **(state.get("metadata", {})),
            "no_results_handler_executed": True
        }
    }

def handle_quality_issues(state: SearchState) -> SearchState:
    """
    Handles quality issues in search results.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with quality-aware response
    """
    query = state["query"]
    intent = state["intent"]
    ranked_results = state["ranked_results"]
    quality_issues = state["metadata"].get("result_quality_issues", [])
    parameters = state["parameters"]
    
    logger.info(f"Handling quality issues: {quality_issues}")
    
    # Create the chain
    chain = QUALITY_ISSUES_PROMPT | get_llm()
    
    # Get quality issues response
    response = safe_llm_call(
        chain=chain,
        inputs={
            "query": query,
            "intent": intent,
            "quality_issues": quality_issues,
            "parameters": str(parameters),
            "top_results": str(ranked_results[:3])
        },
        default_response="I found some products, but they may not be exactly what you're looking for. Here are the closest matches."
    )
    
    return {
        **state, 
        "response": response,
        "error": f"QUALITY_ISSUES: {','.join(quality_issues)}",
        "metadata": {
            **(state.get("metadata", {})),
            "quality_issues_handler_executed": True
        }
    }