"""
Results ranking component for the search pipeline.
"""
import json
import logging
from typing import Dict, Any, List

from models.state import SearchState
from utils.prompts import RESULTS_RANKING_PROMPT
from utils.llm import get_llm, safe_llm_call

logger = logging.getLogger(__name__)

def rank_results(state: SearchState) -> SearchState:
    """
    Re-ranks results using LLM based on user intent.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with ranked results
    """
    intent = state["intent"]
    parameters = state["parameters"]
    retrieval_results = state["retrieval_results"]
    query = state["query"]
    
    logger.info(f"Ranking results for query: '{query}' with intent: {intent}")
    
    # If no results, skip ranking
    if not retrieval_results:
        logger.warning("No results to rank")
        return {
            **state, 
            "ranked_results": [], 
            "error": "No results found"
        }
    
    # Limit to top 5 for LLM processing to reduce costs
    top_results = retrieval_results[:7]
    
    # Create the chain
    chain = RESULTS_RANKING_PROMPT | get_llm()
    
    try:
        # Get rankings from LLM
        ranking_text = safe_llm_call(
            chain=chain,
            inputs={
                "query": query,
                "intent": intent,
                "parameters": str(parameters),
                "results": str(top_results)
            },
            default_response="[]"  # Fallback to empty rankings
        )
        
        logger.info(f"Raw ranking result: {ranking_text}")
        
        # Parse rankings
        try:

            # if start with ```json and end with ```, remove it
            if ranking_text.startswith("```json") and ranking_text.endswith("```"):
                ranking_text = ranking_text[7:-3].strip()

            rankings = json.loads(ranking_text)
            
            # Create ranked results by mapping rankings to full product data
            product_map = {p["id"]: p for p in retrieval_results}
            ranked_results = []
            
            for rank_item in sorted(rankings, key=lambda x: x.get("rank", 999)):
                product_id = rank_item.get("product_id")
                if product_id in product_map:
                    ranked_results.append({
                        **product_map[product_id],
                        "rank": rank_item.get("rank"),
                        "rank_reason": rank_item.get("reason", "")
                    })
            
            # Add any remaining results that weren't ranked
            for product in retrieval_results:
                if product["id"] not in [p["id"] for p in ranked_results]:
                    ranked_results.append(product)
                    
            logger.info(f"Successfully ranked {len(ranked_results)} results")
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from ranking: {ranking_text}")
            # Fallback to original results
            ranked_results = retrieval_results
        
    except Exception as e:
        logger.error(f"Error during result ranking: {str(e)}")
        # Fallback to original results
        ranked_results = retrieval_results
    
    # Update state
    return {
        **state, 
        "ranked_results": ranked_results,
        "metadata": {
            **(state.get("metadata", {})),
            "ranking_timestamp": "timestamp_here",
            "ranking_method": "llm_ranking" if ranked_results != retrieval_results else "vector_similarity"
        }
    }

def apply_business_rules(results: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply business rules to adjust rankings.
    
    Args:
        results: The ranked results
        parameters: Search parameters
        
    Returns:
        Results with business rules applied
    """
    # Example business rules:
    
    # 1. Boost products on promotion
    for product in results:
        if product.get("on_promotion", False):
            # Boost the relevance score by 10%
            product["relevance_score"] = product.get("relevance_score", 0) * 1.1
    
    # 2. Ensure price-filtered results come first when price_range specified
    if "price_range" in parameters and parameters["price_range"]:
        price_min = parameters["price_range"].get("min")
        price_max = parameters["price_range"].get("max")
        
        # Split results into matching and non-matching
        price_matching = []
        price_non_matching = []
        
        for product in results:
            price = product.get("price", 0)
            if ((price_min is None or price >= price_min) and 
                (price_max is None or price <= price_max)):
                price_matching.append(product)
            else:
                price_non_matching.append(product)
                
        # Sort each group by relevance
        price_matching.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        price_non_matching.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Combine with matching first
        results = price_matching + price_non_matching
    
    # 3. Additional business rules could be added here
    
    return results
