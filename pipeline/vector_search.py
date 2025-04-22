"""
Vector search component for the search pipeline.
"""
import logging
from typing import Dict, Any, List

from models.state import SearchState
from vectordb.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Initialize vector store
vector_store = VectorStore()

def retrieve_results(state: SearchState) -> SearchState:
    """
    Retrieves results from vector store based on enhanced query.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with search results
    """
    # Use enhanced query if available, otherwise use original query
    enhanced_query = state.get("enhanced_query") or state["query"]
    
    logger.info(f"Performing vector search with query: {enhanced_query}")
    
    try:
        # Search vector store
        results = vector_store.search(enhanced_query, k=10)
        
        # Format results for the pipeline
        retrieval_results = []
        for product, score in results:
            retrieval_results.append({
                **product,
                "relevance_score": score
            })
        
        logger.info(f"Vector search found {len(retrieval_results)} results")
        
        # Add search metadata
        metadata = {
            **(state.get("metadata", {})),
            "vector_search_timestamp": "timestamp_here",
            "vector_search_result_count": len(retrieval_results)
        }
        
        # If no results were found, add that to metadata
        if not retrieval_results:
            metadata["no_results_found"] = True
            logger.warning("No results found for vector search")
        
        # Update state
        return {
            **state, 
            "retrieval_results": retrieval_results,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        
        # Update state with error
        return {
            **state,
            "retrieval_results": [],
            "error": f"Vector search failed: {str(e)}",
            "metadata": {
                **(state.get("metadata", {})),
                "vector_search_error": str(e)
            }
        }