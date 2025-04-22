"""
Telemetry component for the search pipeline.
"""
import logging
import time
from typing import Dict, Any

from models.state import SearchState

logger = logging.getLogger(__name__)

def add_telemetry(state: SearchState) -> SearchState:
    """
    Adds telemetry data to the search state.
    
    Args:
        state: The current search state
        
    Returns:
        Updated state with telemetry data
    """
    # Get existing metadata
    metadata = state.get("metadata", {})
    
    # Add timing information
    metadata["process_complete_timestamp"] = time.time()
    
    # Calculate execution time if we have start timestamp
    if "query_timestamp" in metadata:
        start_time = metadata["query_timestamp"]
        current_time = metadata["process_complete_timestamp"]
        metadata["total_execution_time"] = current_time - start_time
    
    # Add quality metrics
    has_error = state.get("error") is not None
    metadata["search_quality_score"] = 0.95 if not has_error else 0.5
    
    # Add statistics about components
    metadata["pipeline_components_executed"] = _count_components_executed(state)
    
    # Log telemetry data
    logger.debug(f"Added telemetry data: quality_score={metadata['search_quality_score']}, "
                f"has_error={has_error}")
    
    return {**state, "metadata": metadata}

def _count_components_executed(state: SearchState) -> int:
    """
    Count how many pipeline components were executed.
    
    Args:
        state: The current search state
        
    Returns:
        Number of components executed
    """
    # Simplified component counting logic - in a real system,
    # you'd have more sophisticated tracking of component execution
    count = 0
    
    # Core components
    if state.get("intent"):
        count += 1  # Intent classification
        
    if state.get("parameters"):
        count += 1  # Parameter extraction
        
    if state.get("enhanced_query"):
        count += 1  # Query enhancement
        
    if state.get("retrieval_results"):
        count += 1  # Vector search
        
    if state.get("ranked_results"):
        count += 1  # Results ranking
        
    if state.get("response"):
        count += 1  # Response generation
    
    return count