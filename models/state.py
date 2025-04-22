"""
State definitions for the intent-based search system.
"""
from typing import Dict, List, Any, Optional, TypedDict

class SearchState(TypedDict):
    """
    Represents the state of our search graph.
    Maintains all information as it flows through the pipeline.
    """
    # Core query information
    query: str  # Original user query
    intent: str  # Classified intent
    parameters: Dict[str, Any]  # Extracted search parameters
    enhanced_query: Optional[str]  # Query after enhancement
    
    # Results and response
    retrieval_results: List[Dict[str, Any]]  # Raw search results
    ranked_results: List[Dict[str, Any]]  # Final ranked results
    response: Optional[str]  # Response to return to user
    
    # Error handling
    input_validation_error: Optional[str]  # Error from input validation
    error: Optional[str]  # Any error that occurred
    
    # Context and metadata
    conversation_history: List[Dict[str, str]]  # Chat history for context
    metadata: Dict[str, Any]  # Metadata about the search process