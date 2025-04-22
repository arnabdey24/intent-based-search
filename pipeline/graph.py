"""
Graph structure for the LangGraph search pipeline.
"""
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from models.state import SearchState
from pipeline.input_validation import validate_input, handle_validation_error
from pipeline.intent_classification import classify_intent
from pipeline.parameter_extraction import extract_parameters
from pipeline.query_enhancement import enhance_query
from pipeline.vector_search import retrieve_results
from pipeline.results_ranking import rank_results
from pipeline.quality_validation import validate_results, handle_no_results, handle_quality_issues
from pipeline.response_generation import build_response
from pipeline.telementry import add_telemetry

logger = logging.getLogger(__name__)

def define_edges():
    """
    Define the edges (connections) between nodes in our graph.
    
    Returns:
        Dictionary defining the graph structure
    """
    return {
        # Input validation flow
        "validate_input": "classify_intent",
        
        # Main processing flow with conditional routing
        "classify_intent": {
            "handle_validation_error": "handle_validation_error",
            "DEFAULT": "extract_parameters"
        },
        "extract_parameters": "enhance_query",
        "enhance_query": "retrieve_results",
        "retrieve_results": "rank_results",
        "rank_results": "validate_results",
        
        # Result quality routing
        "validate_results": {
            "handle_no_results": "handle_no_results",
            "handle_quality_issues": "handle_quality_issues",
            "build_response": "build_response"
        },
        
        # Endpoints
        "handle_validation_error": "add_telemetry",
        "handle_no_results": "add_telemetry",
        "handle_quality_issues": "add_telemetry",
        "build_response": "add_telemetry",
        
        # Final telemetry
        "add_telemetry": END
    }

def build_search_graph():
    """Create the LangGraph for our search system."""
    # Initialize with empty state
    graph = StateGraph(SearchState)

    # Add all nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("handle_validation_error", handle_validation_error)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("extract_parameters", extract_parameters)
    graph.add_node("enhance_query", enhance_query)
    graph.add_node("retrieve_results", retrieve_results)
    graph.add_node("rank_results", rank_results)
    graph.add_node("validate_results", validate_results)
    graph.add_node("handle_no_results", handle_no_results)
    graph.add_node("handle_quality_issues", handle_quality_issues)
    graph.add_node("build_response", build_response)
    graph.add_node("add_telemetry", add_telemetry)

    # Define simple edges
    graph.add_edge("validate_input", "classify_intent")
    graph.add_edge("extract_parameters", "enhance_query")
    graph.add_edge("enhance_query", "retrieve_results")
    graph.add_edge("retrieve_results", "rank_results")
    graph.add_edge("rank_results", "validate_results")

    # Add conditional edge for input validation
    def has_validation_error(state):
        return state.get("input_validation_error") is not None

    graph.add_conditional_edges(
        "classify_intent",
        has_validation_error,
        {True: "handle_validation_error", False: "extract_parameters"}
    )

    # Add conditional edges for result validation
    def check_no_results(state):
        return not state.get("ranked_results")

    def check_quality_issues(state):
        return bool(state.get("metadata", {}).get("result_quality_issues"))

    # First condition: check for no results
    graph.add_conditional_edges(
        "validate_results",
        check_no_results,
        {True: "handle_no_results", False: "check_quality"}  # Route to intermediate node
    )

    # Add an intermediate check node
    graph.add_node("check_quality", lambda x: x)  # Identity function, passes state through

    # Second condition: check for quality issues
    graph.add_conditional_edges(
        "check_quality",
        check_quality_issues,
        {True: "handle_quality_issues", False: "build_response"}
    )

    # Connect all endpoints to telemetry
    graph.add_edge("handle_validation_error", "add_telemetry")
    graph.add_edge("handle_no_results", "add_telemetry")
    graph.add_edge("handle_quality_issues", "add_telemetry")
    graph.add_edge("build_response", "add_telemetry")

    # Connect telemetry to end
    graph.add_edge("add_telemetry", END)

    # Set entry point
    graph.set_entry_point("validate_input")

    logger.info("Search pipeline graph built successfully")
    return graph.compile()
