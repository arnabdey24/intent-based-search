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
from pipeline.telemetry import add_telemetry

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
    """
    Create the LangGraph for our search system.
    
    Returns:
        Compiled graph ready for execution
    """
    logger.info("Building search pipeline graph")
    
    # Initialize with empty state
    graph = StateGraph(SearchState)
    
    # Add nodes - Input validation
    graph.add_node("validate_input", validate_input)
    graph.add_node("handle_validation_error", handle_validation_error)
    
    # Add nodes - Core search flow
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("extract_parameters", extract_parameters)
    graph.add_node("enhance_query", enhance_query)
    graph.add_node("retrieve_results", retrieve_results)
    graph.add_node("rank_results", rank_results)
    
    # Add nodes - Output validation and quality handling
    graph.add_node("validate_results", validate_results)
    graph.add_node("handle_no_results", handle_no_results)
    graph.add_node("handle_quality_issues", handle_quality_issues)
    graph.add_node("build_response", build_response)
    
    # Add telemetry node - runs at the end of every path
    graph.add_node("add_telemetry", add_telemetry)
    
    # Add edges
    edges = define_edges()
    for start, end in edges.items():
        if isinstance(end, dict):
            # Handle conditional edges
            for condition, target in end.items():
                if condition == "DEFAULT":
                    graph.add_edge(start, target)
                else:
                    graph.add_conditional_edges(
                        start,
                        lambda state, condition=condition: 
                            state.get("input_validation_error") is not None 
                            if condition == "handle_validation_error" 
                            else condition,
                        {True: condition, False: end["DEFAULT"]}
                    )
        elif end == END:
            graph.add_edge(start, END)
        else:
            graph.add_edge(start, end)
    
    # Set entry point
    graph.set_entry_point("validate_input")
    
    logger.info("Search pipeline graph built successfully")
    return graph.compile()