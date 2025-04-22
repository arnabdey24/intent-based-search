"""
Parameter extraction component for the search pipeline.
"""
import json
import logging
from typing import Dict, Any, List

from models.state import SearchState
from models.parameters import SearchParameters, PriceRange
from utils.prompts import PARAMETER_EXTRACTION_PROMPT
from utils.llm import get_llm

logger = logging.getLogger(__name__)

def extract_parameters(state: SearchState) -> SearchState:
    """
    Extracts structured search parameters from the query based on the identified intent.
    
    Args:
        state: The current search state with intent
        
    Returns:
        Updated state with extracted parameters
    """
    query = state["query"]
    intent = state["intent"]
    
    logger.info(f"Extracting parameters for query: '{query}' with intent: {intent}")
    
    # Get parameters from LLM
    chain = PARAMETER_EXTRACTION_PROMPT | get_llm()
    
    try:
        params_text = chain.invoke({
            "query": query, 
            "intent": intent
        }).content
        
        logger.debug(f"Raw parameter extraction result: {params_text}")
        
        # Parameter validation with error handling
        try:
            # First pass - basic JSON validation
            raw_parameters = json.loads(params_text)
            
            # Second pass - validate with Pydantic model
            if "price_range" in raw_parameters and raw_parameters["price_range"]:
                price_range = PriceRange(**raw_parameters["price_range"])
                raw_parameters["price_range"] = price_range.dict(exclude_none=True)
            
            # Validate full parameters object
            parameters = SearchParameters(**raw_parameters).dict(exclude_none=True)
            
            # Additional sanitization
            sanitized_parameters = sanitize_parameters(parameters)
            
            parameter_validation_status = "SUCCESS"
            logger.info(f"Successfully extracted parameters: {sanitized_parameters}")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from parameter extraction: {params_text}")
            sanitized_parameters = {}
            parameter_validation_status = "JSON_PARSE_ERROR"
            
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}")
            sanitized_parameters = {}
            parameter_validation_status = f"VALIDATION_ERROR: {str(e)}"
            
    except Exception as e:
        logger.error(f"Parameter extraction failed: {str(e)}")
        sanitized_parameters = {}
        parameter_validation_status = f"EXTRACTION_ERROR: {str(e)}"
    
    # Update state with parameters and metadata
    return {
        **state, 
        "parameters": sanitized_parameters,
        "metadata": {
            **(state.get("metadata", {})),
            "parameter_extraction_status": parameter_validation_status,
            "parameter_count": len(sanitized_parameters)
        }
    }

def sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize parameter values to ensure consistent formatting.
    
    Args:
        parameters: The extracted parameters
        
    Returns:
        Sanitized parameters
    """
    sanitized = {}
    
    for key, value in parameters.items():
        # String sanitization
        if isinstance(value, str):
            sanitized[key] = value.strip()
            
        # List of strings sanitization
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            sanitized[key] = [item.strip() for item in value]
            
        # Dictionary sanitization
        elif isinstance(value, dict):
            sanitized[key] = {
                k: v.strip() if isinstance(v, str) else v
                for k, v in value.items()
            }
            
        # Pass through other values unchanged
        else:
            sanitized[key] = value
            
    return sanitized