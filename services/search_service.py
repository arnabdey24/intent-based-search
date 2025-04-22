"""
Main search service for handling search requests.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from fastapi import HTTPException

from models.state import SearchState
from pipeline.graph import build_search_graph
from services.conversation_service import ConversationService
from services.personalization_service import PersonalizationService
from services.telemetry_service import TelemetryService
from config import get_config, FEATURES

logger = logging.getLogger(__name__)

class SearchService:
    """Service for handling search requests."""
    
    def __init__(self):
        """Initialize the search service."""
        logger.info("Initializing search service")
        self.search_executor = build_search_graph()
        self.conversation_service = ConversationService()
        self.personalization_service = PersonalizationService()
        self.telemetry_service = TelemetryService()
        self.config = get_config()
    
    async def search(self, 
                    query: str, 
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    enable_conversation: Optional[bool] = None,
                    enable_personalization: Optional[bool] = None) -> Dict[str, Any]:
        """
        Execute a search with the given query.
        
        Args:
            query: The user search query
            user_id: Optional user identifier for personalization
            session_id: Optional session identifier for conversation context
            enable_conversation: Override for conversation context feature
            enable_personalization: Override for personalization feature
            
        Returns:
            Search results with response
        """
        start_time = time.time()
        
        # Determine feature flags
        use_conversation = enable_conversation if enable_conversation is not None else FEATURES["use_conversation_context"]
        use_personalization = enable_personalization if enable_personalization is not None else FEATURES["use_personalization"]
        
        try:
            # Get conversation context if enabled
            conversation_context = {}
            if use_conversation and session_id:
                conversation_context["history"] = await self.conversation_service.get_session_history(session_id)
            
            # Get user preferences if enabled
            if use_personalization and user_id:
                user_preferences = await self.personalization_service.get_user_preferences(user_id)
                conversation_context["preferences"] = user_preferences
            
            # Execute search
            result = await self._execute_search(query, conversation_context)
            
            # Update conversation history if enabled
            if use_conversation and session_id and "conversation_history" in result:
                await self.conversation_service.update_session_history(
                    session_id, 
                    result["conversation_history"]
                )
            
            # Log telemetry
            execution_time = time.time() - start_time
            await self.telemetry_service.log_search(
                query=query,
                user_id=user_id,
                session_id=session_id,
                intent=result.get("intent", ""),
                execution_time=execution_time,
                error=result.get("error")
            )
            
            # Prepare response
            return self._prepare_response(result)
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            
            # Log error
            await self.telemetry_service.log_error(
                error_type="SEARCH_ERROR",
                error_message=str(e),
                query=query,
                user_id=user_id,
                session_id=session_id
            )
            
            # Raise HTTP exception
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )
    
    async def _execute_search(self, query: str, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the search pipeline.
        
        Args:
            query: The search query
            conversation_context: Context information
            
        Returns:
            Complete search result
        """
        # Preprocessing and query expansion
        processed_query = self._preprocess_query(query, conversation_context)
        
        # Initialize state
        initial_state = SearchState(
            query=processed_query,
            input_validation_error=None,
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            error=None,
            conversation_history=conversation_context.get("history", []),
            metadata={
                "query_timestamp": time.time(),
                "query_length": len(processed_query),
                "conversation_aware": bool(conversation_context.get("history")),
                "personalization_applied": bool(conversation_context.get("preferences"))
            }
        )
        
        # Execute search graph
        result = self.search_executor.invoke(initial_state)
        
        # Apply personalization if needed
        if conversation_context.get("preferences"):
            result = self._apply_personalization(result, conversation_context["preferences"])
        
        # Handle conversation tracking
        if conversation_context.get("history") is not None:
            result = self._update_conversation_tracking(result, query)
        
        # Final validation - ensure we have a response
        if not result.get("response"):
            result["response"] = "I couldn't find what you're looking for. Could you try a different search?"
            result["error"] = "EMPTY_RESPONSE"
        
        return result
    
    def _preprocess_query(self, query: str, conversation_context: Dict[str, Any]) -> str:
        """
        Preprocess the query with conversation awareness.
        
        Args:
            query: The original query
            conversation_context: Conversation context
            
        Returns:
            Processed query
        """
        # Strip whitespace
        processed_query = query.strip()
        
        # Handle reference resolution if there's conversation history
        if (conversation_context.get("history") and 
            any(term in query.lower() for term in ["that one", "those", "this product", "it", "that"])):
            
            # Find the most recent product mentioned
            history = conversation_context["history"]
            recent_products = []
            
            for message in reversed(history):
                if "product_references" in message:
                    recent_products = message["product_references"]
                    break
            
            # Expand query with product reference
            if recent_products and len(recent_products) > 0:
                most_recent = recent_products[0]
                processed_query = f"{query} like {most_recent.get('name', '')}"
                logger.info(f"Expanded reference query: '{query}' → '{processed_query}'")
        
        return processed_query
    
    def _apply_personalization(self, result: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply personalization to search results.
        
        Args:
            result: The search result
            preferences: User preferences
            
        Returns:
            Updated result with personalization
        """
        if not result.get("ranked_results"):
            return result
            
        # Example: Boost products from preferred brands
        preferred_brands = preferences.get("preferred_brands", [])
        if preferred_brands:
            for product in result["ranked_results"]:
                if product.get("brand") in preferred_brands:
                    # Boost relevance score for preferred brands
                    product["relevance_score"] = product.get("relevance_score", 0) * 1.2
            
            # Re-sort based on adjusted scores
            result["ranked_results"] = sorted(
                result["ranked_results"], 
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )
            
            # Update metadata
            result["metadata"]["personalization_brand_boost"] = preferred_brands
        
        return result
    
    def _update_conversation_tracking(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Add conversation tracking information to the result.
        
        Args:
            result: The search result
            original_query: The original user query
            
        Returns:
            Updated result with conversation tracking
        """
        # Get existing history
        history = result.get("conversation_history", [])
        
        # Create new history entry only if we have results
        if result.get("ranked_results"):
            top_products = result["ranked_results"][:3]
            product_references = [
                {"id": p.get("id"), "name": p.get("name")} 
                for p in top_products
            ]
            
            new_entry = {
                "query": original_query,
                "response": result.get("response", ""),
                "product_references": product_references,
                "timestamp": time.time()
            }
            
            # Add to history
            result["conversation_history"] = history + [new_entry]
        
        return result
    
    def _prepare_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the response object from the search result.
        
        Args:
            result: The raw search result
            
        Returns:
            Formatted response suitable for API return
        """
        # Extract essential information for the response
        response = {
            "response": result.get("response", ""),
            "intent": result.get("intent", ""),
            "results": result.get("ranked_results", []),
            "query": result.get("query", ""),
            "enhanced_query": result.get("enhanced_query"),
            "conversation_aware": result.get("metadata", {}).get("conversation_aware", False),
            "error": result.get("error")
        }
        
        return response