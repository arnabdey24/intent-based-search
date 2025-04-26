"""
FastAPI implementation for the intent-based search system.
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, Header
from pydantic import BaseModel
import uvicorn
import time
import uuid

from services.search_service import SearchService
from services.conversation_service import ConversationService
from services.personalization_service import PersonalizationService
from services.telemetry_service import TelemetryService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Intent-Based Search API",
    description="API for intent-based ecommerce search",
    version="1.0.0"
)

# Initialize services
search_service = SearchService()
conversation_service = ConversationService()
personalization_service = PersonalizationService()
telemetry_service = TelemetryService()

# API Models
class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    enable_conversation: Optional[bool] = True
    enable_personalization: Optional[bool] = True

class SearchResponse(BaseModel):
    """Search response model."""
    response: str
    intent: str
    results: List[Dict[str, Any]]
    query: str
    enhanced_query: Optional[str] = None
    conversation_aware: bool = False
    error: Optional[str] = None
    request_id: str

class UserPreferencesRequest(BaseModel):
    """User preferences update request."""
    preferred_brands: Optional[List[str]] = None
    product_categories_of_interest: Optional[List[str]] = None
    price_sensitivity: Optional[str] = None

# Dependency for extracting user and session IDs
async def get_request_metadata(
    user_id: Optional[str] = Header(None, description="User ID for personalization"),
    session_id: Optional[str] = Header(None, description="Session ID for conversation tracking")
) -> Dict[str, Any]:
    """Extract and validate request metadata."""
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    return {
        "user_id": user_id,
        "session_id": session_id
    }

# API Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Intent-Based Search API"}

@app.post("/ai/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    metadata: Dict[str, Any] = Depends(get_request_metadata)
):
    """
    Execute a search query.
    
    Args:
        request: Search request object
        metadata: Request metadata from headers
        
    Returns:
        Search response
    """
    try:
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info(f"Search request: ID={request_id}, Query='{request.query}'")
        
        # Execute search
        start_time = time.time()
        result = await search_service.search(
            query=request.query,
            user_id=metadata.get("user_id"),
            session_id=metadata.get("session_id"),
            enable_conversation=request.enable_conversation,
            enable_personalization=request.enable_personalization
        )
        
        # Add request ID to response
        result["request_id"] = request_id
        
        # Log completion
        execution_time = time.time() - start_time
        logger.info(f"Search completed: ID={request_id}, Time={execution_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/preferences/{user_id}", response_model=Dict[str, Any])
async def get_preferences(user_id: str):
    """
    Get preferences for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        User preferences
    """
    try:
        preferences = await personalization_service.get_user_preferences(user_id)
        return preferences
        
    except Exception as e:
        logger.error(f"Error getting preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/ai/preferences/{user_id}", response_model=Dict[str, Any])
async def update_preferences(user_id: str, preferences: UserPreferencesRequest):
    """
    Update preferences for a user.
    
    Args:
        user_id: User identifier
        preferences: Updated preferences
        
    Returns:
        Updated user preferences
    """
    try:
        # Convert to dictionary
        preferences_dict = preferences.dict(exclude_unset=True)
        
        # Update preferences
        await personalization_service.update_user_preferences(user_id, preferences_dict)
        
        # Return updated preferences
        return await personalization_service.get_user_preferences(user_id)
        
    except Exception as e:
        logger.error(f"Error updating preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/ai/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a conversation session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    try:
        await conversation_service.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    try:
        # Get system health metrics
        health_metrics = telemetry_service.get_system_health()
        
        # Add basic status
        health_metrics["status"] = "healthy"
        health_metrics["timestamp"] = time.time()
        
        return health_metrics
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/ai/metrics")
async def get_metrics():
    """
    Get system metrics.
    
    Returns:
        System metrics
    """
    try:
        # Get performance report
        metrics = telemetry_service.get_performance_report()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoint for improving search quality
@app.post("/ai/feedback")
async def submit_feedback(
    request_id: str,
    rating: int = Query(..., ge=1, le=5, description="Rating from 1-5"),
    feedback: Optional[str] = None,
    selected_product_id: Optional[str] = None,
    metadata: Dict[str, Any] = Depends(get_request_metadata)
):
    """
    Submit feedback for a search request.
    
    Args:
        request_id: The search request ID
        rating: Rating from 1-5
        feedback: Optional feedback text
        selected_product_id: ID of the product selected by the user
        metadata: Request metadata
        
    Returns:
        Success message
    """
    try:
        # Log feedback
        await telemetry_service.log_feedback(
            request_id=request_id,
            user_id=metadata.get("user_id"),
            session_id=metadata.get("session_id"),
            rating=rating,
            feedback_text=feedback,
            selected_product_id=selected_product_id
        )
        
        # If user selected a product, learn from this behavior
        if metadata.get("user_id") and selected_product_id:
            # This would typically involve more complex logic to fetch the product
            # and update user preferences based on selection
            pass
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
