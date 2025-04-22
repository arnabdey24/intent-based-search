"""
Service for collecting telemetry and monitoring data.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from utils.monitoring import SearchSystemMonitor

logger = logging.getLogger(__name__)

class TelemetryService:
    """Service for collecting telemetry and monitoring search performance."""
    
    def __init__(self):
        """Initialize the telemetry service."""
        logger.info("Initializing telemetry service")
        self.monitor = SearchSystemMonitor()
        
        # In-memory storage for events - would be replaced with proper time series DB
        self._search_events = []
        self._error_events = []
        self._feedback_events = []
    
    async def log_search(self, 
                         query: str, 
                         intent: str, 
                         execution_time: float,
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         error: Optional[str] = None):
        """
        Log a search event.
        
        Args:
            query: The search query
            intent: Classified intent
            execution_time: Time taken to execute in seconds
            user_id: Optional user identifier
            session_id: Optional session identifier
            error: Optional error message if the search failed
        """
        # Create event
        event = {
            "timestamp": time.time(),
            "query": query,
            "intent": intent,
            "execution_time": execution_time,
            "user_id": user_id,
            "session_id": session_id,
            "error": error
        }
        
        # Store event
        self._search_events.append(event)
        
        # Update monitor
        mock_result = {"intent": intent, "error": error}
        self.monitor.log_search(query, mock_result, execution_time)
        
        logger.debug(f"Logged search event: '{query}', intent={intent}, time={execution_time:.2f}s")
    
    async def log_error(self, 
                        error_type: str, 
                        error_message: str,
                        query: Optional[str] = None,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None):
        """
        Log an error event.
        
        Args:
            error_type: Type of error
            error_message: Error message
            query: Optional search query that caused the error
            user_id: Optional user identifier
            session_id: Optional session identifier
        """
        # Create event
        event = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_message": error_message,
            "query": query,
            "user_id": user_id,
            "session_id": session_id
        }
        
        # Store event
        self._error_events.append(event)
        
        logger.debug(f"Logged error event: {error_type}, message: {error_message}")
    
    async def log_feedback(self,
                          request_id: str,
                          rating: int,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          feedback_text: Optional[str] = None,
                          selected_product_id: Optional[str] = None):
        """
        Log user feedback.
        
        Args:
            request_id: The search request ID
            rating: Rating from 1-5
            user_id: Optional user identifier
            session_id: Optional session identifier
            feedback_text: Optional feedback text
            selected_product_id: Optional ID of the product selected by the user
        """
        # Create event
        event = {
            "timestamp": time.time(),
            "request_id": request_id,
            "rating": rating,
            "user_id": user_id,
            "session_id": session_id,
            "feedback_text": feedback_text,
            "selected_product_id": selected_product_id
        }
        
        # Store event
        self._feedback_events.append(event)
        
        logger.debug(f"Logged feedback event: request_id={request_id}, rating={rating}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health metrics.
        
        Returns:
            Dictionary of health metrics
        """
        return self.monitor.get_system_health()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get detailed performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.monitor.get_performance_report()
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent error events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent error events
        """
        # Sort by timestamp, most recent first
        sorted_errors = sorted(
            self._error_events,
            key=lambda e: e.get("timestamp", 0),
            reverse=True
        )
        
        # Return limited number
        return sorted_errors[:limit]
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about user feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        if not self._feedback_events:
            return {
                "count": 0,
                "average_rating": 0,
                "rating_distribution": {}
            }
        
        # Calculate statistics
        ratings = [event.get("rating", 0) for event in self._feedback_events]
        rating_counts = {}
        
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        return {
            "count": len(self._feedback_events),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "rating_distribution": rating_counts
        }