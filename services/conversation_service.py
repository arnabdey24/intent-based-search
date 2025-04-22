"""
Service for managing conversation context and session history.
"""
import logging
import time
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for managing conversation context and history."""
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize the conversation service.
        
        Args:
            cache_ttl: Time-to-live for session cache in seconds (default: 1 hour)
        """
        logger.info("Initializing conversation service")
        self.cache_ttl = cache_ttl
        
        # In-memory session store - would be replaced with Redis in production
        self._sessions = {}
        self._session_timestamps = {}
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Conversation history for the session
        """
        # Clean expired sessions
        self._clean_expired_sessions()
        
        # Get session history
        if session_id in self._sessions:
            logger.debug(f"Retrieved history for session: {session_id}")
            return self._sessions[session_id]
        
        # Return empty history for new sessions
        logger.debug(f"No existing history for session: {session_id}")
        return []
    
    async def update_session_history(self, session_id: str, history: List[Dict[str, Any]]):
        """
        Update conversation history for a session.
        
        Args:
            session_id: The session identifier
            history: Updated conversation history
        """
        # Store session history
        self._sessions[session_id] = history
        self._session_timestamps[session_id] = time.time()
        
        # Trim history if too long
        if len(history) > 10:
            # Keep the most recent 10 entries
            self._sessions[session_id] = history[-10:]
        
        logger.debug(f"Updated history for session: {session_id}, entries: {len(history)}")
    
    async def clear_session(self, session_id: str):
        """
        Clear the conversation history for a session.
        
        Args:
            session_id: The session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            
        if session_id in self._session_timestamps:
            del self._session_timestamps[session_id]
            
        logger.info(f"Cleared session: {session_id}")
    
    async def add_history_entry(self, session_id: str, entry: Dict[str, Any]):
        """
        Add a single entry to the conversation history.
        
        Args:
            session_id: The session identifier
            entry: History entry to add
        """
        # Get existing history
        history = await self.get_session_history(session_id)
        
        # Add new entry
        history.append(entry)
        
        # Update session
        await self.update_session_history(session_id, history)
    
    def _clean_expired_sessions(self):
        """Remove expired sessions from memory."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, timestamp in self._session_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if session_id in self._sessions:
                del self._sessions[session_id]
            del self._session_timestamps[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned {len(expired_sessions)} expired sessions")