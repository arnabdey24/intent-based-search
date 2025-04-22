"""
Session management for intent-based search system.
"""
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional
import uuid

from config import REDIS_CONFIG

logger = logging.getLogger(__name__)

class SessionManager:
    """Manager for session data and conversation context."""
    
    def __init__(self, session_ttl: int = 3600):
        """
        Initialize the session manager.
        
        Args:
            session_ttl: Time-to-live for sessions in seconds (default: 1 hour)
        """
        self.session_ttl = session_ttl
        
        # Use Redis for session storage in production
        if os.environ.get("USE_REDIS", "False").lower() == "true":
            import redis
            try:
                self.redis = redis.Redis(
                    host=REDIS_CONFIG["host"],
                    port=REDIS_CONFIG["port"],
                    password=REDIS_CONFIG["password"],
                    db=REDIS_CONFIG["db"],
                    decode_responses=True
                )
                # Test connection
                self.redis.ping()
                self._use_redis = True
                logger.info("Using Redis for session management")
            except Exception as e:
                logger.error(f"Redis connection failed: {str(e)}")
                self._use_redis = False
                self._sessions = {}
                self._session_timestamps = {}
                logger.warning("Falling back to in-memory storage for session management")
        else:
            # In-memory session store for development
            self._sessions = {}
            self._session_timestamps = {}
            self._use_redis = False
            logger.info("Using in-memory storage for session management")
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier to associate with session
            
        Returns:
            Session ID
        """
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session data
        session_data = {
            "id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
            "conversation_history": []
        }
        
        if self._use_redis:
            try:
                # Store in Redis with expiration
                self.redis.setex(
                    f"session:{session_id}", 
                    self.session_ttl,
                    json.dumps(session_data)
                )
                # Store user-session mapping if user_id provided
                if user_id:
                    self.redis.sadd(f"user_sessions:{user_id}", session_id)
                    # Set expiration on user sessions set
                    self.redis.expire(f"user_sessions:{user_id}", self.session_ttl * 2)
            except Exception as e:
                logger.error(f"Redis error creating session: {str(e)}")
                # Fall back to in-memory
                self._sessions[session_id] = session_data
                self._session_timestamps[session_id] = time.time()
        else:
            # Store session in memory
            self._sessions[session_id] = session_data
            self._session_timestamps[session_id] = time.time()
        
        logger.info(f"Created session: {session_id} for user: {user_id or 'anonymous'}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found or expired
        """
        if self._use_redis:
            try:
                # Get from Redis
                session_data = self.redis.get(f"session:{session_id}")
                if not session_data:
                    logger.warning(f"Session not found in Redis: {session_id}")
                    return None
                    
                # Refresh expiration
                self.redis.expire(f"session:{session_id}", self.session_ttl)
                
                # Parse JSON
                return json.loads(session_data)
            except Exception as e:
                logger.error(f"Redis error getting session: {str(e)}")
                # Fall back to in-memory if it exists
                if session_id in self._sessions:
                    return self._sessions[session_id]
                return None
        else:
            # Clean expired sessions
            self._clean_expired_sessions()
            
            # Check if session exists
            if session_id not in self._sessions:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            # Update timestamp
            self._session_timestamps[session_id] = time.time()
            
            return self._sessions[session_id]
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            data: Updated session data
            
        Returns:
            True if session was updated, False if not found or expired
        """
        # Check if session exists
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        # Preserve ID and created_at
        data["id"] = session_id
        data["created_at"] = session.get("created_at", time.time())
        
        if self._use_redis:
            try:
                # Store in Redis with expiration
                self.redis.setex(
                    f"session:{session_id}", 
                    self.session_ttl,
                    json.dumps(data)
                )
                return True
            except Exception as e:
                logger.error(f"Redis error updating session: {str(e)}")
                # Fall back to in-memory if it exists
                if session_id in self._sessions:
                    self._sessions[session_id] = data
                    self._session_timestamps[session_id] = time.time()
                    return True
                return False
        else:
            # Update session
            self._sessions[session_id] = data
            self._session_timestamps[session_id] = time.time()
            
            logger.info(f"Updated session: {session_id}")
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        if self._use_redis:
            try:
                # Get user ID to clean up user sessions set
                session_data = self.redis.get(f"session:{session_id}")
                if session_data:
                    session = json.loads(session_data)
                    user_id = session.get("user_id")
                    
                    # Remove from user sessions set if user_id exists
                    if user_id:
                        self.redis.srem(f"user_sessions:{user_id}", session_id)
                
                # Delete session
                deleted = self.redis.delete(f"session:{session_id}")
                
                # Also remove from memory if it exists there
                if session_id in self._sessions:
                    del self._sessions[session_id]
                if session_id in self._session_timestamps:
                    del self._session_timestamps[session_id]
                
                logger.info(f"Deleted session from Redis: {session_id}")
                return deleted > 0
            except Exception as e:
                logger.error(f"Redis error deleting session: {str(e)}")
                # Fall back to in-memory deletion
                deleted = False
                if session_id in self._sessions:
                    del self._sessions[session_id]
                    deleted = True
                if session_id in self._session_timestamps:
                    del self._session_timestamps[session_id]
                    deleted = True
                return deleted
        else:
            deleted = False
            if session_id in self._sessions:
                del self._sessions[session_id]
                deleted = True
            if session_id in self._session_timestamps:
                del self._session_timestamps[session_id]
                deleted = True
                
            logger.info(f"Deleted session from memory: {session_id}")
            return deleted
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation history for the session
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
            
        return session.get("conversation_history", [])
    
    def add_conversation_entry(self, session_id: str, entry: Dict[str, Any]) -> bool:
        """
        Add a conversation entry to a session.
        
        Args:
            session_id: Session identifier
            entry: Conversation entry to add
            
        Returns:
            True if entry was added, False if session not found
        """
        session = self.get_session(session_id)
        
        if not session:
            return False
            
        # Add timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = time.time()
            
        # Create history list if not present
        if "conversation_history" not in session:
            session["conversation_history"] = []
            
        # Add entry to history
        session["conversation_history"].append(entry)
        
        # Limit history size
        if len(session["conversation_history"]) > 20:
            session["conversation_history"] = session["conversation_history"][-20:]
            
        # Update session
        return self.update_session(session_id, session)
    
    def clear_conversation_history(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if history was cleared, False if session not found
        """
        session = self.get_session(session_id)
        
        if not session:
            return False
            
        # Clear history
        session["conversation_history"] = []
        
        # Update session
        return self.update_session(session_id, session)
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        if self._use_redis:
            try:
                # Search for all session keys
                sessions = self.redis.keys("session:*")
                # Strip prefix
                return [s.replace("session:", "") for s in sessions]
            except Exception as e:
                logger.error(f"Redis error getting active sessions: {str(e)}")
                # Fall back to in-memory if Redis fails
                self._clean_expired_sessions()
                return list(self._sessions.keys())
        else:
            # Clean expired sessions
            self._clean_expired_sessions()
            
            return list(self._sessions.keys())
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """
        Get active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session IDs for the user
        """
        if self._use_redis:
            try:
                # Get session IDs from user's session set
                sessions = self.redis.smembers(f"user_sessions:{user_id}")
                
                # Verify sessions still exist (might have expired)
                active_sessions = []
                for session_id in sessions:
                    if self.redis.exists(f"session:{session_id}"):
                        active_sessions.append(session_id)
                    else:
                        # Clean up the set by removing expired session
                        self.redis.srem(f"user_sessions:{user_id}", session_id)
                
                return active_sessions
            except Exception as e:
                logger.error(f"Redis error getting user sessions: {str(e)}")
                # Fall back to in-memory search
                self._clean_expired_sessions()
                user_sessions = []
                for session_id, session in self._sessions.items():
                    if session.get("user_id") == user_id:
                        user_sessions.append(session_id)
                return user_sessions
        else:
            # Clean expired sessions
            self._clean_expired_sessions()
            
            # Find sessions for user
            user_sessions = []
            
            for session_id, session in self._sessions.items():
                if session.get("user_id") == user_id:
                    user_sessions.append(session_id)
                    
            return user_sessions
    
    def _clean_expired_sessions(self):
        """Remove expired sessions from memory."""
        # Only needed for in-memory storage
        if self._use_redis:
            return
            
        current_time = time.time()
        expired_sessions = []
        
        for session_id, timestamp in self._session_timestamps.items():
            if current_time - timestamp > self.session_ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if session_id in self._sessions:
                del self._sessions[session_id]
            del self._session_timestamps[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned {len(expired_sessions)} expired sessions")

# Singleton instance
session_manager = SessionManager()