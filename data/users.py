"""
User data handling for intent-based search system.
"""
import logging
import json
import os
import time
from typing import Dict, Any, List, Optional
import uuid
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from config import DB_CONFIG

logger = logging.getLogger(__name__)

class UserDataManager:
    """Manager for user data operations."""
    
    def __init__(self, data_file: str = "data/users.json"):
        """
        Initialize the user data manager.
        
        Args:
            data_file: Path to user data file (used only for file-based storage)
        """
        self.data_file = data_file
        self._users = {}
        
        # Default preferences to use for new or unknown users
        self._default_preferences = {
            "preferred_brands": [],
            "price_sensitivity": "medium",
            "product_categories_of_interest": []
        }
        
        # Check if we should use database
        if os.environ.get("USE_DATABASE", "False").lower() == "true":
            self._use_db = True
            self._init_db_connection()
            logger.info("Using database for user data management")
        else:
            self._use_db = False
            self._load_users()
            logger.info("Using file-based storage for user data management")
    
    def _init_db_connection(self):
        """Initialize database connection and tables."""
        try:
            # Create engine
            engine = sa.create_engine(DB_CONFIG["connection_string"])
            
            # Define metadata
            metadata = sa.MetaData()
            
            # Define users table if it doesn't exist
            self.users_table = sa.Table(
                'users', metadata,
                sa.Column('id', sa.String(36), primary_key=True),
                sa.Column('data', sa.JSON, nullable=False)
            )
            
            # Create tables if they don't exist
            metadata.create_all(engine)
            
            # Create session factory
            self.Session = sessionmaker(bind=engine)
            
            logger.info("Database connection initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self._use_db = False
            self._load_users()
    
    def _load_users(self):
        """Load users from data file (used only for file-based storage)."""
        if self._use_db:
            return
            
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self._users = json.load(f)
                logger.info(f"Loaded {len(self._users)} users from {self.data_file}")
            else:
                logger.warning(f"User data file not found: {self.data_file}")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                # Initialize with empty user dict
                self._users = {}
                # Save empty file
                self._save_users()
        except Exception as e:
            logger.error(f"Error loading users: {str(e)}")
            self._users = {}
    
    def _save_users(self):
        """Save users to data file (used only for file-based storage)."""
        if self._use_db:
            return
            
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self._users, f, indent=2)
            logger.info(f"Saved {len(self._users)} users to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving users: {str(e)}")
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User dictionary or None if not found
        """
        if self._use_db:
            try:
                session = self.Session()
                result = session.execute(
                    sa.select(self.users_table.c.data).where(self.users_table.c.id == user_id)
                ).fetchone()
                session.close()
                
                if result:
                    return result[0]
                return None
            except Exception as e:
                logger.error(f"Error getting user from database: {str(e)}")
                return None
        else:
            return self._users.get(user_id)
    
    def create_user(self, user_data: Dict[str, Any]) -> str:
        """
        Create a new user.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            ID of the new user
        """
        # Generate ID if not provided
        if 'id' not in user_data:
            user_id = str(uuid.uuid4())
        else:
            user_id = str(user_data['id'])
        
        # Ensure ID is in the data
        user_data['id'] = user_id
        
        # Add default preferences if not present
        if 'preferences' not in user_data:
            user_data['preferences'] = self._default_preferences.copy()
        
        if self._use_db:
            try:
                session = self.Session()
                session.execute(
                    sa.insert(self.users_table).values(
                        id=user_id,
                        data=user_data
                    )
                )
                session.commit()
                session.close()
                logger.info(f"Created user in database: {user_id}")
            except Exception as e:
                logger.error(f"Error creating user in database: {str(e)}")
        else:
            # Add to users dict
            self._users[user_id] = user_data
            # Save users
            self._save_users()
            logger.info(f"Created user in file storage: {user_id}")
        
        return user_id
    
    def update_user(self, user_id: str, updated_data: Dict[str, Any]) -> bool:
        """
        Update an existing user.
        
        Args:
            user_id: ID of user to update
            updated_data: Updated user data
            
        Returns:
            True if user was updated, False if not found
        """
        # Preserve ID
        updated_data['id'] = user_id

        if self._use_db:
            try:
                session = self.Session()
                
                # Check if user exists
                result = session.execute(
                    sa.select(self.users_table.c.id).where(self.users_table.c.id == user_id)
                ).fetchone()
                
                if not result:
                    session.close()
                    logger.warning(f"User not found for update: {user_id}")
                    return False
                
                # Update user
                session.execute(
                    sa.update(self.users_table)
                    .where(self.users_table.c.id == user_id)
                    .values(data=updated_data)
                )
                
                session.commit()
                session.close()
                logger.info(f"Updated user in database: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error updating user in database: {str(e)}")
                return False
        else:
            if user_id not in self._users:
                logger.warning(f"User not found for update: {user_id}")
                return False
            
            # Update user
            self._users[user_id] = updated_data
            
            # Save users
            self._save_users()
            
            logger.info(f"Updated user in file storage: {user_id}")
            return True
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: ID of user to delete
            
        Returns:
            True if user was deleted, False if not found
        """
        if self._use_db:
            try:
                session = self.Session()
                
                # Check if user exists
                result = session.execute(
                    sa.select(self.users_table.c.id).where(self.users_table.c.id == user_id)
                ).fetchone()
                
                if not result:
                    session.close()
                    logger.warning(f"User not found for deletion: {user_id}")
                    return False
                
                # Delete user
                session.execute(
                    sa.delete(self.users_table)
                    .where(self.users_table.c.id == user_id)
                )
                
                session.commit()
                session.close()
                logger.info(f"Deleted user from database: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting user from database: {str(e)}")
                return False
        else:
            if user_id not in self._users:
                logger.warning(f"User not found for deletion: {user_id}")
                return False
            
            # Delete user
            del self._users[user_id]
            
            # Save users
            self._save_users()
            
            logger.info(f"Deleted user from file storage: {user_id}")
            return True
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get preferences for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences dictionary
        """
        user = self.get_user(user_id)
        
        if not user:
            # Return default preferences for unknown users
            return self._default_preferences.copy()
        
        return user.get('preferences', {})
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update preferences for a user.
        
        Args:
            user_id: User identifier
            preferences: Updated preferences
            
        Returns:
            True if preferences were updated, False if user not found
        """
        if self._use_db:
            try:
                session = self.Session()
                
                # Get current user data
                result = session.execute(
                    sa.select(self.users_table.c.data).where(self.users_table.c.id == user_id)
                ).fetchone()
                
                if not result:
                    session.close()
                    logger.warning(f"User not found for preference update: {user_id}")
                    return False
                
                # Update preferences
                user_data = result[0]
                if 'preferences' not in user_data:
                    user_data['preferences'] = {}
                
                # Merge with existing preferences
                user_data['preferences'].update(preferences)
                
                # Save updated user data
                session.execute(
                    sa.update(self.users_table)
                    .where(self.users_table.c.id == user_id)
                    .values(data=user_data)
                )
                
                session.commit()
                session.close()
                logger.info(f"Updated preferences for user in database: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error updating preferences in database: {str(e)}")
                return False
        else:
            user = self.get_user(user_id)
            
            if not user:
                logger.warning(f"User not found for preference update: {user_id}")
                return False
            
            # Update preferences
            if 'preferences' not in user:
                user['preferences'] = {}
                
            # Merge with existing preferences
            user['preferences'].update(preferences)
            
            # Save users
            self._save_users()
            
            logger.info(f"Updated preferences for user in file storage: {user_id}")
            return True
    
    def add_search_history(self, user_id: str, search_data: Dict[str, Any]) -> bool:
        """
        Add search to user's history.
        
        Args:
            user_id: User identifier
            search_data: Search data to add
            
        Returns:
            True if history was updated, False if user not found
        """
        # Add timestamp if not present
        if 'timestamp' not in search_data:
            search_data['timestamp'] = time.time()
            
        if self._use_db:
            try:
                session = self.Session()
                
                # Get current user data
                result = session.execute(
                    sa.select(self.users_table.c.data).where(self.users_table.c.id == user_id)
                ).fetchone()
                
                if not result:
                    session.close()
                    logger.warning(f"User not found for history update: {user_id}")
                    return False
                
                # Update search history
                user_data = result[0]
                if 'search_history' not in user_data:
                    user_data['search_history'] = []
                
                # Add search to history
                user_data['search_history'].append(search_data)
                
                # Limit history size
                if len(user_data['search_history']) > 100:
                    user_data['search_history'] = user_data['search_history'][-100:]
                
                # Save updated user data
                session.execute(
                    sa.update(self.users_table)
                    .where(self.users_table.c.id == user_id)
                    .values(data=user_data)
                )
                
                session.commit()
                session.close()
                logger.info(f"Added search to history for user in database: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error updating search history in database: {str(e)}")
                return False
        else:
            user = self.get_user(user_id)
            
            if not user:
                logger.warning(f"User not found for history update: {user_id}")
                return False
            
            # Create history list if not present
            if 'search_history' not in user:
                user['search_history'] = []
                
            # Add search to history
            user['search_history'].append(search_data)
            
            # Limit history size
            if len(user['search_history']) > 100:
                user['search_history'] = user['search_history'][-100:]
            
            # Save users
            self._save_users()
            
            logger.info(f"Added search to history for user in file storage: {user_id}")
            return True
    
    def get_search_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get search history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of history items to return
            
        Returns:
            List of search history items
        """
        user = self.get_user(user_id)
        
        if not user or 'search_history' not in user:
            return []
            
        # Return most recent searches first
        history = user['search_history'][::-1]
        
        # Apply limit
        return history[:limit]
    
    def clear_search_history(self, user_id: str) -> bool:
        """
        Clear search history for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if history was cleared, False if user not found
        """
        if self._use_db:
            try:
                session = self.Session()
                
                # Get current user data
                result = session.execute(
                    sa.select(self.users_table.c.data).where(self.users_table.c.id == user_id)
                ).fetchone()
                
                if not result:
                    session.close()
                    logger.warning(f"User not found for history clear: {user_id}")
                    return False
                
                # Clear history
                user_data = result[0]
                user_data['search_history'] = []
                
                # Save updated user data
                session.execute(
                    sa.update(self.users_table)
                    .where(self.users_table.c.id == user_id)
                    .values(data=user_data)
                )
                
                session.commit()
                session.close()
                logger.info(f"Cleared search history for user in database: {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error clearing search history in database: {str(e)}")
                return False
        else:
            user = self.get_user(user_id)
            
            if not user:
                logger.warning(f"User not found for history clear: {user_id}")
                return False
            
            # Clear history
            user['search_history'] = []
            
            # Save users
            self._save_users()
            
            logger.info(f"Cleared search history for user in file storage: {user_id}")
            return True

# Singleton instance
user_manager = UserDataManager()