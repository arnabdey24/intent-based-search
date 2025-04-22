"""
Service for managing user preferences and personalization.
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PersonalizationService:
    """Service for managing user preferences and personalization."""
    
    def __init__(self):
        """Initialize the personalization service."""
        logger.info("Initializing personalization service")
        
        # In-memory user preferences store - would be replaced with database in production
        self._user_preferences = {}
        
        # Default preferences to use for new or unknown users
        self._default_preferences = {
            "preferred_brands": [],
            "price_sensitivity": "medium",
            "product_categories_of_interest": []
        }
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get preferences for a user.
        
        Args:
            user_id: The user identifier
            
        Returns:
            User preferences
        """
        if user_id in self._user_preferences:
            logger.debug(f"Retrieved preferences for user: {user_id}")
            return self._user_preferences[user_id]
        
        # Return default preferences for new users
        logger.debug(f"No existing preferences for user: {user_id}, using defaults")
        return self._default_preferences.copy()
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        Update preferences for a user.
        
        Args:
            user_id: The user identifier
            preferences: Updated user preferences
        """
        # Get existing preferences
        existing = await self.get_user_preferences(user_id)
        
        # Merge with new preferences
        merged = {**existing, **preferences}
        
        # Store updated preferences
        self._user_preferences[user_id] = merged
        logger.info(f"Updated preferences for user: {user_id}")
    
    async def add_preferred_brand(self, user_id: str, brand: str):
        """
        Add a brand to a user's preferred brands.
        
        Args:
            user_id: The user identifier
            brand: Brand to add to preferences
        """
        preferences = await self.get_user_preferences(user_id)
        
        # Add brand if not already in list
        if brand not in preferences.get("preferred_brands", []):
            preferred_brands = preferences.get("preferred_brands", [])
            preferred_brands.append(brand)
            preferences["preferred_brands"] = preferred_brands
            
            # Update preferences
            await self.update_user_preferences(user_id, preferences)
            logger.info(f"Added preferred brand '{brand}' for user: {user_id}")
    
    async def add_category_of_interest(self, user_id: str, category: str):
        """
        Add a product category to a user's interests.
        
        Args:
            user_id: The user identifier
            category: Product category to add to interests
        """
        preferences = await self.get_user_preferences(user_id)
        
        # Add category if not already in list
        if category not in preferences.get("product_categories_of_interest", []):
            categories = preferences.get("product_categories_of_interest", [])
            categories.append(category)
            preferences["product_categories_of_interest"] = categories
            
            # Update preferences
            await self.update_user_preferences(user_id, preferences)
            logger.info(f"Added category of interest '{category}' for user: {user_id}")
    
    async def learn_from_search(self, user_id: str, query: str, selected_products: List[Dict[str, Any]]):
        """
        Learn preferences from search behavior.
        
        Args:
            user_id: The user identifier
            query: The search query
            selected_products: Products the user selected/clicked
        """
        if not selected_products:
            return
            
        # Extract potential preferences from selected products
        brands = set()
        categories = set()
        price_points = []
        
        for product in selected_products:
            # Extract brand
            if "brand" in product:
                brands.add(product["brand"])
                
            # Extract category
            if "category" in product:
                categories.add(product["category"])
                
            # Extract price
            if "price" in product:
                price_points.append(product["price"])
        
        # Update preferences if we found significant data
        preferences = {}
        
        if brands:
            preferences["recently_selected_brands"] = list(brands)
            
        if categories:
            preferences["recently_viewed_categories"] = list(categories)
            
        if price_points:
            avg_price = sum(price_points) / len(price_points)
            preferences["recent_price_point"] = avg_price
        
        if preferences:
            await self.update_user_preferences(user_id, preferences)
            logger.info(f"Learned preferences from search behavior for user: {user_id}")