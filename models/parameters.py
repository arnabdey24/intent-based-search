"""
Parameter models for extracting structured data from search queries.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator

class PriceRange(BaseModel):
    """Model for price range parameters."""
    min: Optional[float] = None
    max: Optional[float] = None
    
    @validator('min', 'max')
    def validate_price(cls, v):
        """Ensure prices are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Price cannot be negative")
        return v

class SearchParameters(BaseModel):
    """Model for structured search parameters extracted from queries."""
    product_type: Optional[str] = None
    specific_product: Optional[str] = None
    attributes: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    price_range: Optional[PriceRange] = None
    brands: Optional[List[str]] = Field(default_factory=list)
    problems: Optional[List[str]] = Field(default_factory=list)
    comparison_items: Optional[List[str]] = Field(default_factory=list)

# Define valid intents for validation
VALID_INTENTS = [
    "PRODUCT_DISCOVERY",  # General browsing
    "SPECIFIC_PRODUCT",   # Looking for specific product
    "ATTRIBUTE_SEARCH",   # Searching by attributes
    "PROBLEM_SOLUTION",   # Describing a problem to solve
    "COMPARISON",         # Comparing products
    "PRICE_BASED",        # Price-focused search
    "AVAILABILITY"        # Stock checking
]