"""
Embedding generation for products in the vector database.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from utils.llm import get_embeddings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generator for creating embeddings from product data."""
    
    def __init__(self):
        """Initialize the embedding generator."""
        logger.info("Initializing embedding generator")
        self.embedding_model = get_embeddings()
    
    def generate_product_embedding(self, product: Dict[str, Any]) -> np.ndarray:
        """
        Generate an embedding for a single product.
        
        Args:
            product: Product data dictionary
            
        Returns:
            Embedding vector as numpy array
        """
        # Create rich text representation for embedding
        text = self._create_product_text(product)
        
        try:
            # Generate embedding
            embedding = self.embedding_model.embed_query(text)
            logger.debug(f"Generated embedding for product: {product.get('id', 'unknown')}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback (in production, handle this better)
            return np.zeros(384)  # Default embedding dimension
    
    def generate_bulk_embeddings(self, products: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple products.
        
        Args:
            products: List of product data dictionaries
            
        Returns:
            List of embedding vectors
        """
        # Create text representations
        texts = [self._create_product_text(product) for product in products]
        
        try:
            # Generate embeddings in bulk
            embeddings = self.embedding_model.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings in bulk")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating bulk embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [np.zeros(384) for _ in products]
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a search query.
        
        Args:
            query: The search query
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.embed_query(query)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(384)
    
    def _create_product_text(self, product: Dict[str, Any]) -> str:
        """
        Create a text representation of a product for embedding.
        
        Args:
            product: Product data dictionary
            
        Returns:
            Text representation
        """
        # Start with name and description
        text = f"{product.get('name', '')} {product.get('description', '')} "
        
        # Add category if present
        if "category" in product:
            text += f"Category: {product['category']} "
        
        # Add brand if present
        if "brand" in product:
            text += f"Brand: {product['brand']} "
        
        # Add price if present
        if "price" in product:
            text += f"Price: {product['price']} "
        
        # Add attributes if present
        if "attributes" in product:
            text += "Attributes: "
            for attr_name, attr_values in product["attributes"].items():
                if isinstance(attr_values, list):
                    attr_text = ", ".join([str(v) for v in attr_values])
                else:
                    attr_text = str(attr_values)
                text += f"{attr_name}: {attr_text}; "
        
        return text

# Singleton instance
embedding_generator = EmbeddingGenerator()


if __name__ == '__main__':

    # Example usage
    product = {
        "id": "123",
        "name": "Sample Product",
        "description": "This is a sample product description.",
        "category": "Electronics",
        "brand": "BrandX",
        "price": 99.99,
        "attributes": {
            "color": ["red", "blue"],
            "size": "M"
        }
    }

    embedding = embedding_generator.generate_product_embedding(product)
    print("Generated embedding:", embedding)
