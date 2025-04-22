"""
Vector store implementation for semantic search.
"""
import os
import logging
import pickle
from typing import List, Dict, Any, Tuple, Optional

from langchain_community.vectorstores import FAISS
import numpy as np

from config import VECTOR_STORE_CONFIG
from utils.llm import get_embeddings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for semantic product search.
    """
    
    def __init__(self, index_name: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            index_name: Optional custom index name
        """
        self.embeddings = get_embeddings()
        self.index_name = index_name or VECTOR_STORE_CONFIG["index_name"]
        self.vector_store = None
        self.index_path = f"data/indexes/{self.index_name}.pkl"
        
        # Try to load existing index
        self._load_or_create_index()
        
    def _load_or_create_index(self):
        """Load an existing index or create a new one if it doesn't exist."""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Loading vector store from {self.index_path}")
                with open(self.index_path, "rb") as f:
                    self.vector_store = pickle.load(f)
            else:
                logger.info("Creating new vector store")
                self.vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index_name=self.index_name
                )
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                self._save_index()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            # Fallback to in-memory vector store
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index_name=self.index_name
            )
    
    def _save_index(self):
        """Save the index to disk."""
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.vector_store, f)
            logger.info(f"Saved vector store to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
    
    def add_products(self, products: List[Dict[str, Any]]):
        """
        Add products to the vector store.
        
        Args:
            products: List of product dictionaries
        """
        try:
            # Create text representations
            texts = []
            metadatas = []
            
            for product in products:
                # Create rich text representation for embedding
                text = f"{product.get('name', '')} {product.get('description', '')} "
                
                # Add attributes if present
                if 'attributes' in product:
                    for attr_name, attr_values in product['attributes'].items():
                        if isinstance(attr_values, list):
                            attr_text = ' '.join([str(v) for v in attr_values])
                        else:
                            attr_text = str(attr_values)
                        text += f"{attr_name}: {attr_text} "
                
                texts.append(text)
                metadatas.append(product)
            
            # Add to vector store
            self.vector_store.add_texts(texts, metadatas=metadatas)
            logger.info(f"Added {len(products)} products to vector store")
            
            # Save updated index
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error adding products to vector store: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the vector store for products matching the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (product, score) tuples
        """
        try:
            # Search vector store
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                product = doc.metadata
                formatted_results.append((product, float(score)))
            
            logger.info(f"Found {len(formatted_results)} results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def update_product(self, product_id: str, updated_product: Dict[str, Any]):
        """
        Update a product in the vector store.
        
        Args:
            product_id: ID of the product to update
            updated_product: Updated product data
        """
        try:
            # For FAISS, we need to remove and re-add since it doesn't support updates
            # In a production system, you'd want to use a vector DB with update support
            
            # Get all products
            all_products = self._get_all_products()
            
            # Filter out the product to update
            filtered_products = [p for p in all_products if p.get('id') != product_id]
            
            # Add the updated product
            filtered_products.append(updated_product)
            
            # Rebuild the index
            self._rebuild_index(filtered_products)
            
            logger.info(f"Updated product {product_id} in vector store")
            
        except Exception as e:
            logger.error(f"Error updating product in vector store: {str(e)}")
            raise
    
    def delete_product(self, product_id: str):
        """
        Delete a product from the vector store.
        
        Args:
            product_id: ID of the product to delete
        """
        try:
            # Get all products
            all_products = self._get_all_products()
            
            # Filter out the product to delete
            filtered_products = [p for p in all_products if p.get('id') != product_id]
            
            # Rebuild the index
            self._rebuild_index(filtered_products)
            
            logger.info(f"Deleted product {product_id} from vector store")
            
        except Exception as e:
            logger.error(f"Error deleting product from vector store: {str(e)}")
            raise
    
    def _get_all_products(self) -> List[Dict[str, Any]]:
        """
        Get all products from the vector store.
        
        Returns:
            List of all product dictionaries
        """
        # For FAISS, we need to iterate through the docstore
        products = []
        for doc_id in self.vector_store.docstore._dict:
            doc = self.vector_store.docstore.search(doc_id)
            if hasattr(doc, 'metadata'):
                products.append(doc.metadata)
        
        return products
    
    def _rebuild_index(self, products: List[Dict[str, Any]]):
        """
        Rebuild the index with the given products.
        
        Args:
            products: List of product dictionaries to include in the index
        """
        # Create a new vector store
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index_name=self.index_name
        )
        
        # Add all products
        self.add_products(products)