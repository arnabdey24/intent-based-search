"""
Vector store implementation for semantic search.
"""
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

import faiss
from langchain_community.docstore import InMemoryDocstore
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
        self.index_path = f"data/indexes/{self.index_name}"

        # Try to load existing index or create new one
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load an existing FAISS index or create a new one."""
        try:
            if os.path.exists(os.path.join(self.index_path, "index.faiss")):
                logger.info(f"Loading vector store from {self.index_path}")
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            else:
                logger.info("Creating new vector store")
                self.vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=faiss.IndexFlatL2(len(self.embeddings.embed_query("Test"))),
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
                self._save_index()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            # fallback to empty in-memory store
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=faiss.IndexFlatL2(len(self.embeddings.embed_query("Test"))),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

    def _save_index(self):
        """Save the FAISS index and metadata to disk using LangChain's save_local."""
        try:
            if self.vector_store:
                os.makedirs(self.index_path, exist_ok=True)
                self.vector_store.save_local(self.index_path)
                logger.info(f"Saved vector store to {self.index_path}")
            else:
                logger.warning("Vector store is None. Skipping save.")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")

    def add_products(self, products: List[Dict[str, Any]]):
        """
        Add products to the vector store.

        Args:
            products: List of product dictionaries
        """
        try:
            texts = []
            metadatas = []

            for product in products:
                text = f"{product.get('name', '')} {product.get('description', '')} "

                if 'attributes' in product:
                    for attr_name, attr_values in product['attributes'].items():
                        if isinstance(attr_values, list):
                            attr_text = ' '.join(map(str, attr_values))
                        else:
                            attr_text = str(attr_values)
                        text += f"{attr_name}: {attr_text} "

                texts.append(text)
                metadatas.append(product)

            self.vector_store.add_texts(texts, metadatas=metadatas)
            logger.info(f"Added {len(products)} products to vector store")

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
            results = self.vector_store.similarity_search_with_score(query, k=k)

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
            all_products = self._get_all_products()
            filtered_products = [p for p in all_products if p.get('id') != product_id]
            filtered_products.append(updated_product)
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
            all_products = self._get_all_products()
            filtered_products = [p for p in all_products if p.get('id') != product_id]
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
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(len(self.embeddings.embed_query("Test"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        self.add_products(products)


if __name__ == '__main__':
    # Example usage
    vector_store = VectorStore()

    # Add products
    products = [
        {"id": "1", "name": "Product 1", "description": "Description 1", "attributes": {"color": ["red", "blue"]}},
        {"id": "2", "name": "Product 2", "description": "Description 2", "attributes": {"size": ["S", "M"]}}
    ]
    vector_store.add_products(products)

    # Search
    results = vector_store.search("red product")
    print(results)

    # Update product
    vector_store.update_product("1", {"id": "1", "name": "Updated Product 1", "description": "Updated Description"})

    # Delete product
    vector_store.delete_product("2")
