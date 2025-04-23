"""
Vector store implementation using Qdrant for semantic search.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import numpy as np

from config import VECTOR_STORE_CONFIG
from utils.llm import get_embeddings
from vectordb.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for semantic product search using Qdrant.
    """

    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the Qdrant vector store.

        Args:
            collection_name: Optional custom collection name
        """
        self.embeddings = get_embeddings()
        self.embedding_generator = EmbeddingGenerator()
        self.collection_name = collection_name or VECTOR_STORE_CONFIG.get("collection_name", "products")
        self.dimension = VECTOR_STORE_CONFIG.get("dimension", 768)

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=VECTOR_STORE_CONFIG.get("url", "http://localhost:6333"),
            api_key=VECTOR_STORE_CONFIG.get("api_key", "")
        )

        # Initialize collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating new Qdrant collection: {self.collection_name}")

                # Create a new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.dimension,
                        distance=qdrant_models.Distance.COSINE
                    )
                )

                # Create schema with payload indexing
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="category",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="brand",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="price",
                    field_schema=qdrant_models.PayloadSchemaType.FLOAT
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="in_stock",
                    field_schema=qdrant_models.PayloadSchemaType.BOOL
                )

                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            raise

    def add_products(self, products: List[Dict[str, Any]]):
        """
        Add products to the vector store.

        Args:
            products: List of product dictionaries
        """
        try:
            points = []

            for product in products:
                # Generate embedding
                embedding = self.embedding_generator.generate_product_embedding(product)

                # Create point ID (use existing ID or generate a new one)
                point_id = product.get('id')
                if not point_id:
                    point_id = str(uuid.uuid4())
                    product['id'] = point_id

                # Convert string ID to UUID if needed
                if isinstance(point_id, str):
                    try:
                        uuid.UUID(point_id)
                    except ValueError:
                        # If not a valid UUID, create a deterministic UUID from the string
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

                # Create Qdrant point
                points.append(
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=product
                    )
                )

            # Add points to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(products)} products to Qdrant collection")

        except Exception as e:
            logger.error(f"Error adding products to Qdrant: {str(e)}")
            raise

    def search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the vector store for products matching the query.

        Args:
            query: The search query
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of (product, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)

            # Create filter if specified
            filter_query = None
            if filters:
                filter_conditions = []

                if 'category' in filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="category",
                            match=qdrant_models.MatchValue(value=filters['category'])
                        )
                    )

                if 'brand' in filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="brand",
                            match=qdrant_models.MatchValue(value=filters['brand'])
                        )
                    )

                if 'price_min' in filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="price",
                            range=qdrant_models.Range(
                                gte=filters['price_min']
                            )
                        )
                    )

                if 'price_max' in filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="price",
                            range=qdrant_models.Range(
                                lte=filters['price_max']
                            )
                        )
                    )

                if 'in_stock' in filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="in_stock",
                            match=qdrant_models.MatchValue(value=filters['in_stock'])
                        )
                    )

                if filter_conditions:
                    filter_query = qdrant_models.Filter(
                        must=filter_conditions
                    )

            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=filter_query
            )

            # Format results
            results = []
            for scored_point in search_result:
                product = scored_point.payload
                score = scored_point.score
                results.append((product, score))

            logger.info(f"Found {len(results)} results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            return []

    def update_product(self, product_id: str, updated_product: Dict[str, Any]):
        """
        Update a product in the vector store.

        Args:
            product_id: ID of product to update
            updated_product: Updated product data
        """
        try:
            # Generate embedding
            embedding = self.embedding_generator.generate_product_embedding(updated_product)

            # Ensure product_id is in updated_product
            updated_product['id'] = product_id

            # Convert string ID to UUID if needed
            if isinstance(product_id, str):
                try:
                    point_id = uuid.UUID(product_id)
                except ValueError:
                    # If not a valid UUID, create a deterministic UUID from the string
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, product_id)
            else:
                point_id = product_id

            # Update point in collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=updated_product
                    )
                ]
            )

            logger.info(f"Updated product {product_id} in Qdrant")

        except Exception as e:
            logger.error(f"Error updating product in Qdrant: {str(e)}")
            raise

    def delete_product(self, product_id: str):
        """
        Delete a product from the vector store.

        Args:
            product_id: ID of product to delete
        """
        try:
            # Convert string ID to UUID if needed
            if isinstance(product_id, str):
                try:
                    point_id = uuid.UUID(product_id)
                except ValueError:
                    # If not a valid UUID, create a deterministic UUID from the string
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, product_id)
            else:
                point_id = product_id

            # Delete point from collection
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[point_id]
                )
            )

            logger.info(f"Deleted product {product_id} from Qdrant")

        except Exception as e:
            logger.error(f"Error deleting product from Qdrant: {str(e)}")
            raise

    def get_count(self) -> int:
        """
        Get the number of products in the vector store.

        Returns:
            Number of products
        """
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return collection_info.vectors_count

        except Exception as e:
            logger.error(f"Error getting count from Qdrant: {str(e)}")
            return 0


# Singleton instance
vector_store = VectorStore()

if __name__ == '__main__':
    # Example usage
    product = {
        "id": "9a077fe9-5360-4b25-b7bf-1d893f50111c",
        "name": "Sample Product",
        "description": "This is a sample product.",
        "category": "Electronics",
        "brand": "BrandX",
        "price": 99.99,
        "in_stock": True
    }

    vector_store.add_products([product])
    results = vector_store.search("sample product")
    print(results)
