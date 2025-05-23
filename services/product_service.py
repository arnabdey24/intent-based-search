from typing import Dict, Any
import logging
from data.products import product_manager
from vectordb.vector_store import vector_store

logger = logging.getLogger(__name__)

class ProductService:
    """Service for managing products across all data stores."""

    def __init__(self):
        self.db_manager = product_manager  # Your SQL database manager
        self.vector_db = vector_store      # Your Qdrant vector store

    async def create_product(self, product_data: Dict[str, Any]) -> str:
        """Create product in both databases atomically."""
        try:
            # First add to primary database
            product_id = self.db_manager.add_product(product_data)

            if not product_id:
                raise ValueError("Failed to add product to primary database")

            # Then add to vector database
            self.vector_db.add_products([product_data])

            return product_id
        except Exception as e:
            # If vector DB fails, try to clean up primary DB
            if product_id:
                self.db_manager.delete_product(product_id)
            raise Exception(f"Failed to create product: {str(e)}")

    async def update_product(self, product_id: str, product_data: Dict[str, Any]) -> bool:
        """Update product in both databases."""
        try:
            # Update primary database
            success = self.db_manager.update_product(product_id, product_data)

            if not success:
                return False

            # Update vector database
            self.vector_db.update_product(product_id, product_data)

            return True
        except Exception as e:
            logger.error(f"Error updating product {product_id}: {str(e)}")
            return False

    async def delete_product(self, product_id: str) -> bool:
        """Delete product from both databases."""
        try:
            # Delete from primary database
            success = self.db_manager.delete_product(product_id)

            if not success:
                return False

            # Delete from vector database
            self.vector_db.delete_product(product_id)

            return True
        except Exception as e:
            logger.error(f"Error deleting product {product_id}: {str(e)}")
            return False

    def verify_sync(self) -> Dict[str, Any]:
        """Verify synchronization between databases and fix inconsistencies."""
        # Get all products from primary database
        db_products = self.db_manager.get_all_products()
        db_product_ids = {p["id"] for p in db_products}

        # Get count from vector database
        vector_db_count = self.vector_db.get_count()

        print(f"Products found in database: {len(db_products)}")
        print(f"Products found in vector DB: {vector_db_count}")

        results = {
            "primary_db_count": len(db_products),
            "vector_db_count": vector_db_count,
            "products_added": 0,
            "sync_status": "unknown"
        }

        if len(db_products) == vector_db_count:
            results["sync_status"] = "synchronized"
        else:
            # Resync all products to ensure consistency, but in batches of 1000
            total_added = 0
            batch_size = 1000

            for i in range(0, len(db_products), batch_size):
                # Get the current batch
                batch = db_products[i:i+batch_size]

                # Add batch to vector database
                self.vector_db.add_products(batch)

                # Update count
                total_added += len(batch)

                # Log progress
                print(f"Processed batch {i//batch_size + 1}: {len(batch)} products ({total_added}/{len(db_products)} total)")

            results["products_added"] = total_added
            results["sync_status"] = "resynced"
            print(f"Sync complete: {total_added} products added to vector database")

        return results



import asyncio

if __name__ == '__main__':

    p = ProductService()
    p.verify_sync()

