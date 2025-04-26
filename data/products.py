"""
Product data handling for intent-based search system.
"""
import logging
import json
import os
from typing import List, Dict, Any, Optional
import uuid
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from config import DB_CONFIG

logger = logging.getLogger(__name__)


class ProductDataManager:
    """Manager for product data operations."""

    def __init__(self, data_file: str = "data/products.json"):
        """
        Initialize the product data manager.
        
        Args:
            data_file: Path to product data file (used only for file-based storage)
        """
        self.data_file = data_file
        self._products = []

        # Check if we should use database
        if os.environ.get("USE_DATABASE", "False").lower() == "true":
            self._use_db = True
            self._init_db_connection()
            logger.info("Using database for product data management")
        else:
            self._use_db = False
            self._load_products()
            logger.info("Using file-based storage for product data management")

    def _init_db_connection(self):
        """Initialize database connection and tables."""
        try:
            # Create engine
            engine = sa.create_engine(DB_CONFIG["connection_string"])

            # Define metadata
            metadata = sa.MetaData()

            # Define products table if it doesn't exist
            self.products_table = sa.Table(
                'products', metadata,
                sa.Column('id', sa.String(36), primary_key=True),
                sa.Column('name', sa.String(255), nullable=False),
                sa.Column('description', sa.Text),
                sa.Column('price', sa.Float),
                sa.Column('category', sa.String(100)),
                sa.Column('brand', sa.String(100)),
                sa.Column('stock', sa.Boolean, default=True),
                sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
                sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
            )

            # Create tables if they don't exist
            metadata.create_all(engine)

            # Create session factory
            self.Session = sessionmaker(bind=engine)

            logger.info("Database connection initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self._use_db = False
            self._load_products()

    def _load_products(self):
        """Load products from data file (used only for file-based storage)."""
        if self._use_db:
            return

        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self._products = json.load(f)
                logger.info(f"Loaded {len(self._products)} products from {self.data_file}")
            else:
                logger.warning(f"Product data file not found: {self.data_file}")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                # Initialize with empty product list
                self._products = []
                # Save empty file
                self._save_products()
        except Exception as e:
            logger.error(f"Error loading products: {str(e)}")
            self._products = []

    def _save_products(self):
        """Save products to data file (used only for file-based storage)."""
        if self._use_db:
            return

        try:
            with open(self.data_file, 'w') as f:
                json.dump(self._products, f, indent=2)
            logger.info(f"Saved {len(self._products)} products to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving products: {str(e)}")

    def get_all_products(self) -> List[Dict[str, Any]]:
        """
        Get all products.
        
        Returns:
            List of all product dictionaries
        """
        if self._use_db:
            try:
                session = self.Session()
                result = session.execute(sa.select(self.products_table))

                products = []
                for row in result:
                    product = {col: getattr(row, col) for col in row._mapping.keys()}
                    products.append(product)

                session.close()
                return products
            except Exception as e:
                logger.error(f"Error getting products from database: {str(e)}")
                return []
        else:
            return self._products.copy()

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a product by ID.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product dictionary or None if not found
        """
        if self._use_db:
            try:
                session = self.Session()
                result = session.execute(
                    sa.select(self.products_table).where(self.products_table.c.id == product_id)
                ).fetchone()
                session.close()

                if result:
                    product = {col: getattr(result, col) for col in result._mapping.keys()}
                    return product
                return None
            except Exception as e:
                logger.error(f"Error getting product from database: {str(e)}")
                return None
        else:
            for product in self._products:
                if str(product.get('id')) == str(product_id):
                    return product.copy()
            return None

    def add_product(self, product: Dict[str, Any]) -> str:
        """
        Add a new product.
        
        Args:
            product: Product data dictionary
            
        Returns:
            ID of the new product
        """
        # Generate ID if not provided
        if 'id' not in product:
            product['id'] = str(uuid.uuid4())
        else:
            product['id'] = str(product['id'])

        # Validate required fields
        if 'name' not in product:
            logger.error("Product name is required")
            return ""

        if self._use_db:
            try:
                session = self.Session()

                # Extract attributes for proper column mapping
                attributes = {}
                for key, value in list(product.items()):
                    if key not in ['id', 'name', 'description', 'price', 'category', 'brand', 'stock', 'created_at',
                                   'updated_at']:
                        attributes[key] = value
                        product.pop(key)

                # Add attributes field
                if attributes:
                    product['attributes'] = attributes
                elif 'attributes' not in product:
                    product['attributes'] = {}

                # Insert product
                session.execute(sa.insert(self.products_table).values(**product))
                session.commit()
                session.close()

                logger.info(f"Added product to database: {product['name']} (ID: {product['id']})")
                return product['id']
            except Exception as e:
                logger.error(f"Error adding product to database: {str(e)}")
                return ""
        else:
            # Add to products list
            self._products.append(product)

            # Save products
            self._save_products()

            logger.info(f"Added product to file storage: {product.get('name')} (ID: {product.get('id')})")
            return product['id']

    def update_product(self, product_id: str, updated_product: Dict[str, Any]) -> bool:
        """
        Update an existing product.
        
        Args:
            product_id: ID of product to update
            updated_product: Updated product data
            
        Returns:
            True if product was updated, False if not found
        """
        # Preserve ID
        updated_product['id'] = str(product_id)

        if self._use_db:
            try:
                session = self.Session()

                # Check if product exists
                result = session.execute(
                    sa.select(self.products_table.c.id).where(self.products_table.c.id == product_id)
                ).fetchone()

                if not result:
                    session.close()
                    logger.warning(f"Product not found for update: {product_id}")
                    return False

                # Extract attributes for proper column mapping
                attributes = {}
                for key, value in list(updated_product.items()):
                    if key not in ['id', 'name', 'description', 'price', 'category', 'brand', 'stock', 'created_at',
                                   'updated_at']:
                        attributes[key] = value
                        updated_product.pop(key)

                # Add attributes field
                if attributes:
                    updated_product['attributes'] = attributes

                # Update product
                session.execute(
                    sa.update(self.products_table)
                    .where(self.products_table.c.id == product_id)
                    .values(**updated_product)
                )

                session.commit()
                session.close()

                logger.info(f"Updated product in database: {updated_product.get('name', 'Unknown')} (ID: {product_id})")
                return True
            except Exception as e:
                logger.error(f"Error updating product in database: {str(e)}")
                return False
        else:
            for i, product in enumerate(self._products):
                if str(product.get('id')) == str(product_id):
                    # Update product
                    self._products[i] = updated_product
                    # Save products
                    self._save_products()

                    logger.info(f"Updated product in file storage: {updated_product.get('name')} (ID: {product_id})")
                    return True

            logger.warning(f"Product not found for update: {product_id}")
            return False

    def delete_product(self, product_id: str) -> bool:
        """
        Delete a product.
        
        Args:
            product_id: ID of product to delete
            
        Returns:
            True if product was deleted, False if not found
        """
        if self._use_db:
            try:
                session = self.Session()

                # Check if product exists
                result = session.execute(
                    sa.select(self.products_table.c.id).where(self.products_table.c.id == product_id)
                ).fetchone()

                if not result:
                    session.close()
                    logger.warning(f"Product not found for deletion: {product_id}")
                    return False

                # Delete product
                session.execute(
                    sa.delete(self.products_table)
                    .where(self.products_table.c.id == product_id)
                )

                session.commit()
                session.close()

                logger.info(f"Deleted product from database: {product_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting product from database: {str(e)}")
                return False
        else:
            initial_count = len(self._products)
            self._products = [p for p in self._products if str(p.get('id')) != str(product_id)]

            if len(self._products) < initial_count:
                # Save products
                self._save_products()

                logger.info(f"Deleted product from file storage: {product_id}")
                return True

            logger.warning(f"Product not found for deletion: {product_id}")
            return False

    def search_products(self, query: str) -> List[Dict[str, Any]]:
        """
        Basic keyword search for products.
        
        Args:
            query: Search query
            
        Returns:
            List of matching products
        """
        query = query.lower()

        if self._use_db:
            try:
                session = self.Session()

                # Build search query using LIKE for each searchable field
                search_query = sa.or_(
                    self.products_table.c.name.ilike(f"%{query}%"),
                    self.products_table.c.description.ilike(f"%{query}%"),
                    self.products_table.c.category.ilike(f"%{query}%"),
                    self.products_table.c.brand.ilike(f"%{query}%")
                )

                result = session.execute(
                    sa.select(self.products_table).where(search_query)
                )

                products = []
                for row in result:
                    product = {col: getattr(row, col) for col in row._mapping.keys()}
                    products.append(product)

                session.close()

                logger.info(f"Found {len(products)} products matching query in database: '{query}'")
                return products
            except Exception as e:
                logger.error(f"Error searching products in database: {str(e)}")
                return []
        else:
            results = []

            for product in self._products:
                if self._product_matches_query(product, query):
                    results.append(product.copy())

            logger.info(f"Found {len(results)} products matching query in file storage: '{query}'")
            return results

    def filter_products(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter products by criteria.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of matching products
        """
        if self._use_db:
            try:
                session = self.Session()

                # Build filter conditions
                conditions = []

                if 'category' in filters:
                    conditions.append(self.products_table.c.category == filters['category'])

                if 'brand' in filters:
                    conditions.append(self.products_table.c.brand == filters['brand'])

                if 'price_min' in filters:
                    conditions.append(self.products_table.c.price >= filters['price_min'])

                if 'price_max' in filters:
                    conditions.append(self.products_table.c.price <= filters['price_max'])

                # Execute query with filters
                if conditions:
                    query = sa.select(self.products_table).where(sa.and_(*conditions))
                else:
                    query = sa.select(self.products_table)

                result = session.execute(query)

                products = []
                for row in result:
                    product = {col: getattr(row, col) for col in row._mapping.keys()}

                    # Check for attribute filters if needed
                    if 'attributes' in filters and isinstance(product.get('attributes'), dict):
                        match = True
                        for attr_name, attr_value in filters['attributes'].items():
                            if not self._product_has_attribute(product, attr_name, attr_value):
                                match = False
                                break

                        if not match:
                            continue

                    products.append(product)

                session.close()
                logger.info(f"Found {len(products)} products matching filters in database")
                return products
            except Exception as e:
                logger.error(f"Error filtering products in database: {str(e)}")
                return []
        else:
            results = self._products.copy()

            # Apply filters
            if 'category' in filters:
                results = [p for p in results if p.get('category') == filters['category']]

            if 'brand' in filters:
                results = [p for p in results if p.get('brand') == filters['brand']]

            if 'price_min' in filters:
                results = [p for p in results if p.get('price', 0) >= filters['price_min']]

            if 'price_max' in filters:
                results = [p for p in results if p.get('price', 0) <= filters['price_max']]

            if 'attributes' in filters:
                for attr_name, attr_value in filters['attributes'].items():
                    results = [p for p in results if self._product_has_attribute(p, attr_name, attr_value)]

            logger.info(f"Found {len(results)} products matching filters in file storage")
            return results

    def _product_matches_query(self, product: Dict[str, Any], query: str) -> bool:
        """
        Check if a product matches a search query.
        
        Args:
            product: Product dictionary
            query: Search query (lowercase)
            
        Returns:
            True if product matches query
        """
        # Check name
        if query in product.get('name', '').lower():
            return True

        # Check description
        if query in product.get('description', '').lower():
            return True

        # Check brand
        if query in product.get('brand', '').lower():
            return True

        # Check category
        if query in product.get('category', '').lower():
            return True

        # Check attributes
        if 'attributes' in product:
            for attr_name, attr_values in product['attributes'].items():
                if isinstance(attr_values, list):
                    # Check if query matches any attribute value
                    for value in attr_values:
                        if query in str(value).lower():
                            return True
                else:
                    # Check if query matches attribute value
                    if query in str(attr_values).lower():
                        return True

        return False

    def _product_has_attribute(self, product: Dict[str, Any], attr_name: str, attr_value: Any) -> bool:
        """
        Check if a product has a specific attribute value.
        
        Args:
            product: Product dictionary
            attr_name: Attribute name
            attr_value: Attribute value to check
            
        Returns:
            True if product has attribute value
        """
        if 'attributes' not in product:
            return False

        if attr_name not in product['attributes']:
            return False

        product_attr = product['attributes'][attr_name]

        # Handle list attributes
        if isinstance(product_attr, list):
            # Check if attribute value is in list
            if isinstance(attr_value, list):
                # Check if any value in attr_value matches any value in product_attr
                return any(val in product_attr for val in attr_value)
            else:
                # Check if attr_value is in product_attr list
                return attr_value in product_attr
        else:
            # Direct comparison for non-list attributes
            return product_attr == attr_value

    def get_unique_categories(self) -> List[str]:
        """
        Get list of unique product categories.
        
        Returns:
            List of category names
        """
        if self._use_db:
            try:
                session = self.Session()
                result = session.execute(
                    sa.select(self.products_table.c.category).distinct()
                )

                categories = [row.category for row in result if row.category]
                session.close()

                return sorted(categories)
            except Exception as e:
                logger.error(f"Error getting categories from database: {str(e)}")
                return []
        else:
            categories = set()

            for product in self._products:
                if 'category' in product and product['category']:
                    categories.add(product['category'])

            return sorted(list(categories))

    def get_unique_brands(self) -> List[str]:
        """
        Get list of unique product brands.
        
        Returns:
            List of brand names
        """
        if self._use_db:
            try:
                session = self.Session()
                result = session.execute(
                    sa.select(self.products_table.c.brand).distinct()
                )

                brands = [row.brand for row in result if row.brand]
                session.close()

                return sorted(brands)
            except Exception as e:
                logger.error(f"Error getting brands from database: {str(e)}")
                return []
        else:
            brands = set()

            for product in self._products:
                if 'brand' in product and product['brand']:
                    brands.add(product['brand'])

            return sorted(list(brands))

    def get_price_range(self) -> Dict[str, float]:
        """
        Get min and max prices across all products.
        
        Returns:
            Dictionary with min and max prices
        """
        if self._use_db:
            try:
                session = self.Session()
                min_result = session.execute(sa.select(sa.func.min(self.products_table.c.price))).scalar()
                max_result = session.execute(sa.select(sa.func.max(self.products_table.c.price))).scalar()
                session.close()

                return {
                    "min": min_result if min_result is not None else 0,
                    "max": max_result if max_result is not None else 0
                }
            except Exception as e:
                logger.error(f"Error getting price range from database: {str(e)}")
                return {"min": 0, "max": 0}
        else:
            if not self._products:
                return {"min": 0, "max": 0}

            prices = [p.get('price', 0) for p in self._products if 'price' in p]

            if not prices:
                return {"min": 0, "max": 0}

            return {"min": min(prices), "max": max(prices)}

    def import_products(self, products: List[Dict[str, Any]]) -> int:
        """
        Import a list of products.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Number of products imported
        """
        count = 0

        if self._use_db:
            try:
                session = self.Session()

                for product in products:
                    # Skip products without required fields
                    if 'name' not in product:
                        logger.warning("Skipping product without name")
                        continue

                    # Generate ID if not provided
                    if 'id' not in product:
                        product['id'] = str(uuid.uuid4())

                    # Extract attributes for proper column mapping
                    attributes = {}
                    for key, value in list(product.items()):
                        if key not in ['id', 'name', 'description', 'price', 'category', 'brand', 'stock',
                                       'created_at', 'updated_at']:
                            attributes[key] = value
                            product.pop(key)

                    # Add attributes field
                    if attributes:
                        product['attributes'] = attributes
                    elif 'attributes' not in product:
                        product['attributes'] = {}

                    # Insert product
                    session.execute(sa.insert(self.products_table).values(**product))
                    count += 1

                session.commit()
                session.close()
                logger.info(f"Imported {count} products to database")
            except Exception as e:
                logger.error(f"Error importing products to database: {str(e)}")
                return 0
        else:
            # Add each product
            for product in products:
                # Skip products without required fields
                if 'name' not in product:
                    logger.warning("Skipping product without name")
                    continue

                # Generate ID if not provided
                if 'id' not in product:
                    product['id'] = str(uuid.uuid4())

                # Add to products list
                self._products.append(product)
                count += 1

            # Save products
            if count > 0:
                self._save_products()

            logger.info(f"Imported {count} products to file storage")

        return count

    def _generate_id(self) -> str:
        """
        Generate a unique product ID.
        
        Returns:
            Unique ID string
        """
        return str(uuid.uuid4())


# Singleton instance
product_manager = ProductDataManager()

if __name__ == '__main__':
    # Example usage
    # product_data = {
    #     "name": "Sample Product",
    #     "description": "This is a sample product.",
    #     "price": 19.99,
    #     "category": "Electronics",
    #     "brand": "BrandX",
    #     "attributes": {
    #         "color": ["red", "blue"],
    #         "size": "M"
    #     }
    # }
    #
    # product_id = product_manager.add_product(product_data)
    # print(f"Added product with ID: {product_id}")

    all_products = product_manager.get_all_products()
    print(f"All products: {all_products}")
