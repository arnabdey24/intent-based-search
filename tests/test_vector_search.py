"""
Tests for the vector search functionality.
"""
import unittest
import sys
import os
import pickle
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch
import logging

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectordb.vector_store import VectorStore
from vectordb.embeddings import EmbeddingGenerator
from vectordb.index import IndexManager

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestVectorStore(unittest.TestCase):
    """Tests for VectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp dir for test indexes
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Patch the index path to use temp directory
        self.index_path_patcher = patch('vectordb.vector_store.os.path.dirname', 
                                        return_value=self.temp_dir.name)
        self.index_path_patcher.start()
        
        # Mock embeddings model
        self.embeddings_patcher = patch('vectordb.vector_store.get_embeddings')
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_query.return_value = np.random.rand(1536)
        self.mock_embeddings.embed_documents.return_value = [np.random.rand(1536), np.random.rand(1536)]
        self.mock_get_embeddings = self.embeddings_patcher.start()
        self.mock_get_embeddings.return_value = self.mock_embeddings
        
        # Create test instance
        self.vector_store = VectorStore(index_name="test_index")
    
    def tearDown(self):
        """Clean up after tests."""
        self.index_path_patcher.stop()
        self.embeddings_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_add_products(self):
        """Test adding products to vector store."""
        test_products = [
            {"id": 1, "name": "Test Product 1", "description": "Description 1"},
            {"id": 2, "name": "Test Product 2", "description": "Description 2", 
             "attributes": {"color": ["red", "blue"]}}
        ]
        
        # Configure mock for add_texts
        self.vector_store.vector_store.add_texts = MagicMock()
        
        # Call method under test
        self.vector_store.add_products(test_products)
        
        # Verify add_texts was called with correct arguments
        self.vector_store.vector_store.add_texts.assert_called_once()
        args, kwargs = self.vector_store.vector_store.add_texts.call_args
        self.assertEqual(len(args[0]), 2)  # Two text representations
        self.assertEqual(len(kwargs['metadatas']), 2)  # Two metadata objects
    
    def test_search(self):
        """Test searching the vector store."""
        # Mock search results
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {"id": 1, "name": "Product 1"}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {"id": 2, "name": "Product 2"}
        
        # Configure mock for similarity_search_with_score
        self.vector_store.vector_store.similarity_search_with_score = MagicMock(
            return_value=[(mock_doc1, 0.95), (mock_doc2, 0.85)]
        )
        
        # Call method under test
        results = self.vector_store.search("test query", k=2)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0]["id"], 1)
        self.assertEqual(results[0][1], 0.95)
        self.assertEqual(results[1][0]["id"], 2)
        self.assertEqual(results[1][1], 0.85)
    
    def test_search_error(self):
        """Test searching with error."""
        # Configure mock to raise exception
        self.vector_store.vector_store.similarity_search_with_score = MagicMock(
            side_effect=Exception("Search error")
        )
        
        # Call method under test
        results = self.vector_store.search("test query")
        
        # Verify empty results on error
        self.assertEqual(results, [])
    
    def test_update_product(self):
        """Test updating a product in the vector store."""
        # Setup mock data
        product_id = "1"
        updated_product = {"id": "1", "name": "Updated Product", "description": "New description"}
        
        # Configure mocks
        self.vector_store._get_all_products = MagicMock(
            return_value=[
                {"id": "1", "name": "Old Product", "description": "Old description"},
                {"id": "2", "name": "Other Product"}
            ]
        )
        self.vector_store._rebuild_index = MagicMock()
        
        # Call method under test
        self.vector_store.update_product(product_id, updated_product)
        
        # Verify _rebuild_index was called with correct products
        args, _ = self.vector_store._rebuild_index.call_args
        rebuild_products = args[0]
        
        # Should have 2 products, with the first one updated
        self.assertEqual(len(rebuild_products), 2)
        self.assertEqual(rebuild_products[0]["id"], "2")  # Unchanged product
        self.assertEqual(rebuild_products[1]["id"], "1")  # Updated product
        self.assertEqual(rebuild_products[1]["name"], "Updated Product")

    def test_delete_product(self):
        """Test deleting a product from the vector store."""
        # Setup mock data
        product_id = "1"
        
        # Configure mocks
        self.vector_store._get_all_products = MagicMock(
            return_value=[
                {"id": "1", "name": "Product to Delete"},
                {"id": "2", "name": "Product to Keep"}
            ]
        )
        self.vector_store._rebuild_index = MagicMock()
        
        # Call method under test
        self.vector_store.delete_product(product_id)
        
        # Verify _rebuild_index was called with correct products
        args, _ = self.vector_store._rebuild_index.call_args
        rebuild_products = args[0]
        
        # Should have only 1 product (the one that wasn't deleted)
        self.assertEqual(len(rebuild_products), 1)
        self.assertEqual(rebuild_products[0]["id"], "2")


class TestEmbeddingGenerator(unittest.TestCase):
    """Tests for EmbeddingGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock embeddings model
        self.embeddings_patcher = patch('vectordb.embeddings.get_embeddings')
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_query.return_value = np.random.rand(1536)
        self.mock_embeddings.embed_documents.return_value = [
            np.random.rand(1536), np.random.rand(1536)
        ]
        self.mock_get_embeddings = self.embeddings_patcher.start()
        self.mock_get_embeddings.return_value = self.mock_embeddings
        
        # Create test instance
        self.generator = EmbeddingGenerator()
    
    def tearDown(self):
        """Clean up after tests."""
        self.embeddings_patcher.stop()
    
    def test_generate_product_embedding(self):
        """Test generating embedding for a single product."""
        test_product = {
            "id": "1",
            "name": "Test Product",
            "description": "A test product",
            "category": "Test",
            "brand": "Test Brand",
            "price": 99.99,
            "attributes": {
                "color": ["red", "blue"],
                "size": "medium"
            }
        }
        
        # Call method under test
        embedding = self.generator.generate_product_embedding(test_product)
        
        # Verify embedding
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 1536)  # Default embedding size
        
        # Verify LLM was called with product text
        args, _ = self.mock_embeddings.embed_query.call_args
        product_text = args[0]
        
        # Check that all key product attributes are in the text
        self.assertIn("Test Product", product_text)
        self.assertIn("test product", product_text)
        self.assertIn("Test Brand", product_text)
        self.assertIn("99.99", product_text)
        self.assertIn("red", product_text)
        self.assertIn("blue", product_text)
    
    def test_generate_bulk_embeddings(self):
        """Test generating embeddings for multiple products."""
        test_products = [
            {"id": "1", "name": "Product 1", "description": "Description 1"},
            {"id": "2", "name": "Product 2", "description": "Description 2"}
        ]
        
        # Call method under test
        embeddings = self.generator.generate_bulk_embeddings(test_products)
        
        # Verify embeddings
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], np.ndarray)
        self.assertEqual(len(embeddings[0]), 1536)
        
        # Verify LLM was called with product texts
        args, _ = self.mock_embeddings.embed_documents.call_args
        product_texts = args[0]
        
        self.assertEqual(len(product_texts), 2)
        self.assertIn("Product 1", product_texts[0])
        self.assertIn("Product 2", product_texts[1])
    
    def test_generate_query_embedding(self):
        """Test generating embedding for a search query."""
        test_query = "red running shoes"
        
        # Call method under test
        embedding = self.generator.generate_query_embedding(test_query)
        
        # Verify embedding
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 1536)
        
        # Verify LLM was called with query
        self.mock_embeddings.embed_query.assert_called_with(test_query)
    
    def test_error_handling(self):
        """Test error handling during embedding generation."""
        # Configure mock to raise exception
        self.mock_embeddings.embed_query.side_effect = Exception("Embedding error")
        
        # Call method under test
        embedding = self.generator.generate_product_embedding({"name": "Test"})
        
        # Should return zero vector on error
        self.assertEqual(np.sum(embedding), 0)
        self.assertEqual(len(embedding), 1536)


class TestIndexManager(unittest.TestCase):
    """Tests for IndexManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp dirs for test indexes and backups
        self.temp_dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.temp_dir.name, "indexes"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir.name, "backups", "test_index"), exist_ok=True)
        
        # Create a test index file
        self.index_path = os.path.join(self.temp_dir.name, "indexes", "test_index.pkl")
        with open(self.index_path, "wb") as f:
            pickle.dump("test index data", f)
        
        # Patch paths to use temp directory
        self.path_patcher = patch('vectordb.index.os.path.dirname', 
                                   return_value=self.temp_dir.name)
        self.path_patcher.start()
        
        # Create test instance
        self.index_manager = IndexManager(index_name="test_index")
        # Update paths to use temp directory
        self.index_manager.index_path = self.index_path
        self.index_manager.backup_dir = os.path.join(self.temp_dir.name, "backups", "test_index")
    
    def tearDown(self):
        """Clean up after tests."""
        self.path_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_create_backup(self):
        """Test creating a backup of the index."""
        # Call method under test
        backup_path = self.index_manager.create_backup(tag="test")
        
        # Verify backup file exists
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue("test_index" in backup_path)
        self.assertTrue("test" in backup_path)
        
        # Verify backup content
        with open(backup_path, "rb") as f:
            content = pickle.load(f)
            self.assertEqual(content, "test index data")
    
    def test_restore_from_backup(self):
        """Test restoring from a backup."""
        # Create a backup file
        backup_path = os.path.join(self.index_manager.backup_dir, "test_index_backup.pkl")
        with open(backup_path, "wb") as f:
            pickle.dump("backup data", f)
        
        # Modify the current index
        with open(self.index_path, "wb") as f:
            pickle.dump("current data", f)
        
        # Call method under test
        result = self.index_manager.restore_from_backup(backup_path)
        
        # Verify restore success
        self.assertTrue(result)
        
        # Verify index content was restored
        with open(self.index_path, "rb") as f:
            content = pickle.load(f)
            self.assertEqual(content, "backup data")