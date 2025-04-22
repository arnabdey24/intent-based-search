"""
Integration tests for the entire search pipeline.
"""
import unittest
import sys
import os
from unittest.mock import MagicMock, patch
import logging
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import execute_search, execute_conversation_search
from pipeline.graph import build_search_graph
from models.state import SearchState

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the full search pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock LLM
        self.llm_patcher = patch('utils.llm.get_llm')
        self.mock_get_llm = self.llm_patcher.start()
        self.mock_llm = MagicMock()
        self.mock_get_llm.return_value = self.mock_llm
        
        # Mock vector store
        self.vector_store_patcher = patch('pipeline.vector_search.vector_store')
        self.mock_vector_store = self.vector_store_patcher.start()
        
        # Mock time
        self.time_patcher = patch('main.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.time.return_value = 1000.0  # Fixed timestamp
        
        # Configure LLM mock for different prompts
        self.configure_llm_mocks()
        
        # Configure vector store mock
        self.configure_vector_store_mock()
    
    def tearDown(self):
        """Clean up after tests."""
        self.llm_patcher.stop()
        self.vector_store_patcher.stop()
        self.time_patcher.stop()
    
    def configure_llm_mocks(self):
        """Configure LLM mocks for different pipeline components."""
        # Using side_effect to return different values based on input
        def mock_llm_invoke(inputs):
            response = MagicMock()
            
            # Intent classification
            if "Categorize the query into ONE of these intents" in str(inputs):
                if "under $100" in str(inputs):
                    response.content = "PRICE_BASED"
                elif "Nike Air" in str(inputs):
                    response.content = "SPECIFIC_PRODUCT"
                else:
                    response.content = "PRODUCT_DISCOVERY"
            
            # Parameter extraction
            elif "Extract search parameters" in str(inputs):
                if "under $100" in str(inputs):
                    response.content = json.dumps({
                        "product_type": "running shoes",
                        "price_range": {"max": 100}
                    })
                elif "Nike Air" in str(inputs):
                    response.content = json.dumps({
                        "specific_product": "Nike Air Max",
                        "brand": ["Nike"]
                    })
                else:
                    response.content = json.dumps({
                        "product_type": "shoes"
                    })
            
            # Query enhancement
            elif "Create an enhanced search query" in str(inputs):
                if "running shoes" in str(inputs):
                    response.content = "running shoes athletic footwear jogging sneakers under 100 dollars"
                elif "Nike Air" in str(inputs):
                    response.content = "Nike Air Max sneakers athletic shoes"
                else:
                    response.content = "shoes footwear for casual and athletic use"
            
            # Results ranking
            elif "Re-rank these product results" in str(inputs):
                response.content = json.dumps([
                    {"product_id": 1, "rank": 1, "reason": "Best match"},
                    {"product_id": 2, "rank": 2, "reason": "Good match"}
                ])
            
            # Response generation
            elif "Create a helpful response for an e-commerce search" in str(inputs):
                if "NO_RESULTS_FOUND" in str(inputs) or "No results found" in str(inputs):
                    response.content = "I couldn't find any products matching your search. Try these alternatives..."
                else:
                    response.content = "Here are some running shoes that match your search. The Nike Running Shoes are a great option priced at $99.99."
            
            # Quality issues
            elif "Create a response for an e-commerce search where we found results" in str(inputs):
                response.content = "I found some products that might interest you, but they don't exactly match your search criteria."
            
            # Cleaning response
            elif "Rewrite this e-commerce search response" in str(inputs):
                response.content = "Here are the products that match your search."
            
            else:
                # Default response for unhandled cases
                response.content = "Default response"
            
            return response
        
        # Set up mock to use our side effect function
        self.mock_llm.invoke.side_effect = mock_llm_invoke
    
    def configure_vector_store_mock(self):
        """Configure vector store mock."""
        # Mock search results
        test_products = [
            {"id": 1, "name": "Nike Running Shoes", "brand": "Nike", "price": 99.99, "category": "footwear"},
            {"id": 2, "name": "Adidas Running Shoes", "brand": "Adidas", "price": 89.99, "category": "footwear"}
        ]
        
        # Set up mock to return test products
        self.mock_vector_store.search.return_value = [
            (test_products[0], 0.95),
            (test_products[1], 0.85)
        ]
    
    @patch('main.build_search_graph')
    def test_standard_search_flow(self, mock_build_search_graph):
        """Test standard search flow with valid query."""
        # Mock search executor
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "query": "running shoes under $100",
            "intent": "PRICE_BASED",
            "parameters": {"product_type": "running shoes", "price_range": {"max": 100}},
            "enhanced_query": "running shoes athletic footwear jogging sneakers under 100 dollars",
            "retrieval_results": [
                {"id": 1, "name": "Nike Running Shoes", "brand": "Nike", "price": 99.99, "relevance_score": 0.95},
                {"id": 2, "name": "Adidas Running Shoes", "brand": "Adidas", "price": 89.99, "relevance_score": 0.85}
            ],
            "ranked_results": [
                {"id": 1, "name": "Nike Running Shoes", "brand": "Nike", "price": 99.99, "relevance_score": 0.95, "rank": 1},
                {"id": 2, "name": "Adidas Running Shoes", "brand": "Adidas", "price": 89.99, "relevance_score": 0.85, "rank": 2}
            ],
            "response": "Here are some running shoes that match your search. The Nike Running Shoes are a great option priced at $99.99.",
            "input_validation_error": None,
            "error": None,
            "conversation_history": [],
            "metadata": {"query_timestamp": 1000.0}
        }
        mock_build_search_graph.return_value = mock_executor
        
        # Call method under test
        result = execute_search("running shoes under $100")
        
        # Verify result structure
        self.assertEqual(result["intent"], "PRICE_BASED")
        self.assertEqual(len(result["ranked_results"]), 2)
        self.assertIsNotNone(result["response"])
        self.assertIsNone(result["error"])
        
        # Verify search executor was called with correct state
        args, _ = mock_executor.invoke.call_args
        state = args[0]
        self.assertEqual(state["query"], "running shoes under $100")
    
    @patch('main.build_search_graph')
    def test_empty_query_flow(self, mock_build_search_graph):
        """Test search flow with empty query."""
        # Mock search executor
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "query": "",
            "intent": "",
            "parameters": {},
            "enhanced_query": None,
            "retrieval_results": [],
            "ranked_results": [],
            "response": "I noticed your search was empty. What kind of products are you looking for?",
            "input_validation_error": "EMPTY_QUERY",
            "error": "Input validation failed: EMPTY_QUERY",
            "conversation_history": [],
            "metadata": {"query_timestamp": 1000.0}
        }
        mock_build_search_graph.return_value = mock_executor
        
        # Call method under test
        result = execute_search("")
        
        # Verify error handling
        self.assertEqual(result["input_validation_error"], "EMPTY_QUERY")
        self.assertIsNotNone(result["error"])
        self.assertIsNotNone(result["response"])
    
    @patch('main.build_search_graph')
    def test_no_results_flow(self, mock_build_search_graph):
        """Test search flow with no results found."""
        # Mock search executor
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "query": "nonexistent product",
            "intent": "SPECIFIC_PRODUCT",
            "parameters": {"specific_product": "nonexistent product"},
            "enhanced_query": "nonexistent product",
            "retrieval_results": [],
            "ranked_results": [],
            "response": "I couldn't find any products matching your search. Try these alternatives...",
            "input_validation_error": None,
            "error": "NO_RESULTS_FOUND",
            "conversation_history": [],
            "metadata": {"query_timestamp": 1000.0, "no_results_found": True}
        }
        mock_build_search_graph.return_value = mock_executor
        
        # Call method under test
        result = execute_search("nonexistent product")
        
        # Verify no results handling
        self.assertEqual(result["error"], "NO_RESULTS_FOUND")
        self.assertIsNotNone(result["response"])
        self.assertEqual(len(result["ranked_results"]), 0)
    
    @patch('main.build_search_graph')
    def test_conversation_search_flow(self, mock_build_search_graph):
        """Test conversation search flow with context."""
        # Mock search executor
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {
            "query": "do you have these in red",
            "intent": "ATTRIBUTE_SEARCH",
            "parameters": {"attributes": {"color": ["red"]}},
            "enhanced_query": "Nike Running Shoes in red color",
            "retrieval_results": [
                {"id": 3, "name": "Nike Running Shoes - Red", "brand": "Nike", "price": 99.99, "relevance_score": 0.95}
            ],
            "ranked_results": [
                {"id": 3, "name": "Nike Running Shoes - Red", "brand": "Nike", "price": 99.99, "relevance_score": 0.95, "rank": 1}
            ],
            "response": "Yes, we have Nike Running Shoes in red, priced at $99.99.",
            "input_validation_error": None,
            "error": None,
            "conversation_history": [
                {"query": "show me running shoes", "response": "Here are some running shoes...",
                 "product_references": [{"id": 1, "name": "Nike Running Shoes"}], "timestamp": 900.0}
            ],
            "metadata": {"query_timestamp": 1000.0, "conversation_aware": True}
        }
        mock_build_search_graph.return_value = mock_executor
        
        # Create conversation context
        conversation_context = {
            "history": [
                {"query": "show me running shoes", "response": "Here are some running shoes...",
                 "product_references": [{"id": 1, "name": "Nike Running Shoes"}], "timestamp": 900.0}
            ],
            "preferences": {
                "preferred_brands": ["Nike", "Adidas"]
            }
        }
        
        # Call method under test
        result = execute_conversation_search("do you have these in red", conversation_context)
        
        # Verify conversation-aware search
        self.assertEqual(result["intent"], "ATTRIBUTE_SEARCH")
        self.assertEqual(len(result["ranked_results"]), 1)
        self.assertIsNotNone(result["response"])
        self.assertTrue(result["metadata"]["conversation_aware"])
        self.assertIn("Nike Running Shoes - Red", result["ranked_results"][0]["name"])
    
    @patch('main.build_search_graph')
    def test_exception_handling(self, mock_build_search_graph):
        """Test exception handling in search execution."""
        # Mock search executor to raise exception
        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = Exception("Test exception")
        mock_build_search_graph.return_value = mock_executor
        
        # Call method under test
        result = execute_search("running shoes")
        
        # Verify error handling
        self.assertIsNotNone(result["error"])
        self.assertIsNotNone(result["response"])
        self.assertIn("trouble processing", result["response"])


if __name__ == '__main__':
    unittest.main()