"""
Tests for the LangGraph pipeline components.
"""
import unittest
import sys
import os
from unittest.mock import MagicMock, patch
import logging

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.state import SearchState
from models.parameters import VALID_INTENTS
from pipeline.input_validation import validate_input, handle_validation_error
from pipeline.intent_classification import classify_intent
from pipeline.parameter_extraction import extract_parameters
from pipeline.query_enhancement import enhance_query
from pipeline.vector_search import retrieve_results
from pipeline.results_ranking import rank_results
from pipeline.quality_validation import validate_results, handle_no_results
from pipeline.response_generation import build_response, clean_response

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestInputValidation(unittest.TestCase):
    """Tests for input validation component."""
    
    def test_validate_valid_query(self):
        """Test validation of a valid query."""
        state = {
            "query": "I need running shoes",
            "metadata": {}
        }
        
        result = validate_input(state)
        
        self.assertIsNone(result.get("input_validation_error"))
        self.assertIn("query_length", result["metadata"])
    
    def test_validate_empty_query(self):
        """Test validation of an empty query."""
        state = {
            "query": "",
            "metadata": {}
        }
        
        result = validate_input(state)
        
        self.assertEqual(result["input_validation_error"], "EMPTY_QUERY")
    
    def test_validate_harmful_query(self):
        """Test validation of a potentially harmful query."""
        state = {
            "query": "How to hack a website",
            "metadata": {}
        }
        
        result = validate_input(state)
        
        self.assertEqual(result["input_validation_error"], "POTENTIALLY_HARMFUL_CONTENT")
    
    def test_handle_validation_error(self):
        """Test handling of validation errors."""
        state = {
            "query": "",
            "input_validation_error": "EMPTY_QUERY",
            "metadata": {}
        }
        
        result = handle_validation_error(state)
        
        self.assertIsNotNone(result["response"])
        self.assertIsNotNone(result["error"])


class TestIntentClassification(unittest.TestCase):
    """Tests for intent classification component."""
    
    @patch('utils.llm.get_llm')
    def test_classify_intent_valid(self, mock_get_llm):
        """Test classification of a valid intent."""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "PRODUCT_DISCOVERY"
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "Show me some running shoes",
            "metadata": {}
        }
        
        result = classify_intent(state)
        
        self.assertEqual(result["intent"], "PRODUCT_DISCOVERY")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    @patch('utils.llm.get_llm')
    def test_classify_intent_invalid(self, mock_get_llm):
        """Test classification with invalid intent returned."""
        # Mock LLM response with invalid intent
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "NOT_A_VALID_INTENT"
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "Show me some running shoes",
            "metadata": {}
        }
        
        result = classify_intent(state)
        
        # Should default to PRODUCT_DISCOVERY
        self.assertEqual(result["intent"], "PRODUCT_DISCOVERY")
    
    def test_classify_intent_with_validation_error(self):
        """Test intent classification with validation error."""
        state = {
            "query": "invalid query",
            "input_validation_error": "SOME_ERROR",
            "metadata": {}
        }
        
        result = classify_intent(state)
        
        # Should route to error handler
        self.assertEqual(result, "handle_validation_error")


class TestParameterExtraction(unittest.TestCase):
    """Tests for parameter extraction component."""
    
    @patch('utils.llm.get_llm')
    def test_extract_parameters_valid(self, mock_get_llm):
        """Test extraction of valid parameters."""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = '''
        {
            "product_type": "running shoes",
            "price_range": {"max": 100},
            "attributes": {"color": ["red", "blue"]}
        }
        '''
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "I need running shoes under $100 in red or blue",
            "intent": "PRICE_BASED",
            "metadata": {}
        }
        
        result = extract_parameters(state)
        
        self.assertEqual(result["parameters"]["product_type"], "running shoes")
        self.assertEqual(result["parameters"]["price_range"]["max"], 100)
        self.assertIn("color", result["parameters"]["attributes"])
        self.assertIn("parameter_extraction_status", result["metadata"])
    
    @patch('utils.llm.get_llm')
    def test_extract_parameters_invalid_json(self, mock_get_llm):
        """Test handling of invalid JSON response."""
        # Mock LLM response with invalid JSON
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "Not valid JSON"
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "I need running shoes",
            "intent": "PRODUCT_DISCOVERY",
            "metadata": {}
        }
        
        result = extract_parameters(state)
        
        # Should have empty parameters
        self.assertEqual(result["parameters"], {})
        self.assertIn("parameter_extraction_status", result["metadata"])
        self.assertIn("JSON_PARSE_ERROR", result["metadata"]["parameter_extraction_status"])


class TestQueryEnhancement(unittest.TestCase):
    """Tests for query enhancement component."""
    
    @patch('utils.llm.get_llm')
    def test_enhance_query(self, mock_get_llm):
        """Test query enhancement."""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "running shoes athletic footwear jogging sneakers under 100 dollars"
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "I need running shoes under $100",
            "intent": "PRICE_BASED",
            "parameters": {"product_type": "running shoes", "price_range": {"max": 100}},
            "metadata": {}
        }
        
        result = enhance_query(state)
        
        self.assertIsNotNone(result["enhanced_query"])
        self.assertIn("query_expansion_ratio", result["metadata"])
    
    @patch('utils.llm.get_llm')
    def test_enhance_query_error(self, mock_get_llm):
        """Test query enhancement with error."""
        # Mock LLM error
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "I need running shoes",
            "intent": "PRODUCT_DISCOVERY",
            "parameters": {},
            "metadata": {}
        }
        
        result = enhance_query(state)
        
        # Should fall back to original query
        self.assertEqual(result["enhanced_query"], "I need running shoes")
        self.assertIn("query_enhancement_error", result["metadata"])


class TestVectorSearch(unittest.TestCase):
    """Tests for vector search component."""
    
    @patch('pipeline.vector_search.vector_store')
    def test_retrieve_results(self, mock_vector_store):
        """Test vector search with results."""
        # Mock vector store search results
        mock_vector_store.search.return_value = [
            ({"id": 1, "name": "Product 1"}, 0.95),
            ({"id": 2, "name": "Product 2"}, 0.85)
        ]
        
        state = {
            "query": "running shoes",
            "enhanced_query": "running shoes athletic footwear",
            "metadata": {}
        }
        
        result = retrieve_results(state)
        
        self.assertEqual(len(result["retrieval_results"]), 2)
        self.assertEqual(result["retrieval_results"][0]["id"], 1)
        self.assertIn("vector_search_result_count", result["metadata"])
    
    @patch('pipeline.vector_search.vector_store')
    def test_retrieve_results_empty(self, mock_vector_store):
        """Test vector search with no results."""
        # Mock empty search results
        mock_vector_store.search.return_value = []
        
        state = {
            "query": "nonexistent product",
            "enhanced_query": "nonexistent product",
            "metadata": {}
        }
        
        result = retrieve_results(state)
        
        self.assertEqual(len(result["retrieval_results"]), 0)
        self.assertTrue(result["metadata"]["no_results_found"])
    
    @patch('pipeline.vector_search.vector_store')
    def test_retrieve_results_error(self, mock_vector_store):
        """Test vector search with error."""
        # Mock search error
        mock_vector_store.search.side_effect = Exception("Search error")
        
        state = {
            "query": "test query",
            "enhanced_query": "test query",
            "metadata": {}
        }
        
        result = retrieve_results(state)
        
        self.assertEqual(len(result["retrieval_results"]), 0)
        self.assertIsNotNone(result["error"])
        self.assertIn("vector_search_error", result["metadata"])


class TestResultsRanking(unittest.TestCase):
    """Tests for results ranking component."""
    
    @patch('utils.llm.get_llm')
    def test_rank_results(self, mock_get_llm):
        """Test ranking of search results."""
        # Mock LLM response with ranking JSON
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = '''
        [
            {"product_id": 2, "rank": 1, "reason": "Better match"},
            {"product_id": 1, "rank": 2, "reason": "Good but not best"}
        ]
        '''
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "blue running shoes",
            "intent": "ATTRIBUTE_SEARCH",
            "parameters": {"attributes": {"color": ["blue"]}},
            "retrieval_results": [
                {"id": 1, "name": "Red Running Shoes", "relevance_score": 0.85},
                {"id": 2, "name": "Blue Running Shoes", "relevance_score": 0.80}
            ],
            "metadata": {}
        }
        
        result = rank_results(state)
        
        # Check if results are reordered
        self.assertEqual(result["ranked_results"][0]["id"], 2)
        self.assertEqual(result["ranked_results"][1]["id"], 1)
        self.assertIn("rank", result["ranked_results"][0])
        self.assertIn("rank_reason", result["ranked_results"][0])
    
    def test_rank_results_empty(self):
        """Test ranking with empty results."""
        state = {
            "query": "test query",
            "intent": "PRODUCT_DISCOVERY",
            "parameters": {},
            "retrieval_results": [],
            "metadata": {}
        }
        
        result = rank_results(state)
        
        self.assertEqual(len(result["ranked_results"]), 0)
        self.assertIsNotNone(result["error"])


class TestQualityValidation(unittest.TestCase):
    """Tests for quality validation component."""
    
    def test_validate_results_good_quality(self):
        """Test validation with good quality results."""
        state = {
            "query": "Nike running shoes",
            "intent": "SPECIFIC_PRODUCT",
            "parameters": {"specific_product": "Nike running shoes"},
            "ranked_results": [
                {"id": 1, "name": "Nike Running Shoes", "relevance_score": 0.95}
            ],
            "metadata": {}
        }
        
        result = validate_results(state)
        
        # Should proceed to build response
        self.assertEqual(result, "build_response")
    
    def test_validate_results_no_exact_match(self):
        """Test validation with no exact product match."""
        state = {
            "query": "Nike Air Zoom Pegasus",
            "intent": "SPECIFIC_PRODUCT",
            "parameters": {"specific_product": "Nike Air Zoom Pegasus"},
            "ranked_results": [
                {"id": 1, "name": "Nike Running Shoes", "relevance_score": 0.75}
            ],
            "metadata": {}
        }
        
        result = validate_results(state)
        
        # Should route to quality issues handler
        self.assertEqual(result, "handle_quality_issues")
    
    def test_validate_results_empty(self):
        """Test validation with no results."""
        state = {
            "query": "test query",
            "intent": "PRODUCT_DISCOVERY",
            "parameters": {},
            "ranked_results": [],
            "metadata": {}
        }
        
        result = validate_results(state)
        
        # Should route to no results handler
        self.assertEqual(result, "handle_no_results")
    
    @patch('utils.llm.get_llm')
    def test_handle_no_results(self, mock_get_llm):
        """Test no results handler."""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "I couldn't find any products. Try these alternatives..."
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "test query",
            "intent": "PRODUCT_DISCOVERY",
            "parameters": {},
            "ranked_results": [],
            "metadata": {}
        }
        
        result = handle_no_results(state)
        
        self.assertIsNotNone(result["response"])
        self.assertEqual(result["error"], "NO_RESULTS_FOUND")


class TestResponseGeneration(unittest.TestCase):
    """Tests for response generation component."""
    
    @patch('utils.llm.get_llm')
    def test_build_response(self, mock_get_llm):
        """Test response generation."""
        # Mock LLM response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "Here are some running shoes for you..."
        mock_get_llm.return_value = mock_chain
        
        state = {
            "query": "running shoes",
            "intent": "PRODUCT_DISCOVERY",
            "parameters": {"product_type": "running shoes"},
            "ranked_results": [
                {"id": 1, "name": "Nike Running Shoes", "price": 99.99},
                {"id": 2, "name": "Adidas Running Shoes", "price": 89.99}
            ],
            "metadata": {}
        }
        
        result = build_response(state)
        
        self.assertIsNotNone(result["response"])
        self.assertIn("response_word_count", result["metadata"])
    
    def test_clean_response(self):
        """Test response cleaning."""
        response_with_prohibited = "I apologize, but I don't have specific information about these products."
        
        cleaned = clean_response(response_with_prohibited)
        
        # Cleaned response should not contain apologies
        self.assertNotIn("apologize", cleaned.lower())
        self.assertNotIn("sorry", cleaned.lower())


if __name__ == '__main__':
    unittest.main()