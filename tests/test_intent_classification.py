"""
Tests for intent classification component.
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
from pipeline.intent_classification import classify_intent
from utils.prompts import INTENT_CLASSIFICATION_PROMPT

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestIntentClassification(unittest.TestCase):
    """Tests for intent classification component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock LLM and prompt template
        self.llm_patcher = patch('pipeline.intent_classification.get_llm')
        self.mock_get_llm = self.llm_patcher.start()
        self.mock_llm = MagicMock()
        self.mock_get_llm.return_value = self.mock_llm
        
        # Mock safe_llm_call
        self.safe_llm_call_patcher = patch('pipeline.intent_classification.safe_llm_call')
        self.mock_safe_llm_call = self.safe_llm_call_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.llm_patcher.stop()
        self.safe_llm_call_patcher.stop()
    
    def test_classify_product_discovery_intent(self):
        """Test classification of product discovery intent."""
        # Configure mock
        self.mock_llm.invoke.return_value.content = "PRODUCT_DISCOVERY"
        
        # Create test state
        state = SearchState(
            query="Show me some running shoes",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify intent classification
        self.assertEqual(result["intent"], "PRODUCT_DISCOVERY")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_specific_product_intent(self):
        """Test classification of specific product intent."""
        # Configure mock
        self.mock_llm.invoke.return_value.content = "SPECIFIC_PRODUCT"
        
        # Create test state
        state = SearchState(
            query="Do you have Nike Air Max 90?",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify intent classification
        self.assertEqual(result["intent"], "SPECIFIC_PRODUCT")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_price_based_intent(self):
        """Test classification of price-based intent."""
        # Configure mock
        self.mock_llm.invoke.return_value.content = "PRICE_BASED"
        
        # Create test state
        state = SearchState(
            query="Show me running shoes under $100",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify intent classification
        self.assertEqual(result["intent"], "PRICE_BASED")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_attribute_search_intent(self):
        """Test classification of attribute search intent."""
        # Configure mock
        self.mock_llm.invoke.return_value.content = "ATTRIBUTE_SEARCH"
        
        # Create test state
        state = SearchState(
            query="I need red running shoes in size 10",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify intent classification
        self.assertEqual(result["intent"], "ATTRIBUTE_SEARCH")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_problem_solution_intent(self):
        """Test classification of problem-solution intent."""
        # Configure mock
        self.mock_llm.invoke.return_value.content = "PROBLEM_SOLUTION"
        
        # Create test state
        state = SearchState(
            query="I need shoes that help with plantar fasciitis",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify intent classification
        self.assertEqual(result["intent"], "PROBLEM_SOLUTION")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_comparison_intent(self):
        """Test classification of comparison intent."""
        # Configure mock
        self.mock_llm.invoke.return_value.content = "COMPARISON"
        
        # Create test state
        state = SearchState(
            query="What's better, Nike or Adidas running shoes?",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify intent classification
        self.assertEqual(result["intent"], "COMPARISON")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_availability_intent(self):
        """Test classification of availability intent."""
        # Configure mock
        self.mock_llm.invoke.return_value.content = "AVAILABILITY"
        
        # Create test state
        state = SearchState(
            query="Do you have AirPods Pro in stock?",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify intent classification
        self.assertEqual(result["intent"], "AVAILABILITY")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_invalid_intent(self):
        """Test classification with invalid intent returned from LLM."""
        # Configure mock to return invalid intent
        self.mock_llm.invoke.return_value.content = "NOT_A_VALID_INTENT"
        
        # Create test state
        state = SearchState(
            query="Show me some products",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify fallback to default intent
        self.assertEqual(result["intent"], "PRODUCT_DISCOVERY")
        self.assertIn("intent_classification_confidence", result["metadata"])
    
    def test_classify_with_validation_error(self):
        """Test intent classification with validation error."""
        # Create test state with validation error
        state = SearchState(
            query="invalid query",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error="SOME_ERROR",
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify routing to error handler
        self.assertEqual(result, "handle_validation_error")
    
    def test_safe_llm_call_usage(self):
        """Test that safe_llm_call is used for robustness."""
        # Configure mock
        self.mock_safe_llm_call.return_value = "PRODUCT_DISCOVERY"
        
        # Create test state
        state = SearchState(
            query="Show me some products",
            intent="",
            parameters={},
            enhanced_query=None,
            retrieval_results=[],
            ranked_results=[],
            response=None,
            input_validation_error=None,
            error=None,
            conversation_history=[],
            metadata={}
        )
        
        # Call method under test
        result = classify_intent(state)
        
        # Verify safe_llm_call was used
        self.mock_safe_llm_call.assert_called_once()
        self.assertEqual(result["intent"], "PRODUCT_DISCOVERY")


if __name__ == '__main__':
    unittest.main()