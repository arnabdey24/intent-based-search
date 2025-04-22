"""
Monitoring and metrics for the search system.
"""
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SearchSystemMonitor:
    """Monitor and evaluate search system performance."""
    
    def __init__(self):
        """Initialize the monitoring system."""
        logger.info("Initializing search system monitor")
        self.queries_processed = 0
        self.error_count = 0
        self.intent_distribution = {}
        self.avg_response_time = 0
        
        # Performance metrics
        self.performance_by_intent = {}
        self.hourly_query_count = {}
        
        # Initialize performance metrics
        for intent in [
            "PRODUCT_DISCOVERY",
            "SPECIFIC_PRODUCT",
            "ATTRIBUTE_SEARCH",
            "PROBLEM_SOLUTION",
            "COMPARISON",
            "PRICE_BASED",
            "AVAILABILITY"
        ]:
            self.performance_by_intent[intent] = {
                "count": 0,
                "avg_time": 0,
                "error_rate": 0
            }
    
    def log_search(self, query: str, result: Dict[str, Any], execution_time: float):
        """
        Log and analyze a search execution.
        
        Args:
            query: The search query
            result: The search result
            execution_time: Time taken to execute the search in seconds
        """
        self.queries_processed += 1
        
        # Track errors
        has_error = bool(result.get("error"))
        if has_error:
            self.error_count += 1
        
        # Track intent distribution
        intent = result.get("intent", "UNKNOWN")
        self.intent_distribution[intent] = self.intent_distribution.get(intent, 0) + 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.queries_processed - 1) + execution_time) / 
            self.queries_processed
        )
        
        # Track performance by intent
        if intent in self.performance_by_intent:
            intent_perf = self.performance_by_intent[intent]
            intent_perf["count"] += 1
            
            # Update average time
            intent_perf["avg_time"] = (
                (intent_perf["avg_time"] * (intent_perf["count"] - 1) + execution_time) /
                intent_perf["count"]
            )
            
            # Update error rate
            if has_error:
                intent_perf["error_rate"] = (
                    (intent_perf["error_rate"] * (intent_perf["count"] - 1) + 1) /
                    intent_perf["count"]
                )
        
        # Track hourly distribution
        current_hour = time.strftime("%Y-%m-%d-%H")
        self.hourly_query_count[current_hour] = self.hourly_query_count.get(current_hour, 0) + 1
        
        logger.debug(f"Logged search metrics for query: '{query}', intent: {intent}, time: {execution_time:.2f}s")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health metrics.
        
        Returns:
            Dictionary of health metrics
        """
        return {
            "queries_processed": self.queries_processed,
            "error_rate": self.error_count / max(1, self.queries_processed),
            "intent_distribution": self.intent_distribution,
            "avg_response_time": self.avg_response_time,
            "performance_by_intent": self.performance_by_intent
        }
    
    def evaluate_quality(self, query: str, expected_results: List[Dict[str, Any]], actual_results: List[Dict[str, Any]]) -> float:
        """
        Evaluate search quality for a given query with known expected results.
        
        Args:
            query: The search query
            expected_results: Expected/ideal results
            actual_results: Actual results returned
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not expected_results or not actual_results:
            return 0.0
            
        # Calculate precision@k
        k = min(3, len(actual_results))
        expected_ids = set(item.get("id") for item in expected_results)
        actual_top_k = [item.get("id") for item in actual_results[:k]]
        
        hits = sum(1 for item_id in actual_top_k if item_id in expected_ids)
        precision_at_k = hits / k
        
        return precision_at_k
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate percentiles for response time if we have enough data
        response_times = []
        
        # This is a mock implementation - in a real system we'd store all response times
        # or use a proper time series database to calculate these metrics
        
        report = {
            "summary": {
                "total_queries": self.queries_processed,
                "error_rate": self.error_count / max(1, self.queries_processed),
                "avg_response_time": self.avg_response_time
            },
            "intent_breakdown": {
                intent: {
                    "query_count": count,
                    "percentage": (count / max(1, self.queries_processed)) * 100
                }
                for intent, count in self.intent_distribution.items()
            },
            "performance": {
                "by_intent": self.performance_by_intent,
                "hourly_distribution": self.hourly_query_count
            }
        }
        
        return report