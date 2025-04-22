"""
Prompt templates for the intent-based search system.
"""
from langchain_core.prompts import ChatPromptTemplate

# Intent Classification Prompt
INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in classifying e-commerce search queries by intent. 
    Categorize the query into ONE of these intents:
    - PRODUCT_DISCOVERY: General browsing or exploring product categories
    - SPECIFIC_PRODUCT: Looking for a specific product
    - ATTRIBUTE_SEARCH: Searching by specific product attributes or features
    - PROBLEM_SOLUTION: Describing a problem seeking products that solve it
    - COMPARISON: Comparing multiple products or types
    - PRICE_BASED: Search primarily focused on price considerations
    - AVAILABILITY: Checking if something is in stock or available
    
    Return ONLY the intent category name, nothing else."""),
    ("human", "{query}")
])

# Parameter Extraction Prompt
PARAMETER_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract search parameters from this e-commerce query.
    The query intent is: {intent}
    
    Based on this intent, extract a JSON object with these possible keys (only include if present):
    - product_type: The type or category of product
    - specific_product: Exact product name if searching for specific item
    - attributes: Dictionary of attributes like color, size, material, etc.
    - price_range: Dictionary with min and/or max if mentioned
    - brands: List of brands mentioned
    - problems: List of problems user wants to solve with product
    - comparison_items: List of items being compared
    
    Return ONLY valid JSON, no other text."""),
    ("human", "{query}")
])

# Query Enhancement Prompt
QUERY_ENHANCEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in enhancing e-commerce search queries.
    Original query: {query}
    Detected intent: {intent}
    Extracted parameters: {parameters}
    
    Create an enhanced search query that:
    1. Expands with relevant synonyms
    2. Adds implicit product attributes based on intent
    3. Clarifies ambiguous terms
    
    Return ONLY the enhanced query text, nothing else."""),
    ("human", "Please enhance this query for better search results.")
])

# Results Ranking Prompt
RESULTS_RANKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in ranking e-commerce search results based on user intent.
    User query: {query}
    Detected intent: {intent}
    Search parameters: {parameters}
    
    Re-rank these product results in order of relevance to the user's intent.
    For each product, explain briefly why it matches or doesn't match the intent.
    
    Return a JSON array with objects containing:
    - product_id: The product ID
    - rank: New position (1 being best match)
    - reason: Brief explanation of ranking decision
    
    Return ONLY valid JSON, no other text."""),
    ("human", "Results to rank: {results}")
])

# Response Generation Prompt
RESPONSE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Create a helpful response for an e-commerce search.
    User query: {query}
    Detected intent: {intent}
    Search parameters: {parameters}
    Top 3 results: {top_results}
    
    Craft a conversational but concise response that:
    1. Acknowledges their search intent
    2. Highlights the top results and why they match
    3. For SPECIFIC_PRODUCT intent, directly address if we found the exact product
    4. For PROBLEM_SOLUTION intent, explain how products solve their problem
    
    Keep focus on addressing their needs without being overly verbose.
    Ensure the response is factual and based only on the results provided."""),
    ("human", "Please create a helpful response for this search.")
])

# No Results Prompt
NO_RESULTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Create a helpful response for an e-commerce search that returned no results.
    User query: {query}
    Detected intent: {intent}
    Search parameters: {parameters}
    
    Your response should:
    1. Acknowledge that we couldn't find exactly what they're looking for
    2. Suggest 2-3 alternative search approaches
    3. If possible, recommend related product categories
    4. Be concise and helpful
    
    The tone should be helpful and solution-oriented."""),
    ("human", "Please help the user with no search results.")
])

# Quality Issues Prompt
QUALITY_ISSUES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Create a response for an e-commerce search where we found results, 
    but they may not perfectly match what the user was looking for.
    
    User query: {query}
    Detected intent: {intent}
    Quality issues: {quality_issues}
    Search parameters: {parameters}
    Top results: {top_results}
    
    Your response should:
    1. Be honest about limitations in what we found
    2. Present the best matches we did find
    3. Acknowledge the specific mismatch (price, exact product, etc.)
    4. Suggest refinements or alternatives
    5. Be helpful and conversational
    
    Do not apologize excessively - just be helpful and direct."""),
    ("human", "Please create a helpful response addressing the quality issues.")
])

# Response Cleaning Prompt
RESPONSE_CLEANING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Rewrite this e-commerce search response to remove any:
    - Apologies or "I'm sorry" statements
    - References to being an AI or limitations
    - Statements about not having access to information
    
    Keep all the product recommendations and helpful content.
    Maintain a confident, helpful tone focused on addressing the customer need."""),
    ("human", "{response}")
])