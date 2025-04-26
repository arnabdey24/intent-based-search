"""
LLM setup and utility functions.
"""
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Dict, Any
import logging

from config import LLM_CONFIG, EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize and return the LLM instance.
    
    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            api_key=LLM_CONFIG["api_key"]
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise

def get_embeddings():
    """
    Initialize and return the embeddings model using Hugging Face's all-MiniLM-L6-v2.

    Returns:
        HuggingFaceEmbeddings: Configured embeddings instance
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {str(e)}")
        raise

def create_llm_chain(prompt_template: str, input_vars: list):
    """
    Create a chain combining a prompt template with the LLM.
    
    Args:
        prompt_template: The template string for the prompt
        input_vars: List of input variable names used in the template
        
    Returns:
        A chain that can be invoked with the input variables
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | get_llm()
    return chain

def safe_llm_call(chain, inputs: Dict[str, Any], default_response: str = "") -> str:
    """
    Safely call an LLM chain with error handling.
    
    Args:
        chain: The LLM chain to call
        inputs: Dictionary of input values
        default_response: Fallback response if the call fails
        
    Returns:
        The LLM response or the default response on failure
    """
    try:
        response = chain.invoke(inputs)
        return response.content
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        return default_response
