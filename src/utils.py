"""This module contains helper functions for LLM initialization, search tool setup,
and other utility functions used throughout the agent.
"""

import logging
from typing import Optional, Any
from langchain_tavily import TavilySearch
from src.config import Config
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_search_tool():
    """Initialize and return the configured search tool.
    
    Supports: 
    - Tavily (tavily-python): High-quality web search API
    
    Configuration via environment variables:
    - SEARCH_PROVIDER: "tavily" 
    - TAVILY_API_KEY: Tavily API key
    - MAX_SEARCH_RESULTS: Number of results per search (default: 5)
    
    Returns:
        Initialized search tool instance
        
    Raises:
        ValueError: If search API key is not configured
    """
    provider = Config.SEARCH_PROVIDER.lower()
    
    if provider == "tavily":
        if not Config.TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY not found in environment. "
                "Please get an API key from https://tavily.com"
            )
        logger.info(f"Initializing Tavily search with max {Config.MAX_SEARCH_RESULTS} results")
        return TavilySearch(
            max_results=Config.MAX_SEARCH_RESULTS,
            api_key=Config.TAVILY_API_KEY,
        )
    else:
        raise ValueError(
            f"Unsupported search provider: {provider}. "
            "Currently only 'tavily' is supported."
        )


def format_sources(sources: list) -> str:
    """Format a list of sources into a citation string.
    
    Args:
        sources: List of source dictionaries with title, url, etc.
        
    Returns:
        Formatted sources string in Markdown format
    """
    if not sources:
        return "## Sources\n\nNo sources available."
    
    sources_markdown = "## Sources\n\n"
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Unknown")
        url = source.get("url", "")
        
        if url:
            sources_markdown += f"{i}. [{title}]({url})\n"
        else:
            sources_markdown += f"{i}. {title}\n"
    
    return sources_markdown


def truncate_content(content: str, max_length: int = 2000) -> str:
    """Truncate content to a maximum length for LLM context.
    
    Args:
        content: Content to truncate
        max_length: Maximum length in characters (default: 2000)
        
    Returns:
        Truncated content with ellipsis if needed
    """
    if len(content) <= max_length:
        return content
    
    # Truncate and add ellipsis
    truncated = content[:max_length-3]
    return truncated + "..."


def validate_configuration() -> bool:
    """Validate that required configuration is present.
    
    Checks:
    - At least one LLM API key is present
    - At least one search API key is present
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check LLM configuration
    provider = Config.LLM_PROVIDER.lower()
    if provider == "anthropic":
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required but not set")
    else:
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required but not set")
    
    # Check search configuration
    search_provider = Config.SEARCH_PROVIDER.lower()
    if search_provider == "tavily":
        if not Config.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required but not set")
    
    logger.info("Configuration validated successfully")
    return True


def get_today_str():
    """Get today's date as a string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def clean_json_response(content: str) -> str:
    """Remove markdown code blocks and other artifacts from JSON responses.
    
    Handles:
    - Markdown code fences (```json or ```)
    - Leading/trailing whitespace
    - Text before/after JSON object
    """
    cleaned = content.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove opening fence (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    
    # Try to extract just the JSON part if there's extra text
    cleaned = cleaned.strip()
    
    # Find first { or [ and last } or ]
    start_brace = cleaned.find('{')
    start_bracket = cleaned.find('[')
    
    # Determine which comes first
    if start_brace == -1 and start_bracket == -1:
        return cleaned  # No JSON found, return as is
    elif start_brace == -1:
        start = start_bracket
        end_char = ']'
    elif start_bracket == -1:
        start = start_brace
        end_char = '}'
    else:
        start = min(start_brace, start_bracket)
        end_char = '}' if start == start_brace else ']'
    
    # Find the matching closing bracket/brace
    end = cleaned.rfind(end_char)
    
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end+1]
    
    return cleaned.strip()
