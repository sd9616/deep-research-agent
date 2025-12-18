"""
Deep Research Agent Configuration

This module handles configuration loading and validation.

- Load environment variables from .env file
- Check for presence of API keys 
- Configure LLM settings: model name, temperature
- Configure search API settings: max results, search depth
"""

import os
from dotenv import load_dotenv
from typing import Optional, Annotated
from dataclasses import dataclass, field, fields
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the research agent. 
    
    Loads settings from environment variables with defaults.
    """
    
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai") 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Search API Settings
    SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # Agent Settings
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    MIN_SOURCES = int(os.getenv("MIN_SOURCES", "5"))
    
@dataclass(kw_only=True)
class Configuration:
    """
    Runtime configuration schema for the research agent graph.

    Allows config to be passed dynamically at graph creation time.     
    
    Usage Eg:
        config = Configuration(
            llm_provider="openai",
            llm_model="gpt-4-turbo",
            max_search_results=10
        )
        result = graph.invoke(input, config={"configurable": config})
    """
    
    # LLM Configuration
    llm_provider: str = field(default_factory=lambda: Config.LLM_PROVIDER)
    llm_model: str = field(default_factory=lambda: Config.LLM_MODEL)
    llm_temperature: float = field(default_factory=lambda: Config.LLM_TEMPERATURE)
    
    # Search Configuration
    search_provider: str = field(default_factory=lambda: Config.SEARCH_PROVIDER)
    max_search_results: int = field(default_factory=lambda: Config.MAX_SEARCH_RESULTS)
    
    # Agent Behavior
    max_iterations: int = field(default_factory=lambda: Config.MAX_ITERATIONS)
    min_sources: int = field(default_factory=lambda: Config.MIN_SOURCES)
    
    # Output Options
    include_sources: bool = True
    include_facts: bool = False
    include_notes: bool = False
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """
        Extract Configuration from RunnableConfig.
        
        This allows nodes to access configuration in a type-safe way
        
        Args: 
            cls:    The Configuration class
            cofnig: Optional RunnableConfig containing `configurable` overrides.
            
        Returns: 
            A Configuration instance with merged values. 
            
        Example:
            def my_node(state: State, config: RunnableConfig):
                cfg = Configuration.from_runnable_config(config)
                llm = init_chat_model(cfg.llm_model, model_provider=cfg.llm_provider)
        """
        
        # Return early if already a Configuration instance
        if isinstance(config, Configuration):
            return config

        # Extract configurable overrides from RunnableConfig
        configurable = {}
        if config is not None:
            configurable = config.get("configurable") or {}
            if isinstance(configurable, Configuration):
                return configurable

        # Merge env vars and configurable
        names = [f.name for f in fields(cls)]
        values = {
            n: (configurable.get(n) if configurable.get(n) is not None else os.environ.get(n.upper()))
            for n in names
            if configurable.get(n) is not None or os.environ.get(n.upper()) is not None
        }
        return cls(**values)

    @classmethod
    def validate(cls) -> None:
        """Validates that required API keys are present.
        
        Checks:
        - At least one LLM API key is set
        - At least one search API key is set
        
        Args: 
            cls: The Configuration class
            
        Raises:
            ValueError: If required API key is missing 
        """
        if cls.LLM_PROVIDER == "anthropic":
            if not cls.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not configured. "
                    "Please set the ANTHROPIC_API_KEY environment variable in .env."
                )
        else:  
            if not cls.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not configured. "
                    "Please set the OPENAI_API_KEY environment variable in .env."
                )
        
        if cls.SEARCH_PROVIDER == "tavily":
            if not cls.TAVILY_API_KEY:
                raise ValueError(
                    "TAVILY_API_KEY is not configured. "
                    "Please set the TAVILY_API_KEY environment variable in .env. "
                    "Get a key from https://tavily.com"
                )
