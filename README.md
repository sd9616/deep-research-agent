# A Simple Deep Research Agent

A LangGraph-based research agent that generate comprehensive reports using web search results. This agent implements an iterative approach to research similar to what you would use while researching a new topic: clarify your thoughts, define your focus, create searches, read web pages, summarize your findings,  evaluate your findings and repeat till you are satisfied with your answer. Inspired by [open_deep_research](https://github.com/langchain-ai/open_deep_research) and [local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher). 

## Architecture

![LangGraph Flow](langgraph_diagram.png)

## Overview

This project implements an iterative deep research agent using LangGraph that: 
- **Clarifies** user research queries to ensure clear understanding
- **Defines** research focus and generates 3-5 research questions
- **Creates** targeted search queries from research questions
- **Executes** web searches using Tavily and collects sources
- **Summarizes** findings using parallel map-reduce pattern (5 concurrent agents per source)
- **Evaluates** progress and iterates (up to 3 cycles) until questions are answered
- **Generates** comprehensive, well-structured research reports

## Features

- Agent asks clarifying questions if the research scope is unclear
- Utilizes an agent to converts research questions into targeted search queries
- Uses Tavily for comprehensive web search
- Utilizes Uses map-reduce with 5 concurrent agents to analyze sources
- Automatically continues research (uptil a user defined maximum) until questions are answered
- Uses GPT-4 for reasoning and the smaller GPT-3.5 for summarization 
- Generates a synthesized report 

## Project Structure

```
deep-research-agent/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration and environment variable management
│   ├── deep_research_flow.py # LangGraph workflow (main graph)
│   ├── main.py              # CLI entry point
│   ├── prompts.py           # All prompt templates
│   ├── state.py             # LangGraph state definitions
│   └── utils.py             # Helper functions (search tools, formatting)
├── requirements.txt         # Python dependencies
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   # cd into folder 
   cd deep-research-agent
   cp .env.example .env
   nano .env
   # Edit .env and add your API keys: 
   # You can also search for "TODO:" to find the following lines
   # OPENAI_API_KEY=
   # TAVILY_API_KEY=
   # LANGCHAIN_API_KEY=
   ```

4. **Run the agent** 

   * **LangGraph Studio**
     ```bash 
     langgraph dev 
     ```
<!-- 
   * **CLI**
      Run the agent with a research query:

      ```bash
      python -m src.main "Investigate the relationship between weather and exercise."
      ```

      Enable verbose mode to see intermediate steps:

      ```bash
      python -m src.main "Research " --verbose
      ```

      Save the report to a file:
      ```bash
      python -m src.main "Climate change impact on ocean ecosystems" --output report.md -->
      ```


## Acknowledgments

- Inspired by [open_deep_research](https://github.com/langchain-ai/open_deep_research) and [local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher)
- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
