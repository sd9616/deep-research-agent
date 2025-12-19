"""Command-line interface for the Deep Research Agent.

Usage:
    python -m src.main "What are the latest developments in quantum computing?"
    python -m src.main "Climate change impacts" --output report.md
    python -m src.main "AI ethics" --verbose
"""

import argparse
import sys
import logging
import traceback

from pathlib import Path
from langchain_core.messages import HumanMessage
from src.config import Config, Configuration
from src.deep_research_flow import deep_research_flow 
from src.state import UserInputQueryState
from src.utils import validate_configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the deep research agent CLI.
    
    Workflow:
    1. Parse command-line arguments
    2. Validate configuration
    3. Create the graph
    4. Prepare initial state with user query
    5. Invoke the graph workflow
    6. Extract and display the final report
    7. Optionally save report to file
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - Generate reports grounded in web search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python -m src.main "What is quantum computing?"
            python -m src.main "Climate change impacts" --output report.md
            python -m src.main "AI ethics" --verbose
        """
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Research question or topic (optional if using interactive mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save the report (optional, prints to stdout if not specified)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show intermediate steps and debug information"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Info mode enabled")
    
    # Get query from arguments or prompt user
    query = args.query
    if not query:
        # Interactive mode
        print("\n" + "="*80)
        print("DEEP RESEARCH AGENT")
        print("="*80 + "\n")
        query = input("Enter your research question: ").strip()
        if not query:
            print("Error: Research question is required. Exiting")
            sys.exit(1)
    
    logger.info(f"Starting research on: {query}")
    
    try:
        logger.info("Validating configuration...")
        validate_configuration()
        
        # Prepare input: messages with research query
        input_data: UserInputQueryState = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Configure runtime behavior
        config = {
            "configurable": {
                "max_search_results": Config.MAX_SEARCH_RESULTS,
                "include_sources": args.verbose,
                "include_facts": args.verbose,
                "include_notes": args.verbose
            }
        }
        
        # Execute research workflow. 
        logger.info("Executing research workflow...")
        print("\nResearching... This may take a few moments...\n")
        
        final_state = deep_research_flow.invoke(input_data, config=config)
        
        # Extract report from structured state (preferred) or fallback to messages
        report = final_state.get("final_report", "")
        if not report:
            messages = final_state.get("messages", [])
            assistant_messages = [msg for msg in messages if msg.type == "ai" and not hasattr(msg, 'tool_calls')]
            report = assistant_messages[-1].content if assistant_messages else "No report generated"
        
        #  Display results
        print("\n" + "="*80)
        print("RESEARCH REPORT")
        print("="*80 + "\n")
        print(report)
        print("\n" + "="*80 + "\n")
        
        # Save to file if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Research Report\n\nQuery: {query}\n\n")
                f.write(report)
            
            logger.info(f"Report saved to {output_path}")
            print(f"[OK] Report saved to {output_path}")
        
        # Print summary statistics from structured state
        num_sources = len(final_state.get("sources", []))
        num_facts = len(final_state.get("extracted_facts", []))
        num_notes = len(final_state.get("synthesized_notes", []))
        plan = final_state.get("research_plan")
        open_questions = len(plan.open_questions) if plan else 0
        
        print(f"\nResearch Summary:")
        print(f"    Raw sources collected: {num_sources}")
        print(f"    Facts extracted: {num_facts}")
        print(f"    Conclusions synthesized: {num_notes}")
        print(f"    Open questions remaining: {open_questions}")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: {e}")
        print("\nPlease ensure all required API keys are set in your .env file.")
        print("See .env.example for configuration details.")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUnexpected error occurred: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
