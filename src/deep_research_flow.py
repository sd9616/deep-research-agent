"""This module implements the main logic for the deep research agent. 

The workflow progresses through the following steps:

1. Clarification:           Validate and clarify user's research query
2. Researcher:              Define research focus and key questions
3. Query Generator:         Create targeted search queries
4. web_searcher:               Execute web searches and gather sources
5. Summarizer:              Extract and synthesize findings
6. Evaluator:               Assess if questions are answered; loop control
7. Report Generation:       Create final comprehensive report
"""

import logging
import json
import re
from typing import Dict, Any, List, Literal
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, get_buffer_string
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from src.state import (
    ResearchState,
    UserInputQueryState,
    UserOutputReportState,
    ResearchPlan,
    RawSource,
    ResearcherState,
    ClarifyWithUser,
)
from src.utils import get_search_tool, get_today_str, clean_json_response
from src.config import Configuration, Config
from src.prompts import (
    CLARIFICATION_FLOW_PROMPT,
    RESEARCHER_FOCUS_PROMPT,
    QUERY_GENERATION_PROMPT,
    EVALUATION_PROMPT,
    REPORT_GENERATION_PROMPT,
    SINGLE_SOURCE_SUMMARY_PROMPT,
    MULTI_SOURCE_SYNTHESIS_PROMPT,
)

logger = logging.getLogger(__name__)

# Initialize chat model that can be used throughout the graph
configurable_model = init_chat_model(
    model=Config.LLM_MODEL,
    model_provider=Config.LLM_PROVIDER,
    configurable_fields=("model", "max_tokens", "api_key"),
)

# Initialize smaller model for summarization tasks
summary_model = init_chat_model(
    model=Config.SUMMARY_MODEL,
    model_provider=Config.LLM_PROVIDER,
    configurable_fields=("model", "max_tokens", "api_key"),
)

# Initialize StateGraph with input/output schemas
graph = StateGraph(
    ResearchState,
    input=UserInputQueryState,
    output=UserOutputReportState,
    config_schema=Configuration
)

def initialize_state(state: ResearchState):
    """
    Initialize internal state from user input.
    
    Simply logs the incoming messages and passes state through to clarification.
    The clarification node will create a refined research brief from the messages.
    
    Args:
        state: Input state containing user's research query in messages
    """
    logger.info("Initializing")
    
    messages = state["messages"]
    logger.info(f"Received {len(messages)} messages")
    
    if messages:
        logger.debug(f"User message: {messages[0].content[:100]}")
    
    logger.debug("Passing to clarification node")
    return

# Add initialize node to graph
graph.add_node("initialize", initialize_state)
graph.add_edge(START, "initialize")

def clarification_node(state: ResearchState) -> Command[Literal["researcher", "__end__"]]:
    """
    Extract topic from user messages, analyze clarity, and create research topic.
    
    This function extracts the research topic from the user's message, determines
    whether clarification is needed, and creates the topic field in state for
    subsequent nodes to use.
    
    Args:
        state: Current research state containing user messages
        
    Returns:
        Command to either end with a clarifying question or proceed to researcher node
    """
    logger.info("Clarification node called")
    
    # Step 1: Get Messages
    messages = state["messages"]
    logger.info(f"Number of messages in state: {len(messages)}")
    for i, msg in enumerate(messages):
        logger.info(f"  Message {i}: {type(msg).__name__} - {str(msg.content)[:100]}")
    
    # Check if we've already asked for clarification (limit to 1 clarification round)
    ai_message_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
    if ai_message_count > 0:
        
        # If already asked for clarification, proceed to next node. 
        logger.info("Already asked for clarification once, proceeding to researcher")
        return Command(goto="researcher")
    
    # Step 2: Analyze whether clarification is needed
    prompt_content = CLARIFICATION_FLOW_PROMPT.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    
    
    response = configurable_model.invoke([HumanMessage(content=prompt_content)])
    
    # Parse JSON response
    raw = response.content if isinstance(response, AIMessage) else str(response)
    cleaned = clean_json_response(raw)
    
    data = json.loads(cleaned)
    clarify = ClarifyWithUser(**data)
        
    # Step 3: Route based on clarification result
    if clarify.need_clarification:
        logger.info("Asking user for clarification")
        
        # Prompt user with clarifying question
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=clarify.question)]}
        )
    else:
        # Proceed to research - researcher will extract brief from messages
        logger.info("Query is clear, proceeding to researcher")
        return Command(
            goto="researcher"
        )

# Add clarification node to graph
graph.add_node("clarifier", clarification_node)
graph.add_edge("initialize", "clarifier")

def researcher_node(state: ResearchState) -> Command[Literal["query_generator"]]:
    """Researcher node - defines research focus and key questions.
    
    Acts as researcher supervising the research work. Narrows the topic,
    identifies key research questions, and determines relevant data/variables/findings.
    Incorporates context from previous iterations to refine approach.
    
    Args:
        state: Current research state with messages and optional previous context
        
    Returns:
        Command to proceed to query_generator with updated researcher and plan
    """
    
    # Step 1: Extract research brief from messages, including clarification context
    logger.info("researcher_node called.")
    
    messages = state["messages"]
        
    # Build research brief from all messages using get_buffer_string
    research_brief = get_buffer_string(messages)
    logger.info("Built research brief from message history")
    
    if not research_brief:
        logger.error("No research brief in messages")
        raise ValueError("No research brief provided")
    
    logger.debug(f"Extracted research brief: {research_brief[:100]}...")
    
    existing_researcher = state.get("researcher")
    if existing_researcher:
        iteration = existing_researcher.get('iteration', 0) if isinstance(existing_researcher, dict) else existing_researcher.iteration
    else:
        iteration = 0
    
    # Step 2: Invoke LLM to generate research focus and questions
    system_prompt = SystemMessage(content=RESEARCHER_FOCUS_PROMPT)
    user_prompt = HumanMessage(content=f"Research Brief: {research_brief}")
    
    focus = ""
    questions: List[str] = []
    
    response = configurable_model.invoke([system_prompt, user_prompt])
    cleaned_content = clean_json_response(response.content)
    
    # Try parsing JSON
    try:
        data = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Cleaned content that failed to parse:\n{cleaned_content}")
        
        # Fallback: try to extract JSON from response using regex
        json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            logger.info("Successfully extracted JSON using regex fallback")
        else:
            # Ultimate fallback - use a basic structure to avoid crashing. 
            logger.warning("Could not parse JSON, using fallback structure")
            data = {
                "focus": research_brief[:200],
                "questions": [
                    "What are the main aspects to research?",
                    "What are the key findings?",
                    "What conclusions can be drawn?"
                ]
            }
    
    # Step 4: Extract focus and questions from response
    focus = data.get("focus", research_brief)
    questions_raw = data.get("questions", [])
    
    # Handle various question formats
    questions = []
    for q in questions_raw:
        if isinstance(q, dict):
            questions.append(q.get("question", q.get("text", str(q))))
        else:
            questions.append(str(q))
    
    logger.info(f"Generated focus and {len(questions)} research questions")

    # Step 5: Create researcher state object using ResearcherState class
    researcher = ResearcherState(
        messages=[],
        focus=focus,
        research_questions=questions,
        iteration=iteration,
        satisfied=False,
    )
    
    plan = ResearchPlan(
        original_query=research_brief,
        search_queries=[],
    )
    
    logger.info(f"Researcher initialized with focus: {focus[:80]}...")
    
    # Step 6: Return Command to proceed to query generator
    focus_summary = f"Research Focus: {focus}\n\nKey Questions:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    
    return Command(
        goto="query_generator",
        update={
            "researcher": researcher,
            "research_plan": plan,
            "messages": [AIMessage(content=focus_summary)],
        }
    )

# Add researcher node to graph
graph.add_node("researcher", researcher_node)

def query_generator_node(state: ResearchState) -> Command[Literal["web_searcher"]]:
    """Create search queries from research questions, ensuring that search queries are optimized for web searches.
    
    Translates research questions into actionable web search
    queries. 
    
    Args:
        state: Current state with researcher and research plan
        
    Returns:
        Command to proceed to web_searcher with updated research plan
    """
    logger.info("Query generator node.")
    
    researcher = state.get("researcher")
    plan = state.get("research_plan")
    
    logger.debug(f"Researcher exists: {researcher is not None}")
    logger.debug(f"Plan exists: {plan is not None}")
    
    if not researcher or not plan:
        logger.warning("Missing researcher or plan state")
        return {"research_plan": plan or ResearchPlan(original_query="", search_queries=[])}
    
    focus = researcher.get('focus', '')
    questions = researcher.get('research_questions', [])
    
    logger.info(f"Focus: {focus[:100] if focus else 'None'}...")
    logger.info(f"Research questions: {len(questions)} questions")
    
    system_prompt = SystemMessage(content=QUERY_GENERATION_PROMPT.format(
        topic=focus,
        research_questions="\n".join(questions)
    ))
    
    search_queries: List[str] = []
    
    response = configurable_model.invoke([system_prompt])
    logger.info(f"LLM response: {response.content[:200]}...")
    cleaned_content = clean_json_response(response.content)
    search_queries = json.loads(cleaned_content)
    logger.info(f"Parsed {len(search_queries)} search queries: {search_queries}")
    
    plan.search_queries = search_queries
    
    logger.info(f"Updated plan with {len(search_queries)} queries")
    
    # Add generated queries to message history
    queries_message = "Generated Search Queries:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(search_queries))
    
    return Command(
        goto="web_searcher",
        update={
            "research_plan": plan,
            "messages": [AIMessage(content=queries_message)],
        }
    )

# Add query_generator node to graph
graph.add_node("query_generator", query_generator_node)
graph.add_edge("researcher", "query_generator")

def web_searcher_node(state: ResearchState) -> Command[Literal["summarizer"]]:
    """Execute web searches and gather raw sources.
    
    The web_searcher executes all generated search queries and retrieves web content.
    Results are stored as RawSource objects (unprocessed raw content from web),
    maintaining explicit separation between raw data collection and processing.
    
    Args:
        state: Current state with research plan containing search queries
        
    Returns:
        Command to proceed to summarizer with search_results list of RawSource objects
    """
    try:
        logger.info("web_searcher node called.")
        
        search_tool = get_search_tool()
        plan = state.get("research_plan")
        
        logger.info(f"Research plan exists: {plan is not None}")
        if plan:
            logger.info(f"Plan type: {type(plan)}")
            logger.info(f"Search queries: {plan.search_queries}")
        
        if not plan:
            logger.warning("No research plan in state")
            return Command(goto="summarizer", update={"search_results": []})
        
        search_queries = plan.search_queries
        
        if not search_queries:
            logger.warning(f"No search queries in plan. Plan content: {plan}")
            return Command(goto="summarizer", update={"search_results": []})
        
        logger.info(f"Executing {len(search_queries)} search queries")
        sources: List[RawSource] = []
        
        for i, query in enumerate(search_queries):
            logger.info(f"Query {i+1}/{len(search_queries)}: {query}")
            
            try:
                # Execute search
                results = search_tool.invoke({"query": query})
                logger.info(f"Search returned: {type(results)}")
                
                # Handle both dict and list responses from search tool
                results_list = []
                
                if isinstance(results, dict):
                    # Tavily returns dict with 'results' key
                    results_list = results.get('results', [])
                    logger.info(f"Extracted {len(results_list)} results from dict")
                    
                elif isinstance(results, list):
                    results_list = results
                    logger.info(f"Got {len(results_list)} results as list")
                    
                else:
                    logger.warning(f"Unexpected search result type: {type(results)}")
                
                # Convert to RawSource objects
                for result in results_list:
                    source = RawSource(
                        url=result.get("url", ""),
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        snippet=result.get("snippet"),
                        retrieved_at=datetime.now(timezone.utc).isoformat()
                    )
                    sources.append(source)
                    logger.info(f"Added source: {source.title[:50]}...")
                
            except Exception as e:
                logger.error(f"Search error for '{query}': {e}")
        
        logger.info(f"Collected {len(sources)} raw sources")
        
        return Command(
            goto="summarizer",
            update={"search_results": sources}
        )
        
    except Exception as e:
        logger.error(f"web_searcher error: {e}", exc_info=True)
        return Command(goto="summarizer", update={"search_results": []})

# Add web_searcher node to graph
graph.add_node("web_searcher", web_searcher_node)
graph.add_edge("query_generator", "web_searcher")

def summarize_single_source(source: RawSource, focus: str, research_questions: List[str]) -> str:
    """Summarize a single web source using an independent agent.
    
    Args:
        source: Raw source to summarize
        focus: Research focus
        research_questions: List of research questions
        
    Returns:
        Summary text for this source
    """
    try:
        # Create prompt for single source using imported template
        source_prompt = SINGLE_SOURCE_SUMMARY_PROMPT.format(
            focus=focus,
            research_questions=chr(10).join(f"{i+1}. {q}" for i, q in enumerate(research_questions)),
            title=source.title,
            url=source.url,
            content=source.content[:2000]
        )
                
        response = summary_model.invoke([HumanMessage(content=source_prompt)])
        logger.info(f"Summarized source: {source.title[:50]}...")
        return f"{source.title}\n{response.content}\n"
        
    except Exception as e:
        logger.error(f"Error summarizing source {source.title}: {e}")
        return f"{source.title} (Error: could not summarize)\n"

def summarizer_node(state: ResearchState) -> Command[Literal["evaluator"]]:
    """Extract and synthesize key findings from collected sources using parallel agents.
    
    Map-Reduce Pattern:
    1. MAP: Spawn independent agents to summarize each source in parallel
    2. REDUCE: Combine all source summaries into final synthesis
    
    Args:
        state: Current state with search results (raw sources)
        
    Returns:
        Command to proceed to evaluator with current_summary containing processed findings
    """
    try:
        sources = state.get("search_results", [])
        researcher = state.get("researcher")
        
        focus = researcher.get('focus', '')
        research_questions = researcher.get('research_questions', [])
        
        logger.info(f"MAP Phase: Spawning {len(sources)} agents to summarize sources in parallel")
        
        # MAP Phase: Summarize each source in parallel using thread pool
        source_summaries = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            
            # Submit all tasks
            future_to_source = {
                executor.submit(summarize_single_source, source, focus, research_questions): source 
                for source in sources
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_source):
                try:
                    summary = future.result()
                    source_summaries.append(summary)
                except Exception as e:
                    source = future_to_source[future]
                    logger.error(f"Source summarization failed for {source.title}: {e}")
        
        logger.info(f"MAP Phase complete: {len(source_summaries)} sources summarized")
        
        # REDUCE Phase: Combine all source summaries into final synthesis
        logger.info("REDUCE Phase: Synthesizing all source summaries")
        
        combined_summaries = "\n\n".join(source_summaries)
        
        # Create synthesis prompt using imported template
        reduce_prompt = MULTI_SOURCE_SYNTHESIS_PROMPT.format(
            focus=focus,
            research_questions=chr(10).join(f"{i+1}. {q}" for i, q in enumerate(research_questions)),
            combined_summaries=combined_summaries
        )
        
        response = summary_model.invoke([HumanMessage(content=reduce_prompt)])
        final_summary = response.content
        
        logger.info(f"REDUCE Phase complete: Final synthesis generated from {len(sources)} sources")
        
        return Command(
            goto="evaluator",
            update={
                "current_summary": final_summary,
                "messages": [AIMessage(content=f"Iteration Summary (from {len(sources)} sources):\n{final_summary}")],
            }
        )
        
    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        return Command(goto="evaluator", update={"current_summary": ""})

# Add summarizer node to graph
graph.add_node("summarizer", summarizer_node)
graph.add_edge("web_searcher", "summarizer")

def evaluate_progress_node(state: ResearchState) -> Command[Literal["researcher", "report"]]:
    """Evaluate if research questions are answered and control loop iteration.
    
    Determines whether the research has sufficiently answered the key questions.
    If not satisfied, it identifies unanswered questions and proposes new search
    directions. This node controls the main loop: satisfied -> report, unsatisfied
    and iterations < MAX -> loop back to researcher.
    
    Args:
        state: Current state with researcher questions and summary findings
        
    Returns:
        Command to either continue research loop or proceed to report generation
    """
    researcher = state.get("researcher")
    current_summary = state.get("current_summary", "")
    
    if not researcher:
        logger.warning("No researcher state")
        return {"researcher": ResearcherState(
            messages=[],
            focus="", 
            research_questions=[], 
            iteration=0,
            satisfied=False,
        )}
    
    focus = researcher.get('focus', '')
    research_questions = researcher.get('research_questions', [])
    current_iteration = researcher.get('iteration', 0)
    
    system_prompt = SystemMessage(content=EVALUATION_PROMPT.format(
        topic=focus,
        questions="\n".join(research_questions),
        summary=current_summary[:1000]
    ))
    
    satisfied = False
    unanswered: List[str] = research_questions
    next_directions: List[str] = []
    
    try:
        response = configurable_model.invoke([system_prompt])
        cleaned_content = clean_json_response(response.content)
        
        try:
            data = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in evaluation: {e}")
            logger.error(f"Cleaned content:\n{cleaned_content}")
            
            # Try regex fallback
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                logger.info("Successfully extracted JSON using regex fallback")
            else:
                raise
        
        satisfied = bool(data.get("satisfied", False))
        unanswered = data.get("unanswered", unanswered)
        next_directions = data.get("next_directions", [])
        
    except Exception as e:
        
        logger.error(f"Evaluation parsing error: {e}", exc_info=True)
        logger.warning("Using fallback evaluation values")
    
    # Stop if max iterations reached or research is satisfied
    if current_iteration + 1 >= Config.MAX_ITERATIONS:
        satisfied = True
        logger.info(f"Max iterations ({Config.MAX_ITERATIONS}) reached. Stopping research loop.")
    elif satisfied:
        logger.info(f"Research questions satisfied after {current_iteration + 1} iteration(s).")
    
    # Update researcher state with new iteration
    updated_researcher = ResearcherState(
        messages=[],
        focus=focus,
        research_questions=research_questions,
        iteration=current_iteration + 1,
        satisfied=satisfied,
    )
    
    logger.info(f"Evaluation: satisfied={satisfied}, unanswered={len(unanswered)}, iteration={current_iteration + 1}")
    
    # Add evaluation result to message history
    eval_message = f"Evaluation (Iteration {current_iteration + 1}):\n"
    eval_message += f"Satisfied: {satisfied}\n"
    if unanswered:
        eval_message += f"Unanswered Questions: {len(unanswered)}\n"
    
    # Return Command with conditional routing based on satisfaction
    next_node = "report" if satisfied else "researcher"
    logger.info(f"Routing to: {next_node}")
    
    return Command(
        goto=next_node,
        update={
            "researcher": updated_researcher,
            "messages": [AIMessage(content=eval_message)],
        }
    )

# Add evaluator node to graph - Command handles conditional routing internally
graph.add_node("evaluator", evaluate_progress_node)

def report_generation_node(state: ResearchState) -> Command[Literal["__end__"]]:
    """Generate comprehensive final research report.
    
    Creates a well-structured final report from all accumulated research findings.
    This is only executed after the evaluator marks the research as satisfied
    (all key questions answered or MAX_ITERATIONS reached).
    
    Args:
        state: Complete research state with findings, summary, and questions
        
    Returns:
        Command to end workflow with final_report containing the comprehensive markdown report
    """
    try:
        researcher = state.get("researcher")
        summary = state.get("current_summary", "")
        
        if not researcher:
            return Command(
                goto=END,
                update={"final_report": "Unable to generate report - no researcher state."}
            )
        
        focus = researcher.get('focus', '')
        research_questions = researcher.get('research_questions', [])
        
        system_prompt = SystemMessage(content=REPORT_GENERATION_PROMPT.format(focus=focus))
        
        user_prompt = HumanMessage(content=f"""
        Research Questions:
        {' '.join(research_questions)}

        Summary of Findings:
        {summary}

        Generate the final report.
        """)
        
        response = configurable_model.invoke([system_prompt, user_prompt])
        report = response.content
        
        logger.info("Final report generated")
        
        return Command(
            goto=END,
            update={
                "final_report": report,
                "messages": [AIMessage(content=report)],
            }
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        return Command(
            goto=END,
            update={"final_report": f"Error generating report: {str(e)}"}
        )

# Add report node 
graph.add_node("report", report_generation_node)

# Compile graph
deep_research_flow = graph.compile()
logger.info("Deep research flow graph compiled successfully")
