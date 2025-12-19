"""This module defines the state schema for the LangGraph research agent."""

from typing import Annotated, List, Optional
try:
    from typing import NotRequired  
except ImportError:
    from typing_extensions import NotRequired
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

class ResearchPlan(BaseModel):
    """Structured research plan defining what to investigate.
    
    Keeps track of:
    - Original query
    - Generated search queries
    """
    original_query: str = Field(description="The user's original research question")
    search_queries: List[str] = Field(default_factory=list, description="Generated search queries to execute")

class RawSource(BaseModel):
    """Raw web content from a single source (unprocessed).
    
    This is the unfiltered data directly from search results.
    """
    url: str = Field(description="Source URL")
    title: str = Field(description="Page title")
    content: str = Field(description="Raw text content from the page")
    snippet: Optional[str] = Field(default=None, description="Short excerpt/snippet")
    retrieved_at: Optional[str] = Field(default=None, description="Timestamp of retrieval")

class ClarifyWithUser(BaseModel):
    """
    Structured response from query clarification.
    
    Used to determine whether to ask user for clarification or proceed with research.
    Enables inline routing via Command API without intermediate pause nodes.
    """
    need_clarification: bool = Field(description="Whether clarification is needed from user")
    question: str = Field(default="", description="Clarification question(s) to ask if needed")
    verification: str = Field(default="", description="Confirmation of understanding if no clarification needed")

class ResearcherState(MessagesState):
    """ResearcherState - tracks progress across research iterations.
    Controls the research loop by tracking focus, questions, satisfaction, and iteration count.
    """
    focus: str = Field(description="Narrowed research focus")
    research_questions: List[str] = Field(default_factory=list, description="Key questions to answer")
    iteration: int = Field(default=0, description="Current loop iteration")
    satisfied: bool = Field(default=False, description="Whether questions are sufficiently answered")

class UserInputQueryState(MessagesState):
    """Input schema for the research agent graph.
    
    This defines what the user provides to start a research workflow.
    """
    pass

class UserOutputReportState(MessagesState):
    """Output schema for the research agent graph.
    
    This defines what the completed research workflow returns.
    Only includes messages - the final report is included as the last AIMessage.
    """
    pass  # Only return messages (inherited from MessagesState)

class ResearchState(MessagesState):
    """Main state schema for the deep research agent graph.
    
    Inherits from MessagesState to get automatic message handling with add_messages reducer.
    
    Fields:
    - messages: Message history (from MessagesState, with add_messages reducer)
    - researcher: Current research state (Researcher with focus, questions, iteration, satisfaction)
    - research_plan: Research planning (queries to execute)
    - search_results: Current iteration's search results (raw sources)
    - current_summary: Current iteration's findings summary
    - final_report: Final research report output
    """
    
    researcher: NotRequired[ResearcherState]  # ResearcherState (focus, questions, loop control)
    search_results: NotRequired[List[RawSource]]  # Current iteration search results
    current_summary: NotRequired[str]  # Current iteration summary
    research_plan: NotRequired[Optional[ResearchPlan]]  # Research plan (queries)
    final_report: NotRequired[str]  # Final output
