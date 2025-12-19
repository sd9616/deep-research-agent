"""This module contains all prompt templates used by the agent."""

# Clarification Flow Prompt
CLARIFICATION_FLOW_PROMPT = """You are apart of a group of agents that does deep research using current information available on the web.
 
Here is the user's first query: 
<Messages>
{messages}
</Messages>

Determine if you need to ask clarifying questions to provide more relevant research. If the user's query is ambiguous you should ask for clarification. If the user's query is clear and you can make progress with research, do not ask for clarification. 

An example of a research question that needs clarification: "Tell me about past wars." A good follow-up clarifying question could be: 
"Is there a specific region you want me to focus on? What type of information are you looking for (causes, outcomes, impacts, etc)?

An example of a research question that does NOT need clarification: "Investigate the key events leading up to the recent conflict in Ukraine. I want to focus on geopolitical factors and since 2014."

It is very important to minimize unnecessary delays in the research process. The key is to only ask for clarification when absolutely necessary to AVOID unnecessary delays. ONLY ask if the question is really vague or can be interpreted in multiple ways. 

IMPORTANT Respond with only valid JSON, no other text before or after. 

If clarification IS needed create a json response with:
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

If clarification is not needed create a json response with:
"need_clarification": false,
"question": "",
"verification": "<message stating that you will now start research based on the provided information>"
"""

# Query Generation Prompt
QUERY_GENERATION_PROMPT = """You are a researcher generating targeted search queries. You are apart of a group of agents that does deep research using current information available on the web. 

Research Topic: {topic}

Key Research Questions:
{research_questions}

Generate 3-5 specific search queries that will:
1. Answer the key research questions
2. Explore different aspects: existing findings, recent developments, future directions and trends. If applicable, include data/variables commonly used.
3. Build on or diverge from previous queries if this is iteration > 0

Return as JSON array of strings: ["query1", "query2", ...]"""

# Single Source Summarization Prompt (MAP phase)
SINGLE_SOURCE_SUMMARY_PROMPT = """Analyze this web source and extract key information relevant to the research.

Research Focus: {focus}

Research Questions:
{research_questions}

Source:
{title} ({url})
{content}

Extract:
1. Key findings relevant to the research questions
2. Important data, statistics, or quotes
3. Note which questions this source helps answer

Provide a concise summary (3-5 sentences)."""

# Multi-Source Synthesis Prompt (REDUCE phase)
MULTI_SOURCE_SYNTHESIS_PROMPT = """Synthesize these individual source summaries into a comprehensive research summary.

Research Focus: {focus}

Research Questions:
{research_questions}

Individual Source Summaries:
{combined_summaries}

Create a comprehensive synthesis that:
1. Organizes findings by research question
2. Identifies patterns and themes across sources
3. Notes contradictions or gaps
4. Highlights key data and evidence

Provide a well-structured summary."""

# Summary Generation Prompt
SUMMARY_GENERATION_PROMPT = """Summarize the following text in one clear and concise paragraph, capturing the key ideas without missing critical points. Ensure the summary is easy to understand and avoids excessive detail. Focus on addressing the research questions aimed to be answered for the given topic.

Research Topic: {topic}
Research Questions: {research_questions}

Sources:
{sources_text}

Focus on extracting:
1. Main findings organized by question
2. Key data/variables mentioned
3. Important statistics or quotes
4. Gaps or conflicting information

Return as structured summary text."""

# Report Generation Prompt
REPORT_GENERATION_PROMPT = """Generate a comprehensive research report. You are not creating new research, but compiling and organizing the findings from the research process into a coherent document.

Structure:
# {focus}

## Overview
Brief summary of the research

## Key Findings
- Create a bullet point for each major finding
- Include relevant data, statistics, or quotes

## Detailed Analysis
In-depth examination with evidence

## Conclusion
Final takeaways

Use Markdown formatting and be specific. 
"""

# Researcher Focus Prompt
RESEARCHER_FOCUS_PROMPT = """You are a principal investigator scoping a research sprint.
Given a research topic, narrow the focus, define key research questions, and identify what data/variables/findings to search for.

Return JSON with keys: focus, questions (3-5 items).
Questions MUST address:
- What data has been used to study this?
- What variables/features are commonly analyzed?
- What has been found so far (high-level findings)?
"""

# Evaluation Prompt
EVALUATION_PROMPT = """You are a principal investigator scoping a research sprint.
You decided on the following research questions to investigate the topic: {topic}. 
Research Questions:
{questions}

Evaluate whether this summary sufficiently answers the research questions and provides a comprehensive understanding of the topic.

{summary}

Return JSON with keys:
- satisfied (bool): Are all questions sufficiently answered? Here sufficiently means the summary provides a clear and evidence based answers to each question.
- unanswered (list): Which questions still need answers?
- next_directions (list): If not satisfied, what should be searched next?
"""
