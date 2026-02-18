"""
MWR AI Chat - Streamlit Interface with Session Memory
Embedded chat for Power BI dashboard
"""

import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from neo4j_graphrag import MWRGraphRAG
from tavily import TavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import plotly.graph_objects as go

# Load environment variables
load_dotenv(override=True)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MWR AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS FOR CLEAN LOOK
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat styling */
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    /* Title styling */
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    
    /* Compact layout */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Memory indicator */
    .memory-badge {
        background-color: #e8f4e8;
        color: #2d7d2d;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Conversation history for LLM context (stores the rolling context)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Track pending button question
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "graph_rag" not in st.session_state:
    with st.spinner("🔄 Connecting to MWR database..."):
        try:
            st.session_state.graph_rag = MWRGraphRAG()
            st.session_state.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            st.session_state.llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=4096
            )
            st.session_state.connected = True
        except Exception as e:
            st.session_state.connected = False
            st.error(f"❌ Connection failed: {e}")

# ============================================================
# SESSION MEMORY CONFIG
# ============================================================
MAX_MEMORY_TURNS = 10  # Keep last 10 exchanges (20 messages) for context

def get_conversation_context() -> str:
    """Build conversation context string from recent history"""
    if not st.session_state.conversation_history:
        return ""
    
    # Take last MAX_MEMORY_TURNS exchanges
    recent = st.session_state.conversation_history[-(MAX_MEMORY_TURNS * 2):]
    
    context_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long responses to keep context manageable
        content = msg["content"][:500] if len(msg["content"]) > 500 else msg["content"]
        context_lines.append(f"{role}: {content}")
    
    return "\n".join(context_lines)

def resolve_follow_up(question: str) -> str:
    """
    Use conversation history to resolve follow-up questions.
    E.g., "drill into the top one" → understands what "the top one" refers to
    """
    conversation_context = get_conversation_context()
    
    if not conversation_context:
        return question  # No history, return as-is
    
    # Check if the question seems like a follow-up
    follow_up_indicators = [
        "it", "that", "those", "them", "this", "these",
        "the top one", "the first", "the same", "more detail",
        "drill", "expand", "compare", "also", "what about",
        "and", "now show", "now filter", "instead", "versus"
    ]
    
    is_follow_up = any(indicator in question.lower() for indicator in follow_up_indicators)
    
    if not is_follow_up:
        return question  # Standalone question, no resolution needed
    
    # Use LLM to resolve the follow-up into a standalone question
    resolve_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question resolver. Given the conversation history and a follow-up question, 
rewrite the follow-up as a STANDALONE question that includes all necessary context.

If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Examples:
- History: "User: What are the top 5 risk states?" → Follow-up: "Tell me more about the top one" → "Tell me more about New York's risk score and factors"
- History: "User: How many Critical ZIP codes?" → Follow-up: "Which states are they in?" → "Which states have the most Critical risk tier ZIP codes?"
- History: "User: What is California's risk?" → Follow-up: "Compare it to New York" → "Compare California's risk score to New York's risk score"
"""),
        ("human", """Conversation History:
{history}

Follow-up Question: {question}

Rewritten standalone question:""")
    ])
    
    chain = resolve_prompt | st.session_state.llm | StrOutputParser()
    
    try:
        resolved = chain.invoke({
            "history": conversation_context,
            "question": question
        }).strip()
        return resolved if resolved else question
    except:
        return question  # Fallback to original

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def classify_question(question: str) -> str:
    """Classify question type"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify this question about Minimum Wage Risk and workforce market intelligence:
- DATABASE: Questions about specific data, scores, states, ZIP codes, education levels, workforce population, cost of labor, cost of living, demographics, unemployment rates, market profiles
- WEB_SEARCH: Questions about current news, recent legislation, pending bills
- BOTH: Need both database and web info
- GENERAL: General concepts, no specific data needed

Respond with ONLY one word: DATABASE, WEB_SEARCH, BOTH, or GENERAL"""),
        ("human", "{question}")
    ])
    
    chain = prompt | st.session_state.llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().upper()
    
    if result not in ["DATABASE", "WEB_SEARCH", "BOTH", "GENERAL"]:
        result = "DATABASE"
    return result

def search_web(query: str) -> list:
    """Search web for minimum wage news"""
    try:
        response = st.session_state.tavily.search(
            query=f"minimum wage {query} 2025",
            max_results=3
        )
        return response.get("results", [])
    except:
        return []

# ============================================================
# PLOTLY CHART GENERATION — INTELLIGENT CHART SYSTEM
# ============================================================

# Sodexo-friendly color palette
CHART_COLORS = [
    '#1B4F5C',  # Dark teal (primary)
    '#E74C3C',  # Red
    '#2ECC71',  # Green
    '#F39C12',  # Orange
    '#9B59B6',  # Purple
    '#3498DB',  # Blue
    '#1ABC9C',  # Turquoise
    '#E67E22',  # Dark orange
]

# Risk tier color mapping (matches Power BI)
RISK_COLORS = {
    'Critical': '#CC0000',
    'High': '#E74C3C',
    'Elevated': '#F39C12',
    'Moderate': '#F1C40F',
    'Low': '#2ECC71'
}

PERIOD_LABELS = {
    '2024-05': 'May 2024', '2024-07': 'Jul 2024', '2024-10': 'Oct 2024',
    '2024-11': 'Nov 2024', '2025-01': 'Jan 2025', '2025-02': 'Feb 2025',
    '2025-04': 'Apr 2025', '2025-05': 'May 2025', '2025-07': 'Jul 2025',
    '2025-08': 'Aug 2025', '2025-10': 'Oct 2025', '2025-11': 'Nov 2025',
    '2026-01': 'Jan 2026'
}

def should_generate_chart(question: str) -> bool:
    """Detect if a question would benefit from any type of chart"""
    chart_keywords = [
        'trend', 'over time', 'history', 'historical', 'changed', 'change',
        'direction', 'show trend', 'trajectory', 'how has', 'evolve', 'movement',
        'compare', 'comparison', 'versus', ' vs ', ' vs.', 'bar chart', 'chart', 'graph', 'plot',
        'top', 'highest', 'lowest', 'rank', 'worst', 'best', 'most', 'least',
        'riskiest', 'safest', 'expensive', 'cheapest'
    ]
    return any(kw in question.lower() for kw in chart_keywords)

def classify_chart_type(question: str) -> str:
    """Use LLM to intelligently determine the best chart type for a question"""
    chart_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data visualization expert. Classify what chart type best answers this question.

Chart types:
- LINE_TREND: ONLY for ERI Cost of Labor or Cost of Living time-series trends. We only have historical data for cost of labor and cost of living.
- BAR_COMPARE: Side-by-side comparison of current values between 2-5 specific named locations
- HBAR_RANK: Ranked list of items (top/bottom states, counties, highest/lowest risk, most expensive, highest unemployment, most educated)
- NONE: Question doesn't benefit from a chart (single location info, education breakdowns, general info, ZIP counts, news)

CRITICAL RULES:
- LINE_TREND is ONLY for "cost of labor trend" or "cost of living trend" questions. We do NOT have time-series data for unemployment, risk, education, income, or population.
- "unemployment trend" → NONE (no historical unemployment data available — only a single snapshot)
- "risk trend" → NONE (no historical risk data)
- "education trend" → NONE (no historical education data)
- "income trend" → NONE
- "which states have highest unemployment" → HBAR_RANK
- "rank counties by education" → HBAR_RANK

Examples:
- "top 5 risk states" → HBAR_RANK
- "highest risk counties" → HBAR_RANK
- "most expensive states for labor" → HBAR_RANK
- "which states have highest cost of labor" → HBAR_RANK
- "which states have highest unemployment" → HBAR_RANK
- "rank states by cost of living" → HBAR_RANK
- "rank counties by education level" → HBAR_RANK
- "rank states by income" → HBAR_RANK
- "wealthiest states" → HBAR_RANK
- "top 10 most populated states" → HBAR_RANK
- "safest states for minimum wage risk" → HBAR_RANK
- "lowest cost of living counties" → HBAR_RANK
- "cost of labor trend in SF" → LINE_TREND
- "how has cost of living changed over time" → LINE_TREND
- "ERI trend in Maryland" → LINE_TREND
- "compare cost of labor NY vs MD" → BAR_COMPARE
- "compare cost of living in San Francisco vs Los Angeles vs San Diego" → BAR_COMPARE
- "what is California's risk?" → NONE
- "unemployment trend in Montgomery County" → NONE (no time-series for unemployment)
- "unemployment rate in Boulder" → NONE
- "education breakdown in SF" → NONE
- "how many critical ZIP codes" → NONE
- "what are the latest minimum wage news" → NONE

Return ONLY one word: LINE_TREND, BAR_COMPARE, HBAR_RANK, or NONE"""),
        ("human", "{question}")
    ])
    
    chain = chart_prompt | st.session_state.llm | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip().upper()
        if result not in ["LINE_TREND", "BAR_COMPARE", "HBAR_RANK", "NONE"]:
            return "NONE"
        return result
    except:
        return "NONE"

def detect_metric_type(question: str) -> str:
    """Detect whether the question is about labor, living, risk, or both costs"""
    q_lower = question.lower()
    has_labor = any(kw in q_lower for kw in ['labor', 'col ', 'cost of labor'])
    has_living = any(kw in q_lower for kw in ['living', 'cost of living', 'coliv'])
    has_risk = any(kw in q_lower for kw in ['risk', 'score', 'tier', 'critical', 'riskiest'])
    
    if has_risk:
        return "risk"
    elif has_labor and not has_living:
        return "labor"
    elif has_living and not has_labor:
        return "living"
    elif has_labor and has_living:
        return "both"
    else:
        return "risk"

# ============================================================
# DATA FETCHING FUNCTIONS
# ============================================================

def fetch_trend_data(question: str) -> list:
    """Fetch ERI time-series data from Neo4j for line charts"""
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate a Neo4j Cypher query to fetch ERI time-series data.
The ZipCode nodes have these properties:
- eri_periods: list of period labels like ['2024-05', '2024-07', ...]
- eri_labor_history: list of Cost of Labor values per period
- eri_living_history: list of Cost of Living values per period
- county: county name (e.g., 'San Francisco')
- state: state abbreviation (e.g., 'CA')

CRITICAL RULES:
1. Always use DISTINCT to avoid duplicate rows
2. Always include: AND z.eri_periods IS NOT NULL
3. For a SINGLE county query, use LIMIT 1
4. For state-level queries, return max 5 representative counties (use LIMIT 5)
5. Return fields AS: county, state, periods, labor, living

Examples:
- "trend in San Francisco" → 
  MATCH (z:ZipCode) WHERE z.county = 'San Francisco' AND z.state = 'CA' AND z.eri_periods IS NOT NULL 
  RETURN DISTINCT z.county AS county, z.state AS state, z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living LIMIT 1

- "compare SF and LA" → 
  MATCH (z:ZipCode) WHERE z.county IN ['San Francisco', 'Los Angeles'] AND z.state = 'CA' AND z.eri_periods IS NOT NULL 
  WITH DISTINCT z.county AS county, z.state AS state, z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living 
  RETURN county, state, periods, labor, living

Return ONLY the Cypher query, nothing else."""),
        ("human", "{question}")
    ])
    
    chain = cypher_prompt | st.session_state.llm | StrOutputParser()
    try:
        cypher = chain.invoke({"question": question}).strip().replace("```cypher", "").replace("```", "").strip()
    except:
        return []
    
    try:
        driver = st.session_state.graph_rag.driver
        with driver.session() as session:
            records = [dict(r) for r in session.run(cypher)]
    except:
        return []
    
    seen = set()
    results = []
    for r in records:
        if r.get("periods") and r.get("labor"):
            key = f"{r['county']}|{r['state']}"
            if key not in seen:
                seen.add(key)
                results.append({"label": f"{r['county']}, {r['state']}", "periods": r["periods"], "labor": r["labor"], "living": r["living"]})
    return results[:8]

def fetch_ranked_data(question: str) -> list:
    """HYBRID approach: LLM generates Cypher first, validated, with hardcoded fallback.
    
    Flow:
    1. LLM generates a ranking Cypher query (dynamic — handles any question)
    2. Validate results: must have 'label' + 'value', non-empty, correct types
    3. If validation fails → fall back to closest hardcoded proven query
    """
    
    q_lower = question.lower()
    
    # Determine limit from question (default 10)
    import re as _re
    limit_match = _re.search(r'top\s+(\d+)', q_lower)
    limit = int(limit_match.group(1)) if limit_match else 10
    if limit > 15:
        limit = 15
    
    # ------------------------------------------------------------------
    # STEP 1: TRY LLM-GENERATED CYPHER FIRST
    # ------------------------------------------------------------------
    llm_results = _try_llm_ranked_query(question, limit)
    if llm_results:
        st.session_state["_rank_debug"] = f"LLM OK: {len(llm_results)} items"
        return llm_results[:limit]
    
    # ------------------------------------------------------------------
    # STEP 2: FALLBACK TO HARDCODED PROVEN QUERIES
    # ------------------------------------------------------------------
    st.session_state["_rank_debug"] = "LLM failed → using fallback"
    return _fallback_ranked_query(question, limit)


def _try_llm_ranked_query(question: str, limit: int) -> list:
    """Attempt to generate and execute a ranking query via LLM.
    Returns validated results or empty list if anything fails."""
    
    rank_cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Neo4j Cypher expert for the MWR (Minimum Wage Risk) database.
Generate a Cypher query that returns RANKED data for charting.

DATABASE STRUCTURE:
- ZipCode nodes have: newRiskScorePct (0-100 risk score), county, state (abbreviation), 
  cost_of_labor (ERI index, 100=national avg), cost_of_living (ERI index), 
  unemployment_rate, median_household_income, workforce_population,
  pct_bachelors, pct_graduate, pct_no_diploma, total_population
- State nodes have: name (full name), abbr (2-letter abbreviation)
- Relationships: (ZipCode)-[:IN_STATE]->(State)

CRITICAL OUTPUT FORMAT — YOUR QUERY MUST RETURN EXACTLY THESE ALIASES:
- label: (string) The name to display on the chart (state name, "county, ST", etc.)
- value: (number) The numeric value to rank by
- tier:  (string or null) Risk tier if applicable — use this CASE expression for risk queries:
         CASE WHEN value >= 80 THEN 'Critical' WHEN value >= 60 THEN 'High' 
              WHEN value >= 40 THEN 'Elevated' WHEN value >= 20 THEN 'Moderate' ELSE 'Low' END

QUERY PATTERNS:
- State-level rankings: Aggregate ZipCode by z.state, JOIN to State node for full name
  MATCH (z:ZipCode) WHERE z.[field] IS NOT NULL
  WITH z.state AS st, avg(z.[field]) AS value
  MATCH (s:State {{abbr: st}})
  RETURN s.name AS label, value, [tier or null] AS tier
  ORDER BY value DESC/ASC LIMIT N

- County-level rankings: Aggregate ZipCode by county+state
  MATCH (z:ZipCode) WHERE z.[field] IS NOT NULL
  WITH z.county + ', ' + z.state AS label, avg(z.[field]) AS value
  RETURN label, value, [tier or null] AS tier
  ORDER BY value DESC/ASC LIMIT N

FIELD MAPPING:
- "risk", "riskiest", "highest risk" → use newRiskScorePct, include tier
- "cost of labor", "expensive labor", "labor cost" → use cost_of_labor, tier = null
- "cost of living", "expensive living" → use cost_of_living, tier = null  
- "unemployment", "jobless" → use unemployment_rate, tier = null
- "income", "wealthy", "richest" → use median_household_income (use AVG), tier = null
- "population", "largest", "populated" → use total_population, tier = null
  IMPORTANT: Population is stored at county level on each ZIP. To avoid double-counting, FIRST group by county to get one value per county, THEN sum counties for state totals:
  MATCH (z:ZipCode) WHERE z.total_population IS NOT NULL
  WITH z.state AS st, z.county AS county, avg(z.total_population) AS county_pop
  WITH st, sum(county_pop) AS value
  MATCH (s:State {abbr: st}) RETURN s.name AS label, round(value) AS value, null AS tier
- "workforce" → use workforce_population, same deduplication pattern as population, tier = null
- "educated", "education", "college", "degree" → use (z.pct_bachelors + z.pct_graduate) as COMBINED college-educated %, tier = null
  IMPORTANT: For education rankings, always compute the value as avg(z.pct_bachelors + z.pct_graduate).
  Example: WITH z.county + ', ' + z.state AS label, avg(z.pct_bachelors + z.pct_graduate) AS value
- "no diploma", "least educated", "uneducated" → use pct_no_diploma, tier = null
- "workforce" → use workforce_population, tier = null

SORTING:
- "top", "highest", "most expensive", "riskiest" → ORDER BY value DESC
- "lowest", "cheapest", "safest", "least" → ORDER BY value ASC

Return ONLY the Cypher query. No explanations, no markdown."""),
        ("human", "Question: {question}\nLimit: {limit}")
    ])
    
    chain = rank_cypher_prompt | st.session_state.llm | StrOutputParser()
    
    try:
        cypher = chain.invoke({"question": question, "limit": str(limit)}).strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        st.session_state["_rank_debug"] = f"LLM prompt error: {str(e)[:100]}"
        return []
    
    st.session_state["_rank_cypher"] = cypher[:300]
    
    # Execute
    try:
        driver = st.session_state.graph_rag.driver
        with driver.session() as session:
            records = [dict(r) for r in session.run(cypher)]
    except Exception as e:
        st.session_state["_rank_debug"] = f"LLM Cypher exec error: {str(e)[:100]}"
        return []
    
    # ------------------------------------------------------------------
    # VALIDATE: must have 'label' and 'value', at least 1 record
    # ------------------------------------------------------------------
    if not records:
        return []
    
    validated = []
    for r in records:
        label = r.get("label")
        value = r.get("value")
        if label is not None and value is not None:
            try:
                validated.append({
                    "label": str(label),
                    "value": round(float(value), 1),
                    "tier": r.get("tier")
                })
            except (ValueError, TypeError):
                continue
    
    # Need at least 2 items for a meaningful ranking chart
    if len(validated) < 2:
        return []
    
    return validated


def _fallback_ranked_query(question: str, limit: int) -> list:
    """Hardcoded proven queries — guaranteed to work. Used when LLM fails."""
    
    q_lower = question.lower()
    
    # Determine sort direction
    is_ascending = any(kw in q_lower for kw in ['lowest', 'safest', 'least', 'cheapest', 'bottom'])
    order = "ASC" if is_ascending else "DESC"
    
    # Select the right query
    if any(kw in q_lower for kw in ['county', 'counties']):
        # County-level ranking
        is_education = any(kw in q_lower for kw in ['educated', 'education', 'college', 'bachelor', 'degree'])
        is_no_diploma = any(kw in q_lower for kw in ['no diploma', 'uneducated', 'least educated', 'dropout'])
        
        if is_education or is_no_diploma:
            # Education needs special Cypher — combined field or pct_no_diploma
            if is_no_diploma:
                cypher = f"""
                MATCH (z:ZipCode) WHERE z.pct_no_diploma IS NOT NULL
                WITH z.county + ', ' + z.state AS label, avg(z.pct_no_diploma) AS value
                RETURN label, value, null AS tier
                ORDER BY value {order} LIMIT {limit}
                """
            else:
                cypher = f"""
                MATCH (z:ZipCode) WHERE z.pct_bachelors IS NOT NULL AND z.pct_graduate IS NOT NULL
                WITH z.county + ', ' + z.state AS label, avg(z.pct_bachelors + z.pct_graduate) AS value
                RETURN label, value, null AS tier
                ORDER BY value {order} LIMIT {limit}
                """
        else:
            if any(kw in q_lower for kw in ['labor', 'cost of labor', 'expensive labor']):
                field, tier_expr = "cost_of_labor", "null"
                where = "z.cost_of_labor IS NOT NULL AND z.cost_of_labor > 0"
            elif any(kw in q_lower for kw in ['living', 'cost of living']):
                field, tier_expr = "cost_of_living", "null"
                where = "z.cost_of_living IS NOT NULL AND z.cost_of_living > 0"
            elif any(kw in q_lower for kw in ['unemployment', 'jobless']):
                field, tier_expr = "unemployment_rate", "null"
                where = "z.unemployment_rate IS NOT NULL"
            else:
                field = "newRiskScorePct"
                tier_expr = """CASE WHEN value >= 80 THEN 'Critical' WHEN value >= 60 THEN 'High' 
                              WHEN value >= 40 THEN 'Elevated' WHEN value >= 20 THEN 'Moderate' ELSE 'Low' END"""
                where = "z.newRiskScorePct IS NOT NULL"
            
            cypher = f"""
            MATCH (z:ZipCode) WHERE {where}
            WITH z.county + ', ' + z.state AS label, avg(z.{field}) AS value
            RETURN label, value, {tier_expr} AS tier
            ORDER BY value {order} LIMIT {limit}
            """
    
    else:
        # State-level ranking
        is_education = any(kw in q_lower for kw in ['educated', 'education', 'college', 'bachelor', 'degree'])
        is_no_diploma = any(kw in q_lower for kw in ['no diploma', 'uneducated', 'least educated', 'dropout'])
        
        if is_education or is_no_diploma:
            if is_no_diploma:
                cypher = f"""
                MATCH (z:ZipCode) WHERE z.pct_no_diploma IS NOT NULL
                WITH z.state AS st, avg(z.pct_no_diploma) AS value
                MATCH (s:State {{abbr: st}})
                RETURN s.name AS label, value, null AS tier
                ORDER BY value {order} LIMIT {limit}
                """
            else:
                cypher = f"""
                MATCH (z:ZipCode) WHERE z.pct_bachelors IS NOT NULL AND z.pct_graduate IS NOT NULL
                WITH z.state AS st, avg(z.pct_bachelors + z.pct_graduate) AS value
                MATCH (s:State {{abbr: st}})
                RETURN s.name AS label, value, null AS tier
                ORDER BY value {order} LIMIT {limit}
                """
        else:
            if any(kw in q_lower for kw in ['labor', 'cost of labor', 'expensive labor']):
                field, tier_expr = "cost_of_labor", "null"
                where = "z.cost_of_labor IS NOT NULL AND z.cost_of_labor > 0"
            elif any(kw in q_lower for kw in ['living', 'cost of living']):
                field, tier_expr = "cost_of_living", "null"
                where = "z.cost_of_living IS NOT NULL AND z.cost_of_living > 0"
            elif any(kw in q_lower for kw in ['unemployment', 'jobless']):
                field, tier_expr = "unemployment_rate", "null"
                where = "z.unemployment_rate IS NOT NULL"
            elif any(kw in q_lower for kw in ['income', 'wealthy', 'richest']):
                field, tier_expr = "median_household_income", "null"
                where = "z.median_household_income IS NOT NULL"
            elif any(kw in q_lower for kw in ['population', 'largest', 'biggest', 'populated']):
                # Population: deduplicate by county first (since all ZIPs in a county share the county pop),
                # then sum counties to get state total
                cypher = f"""
                MATCH (z:ZipCode) WHERE z.total_population IS NOT NULL
                WITH z.state AS st, z.county AS county, avg(z.total_population) AS county_pop
                WITH st, sum(county_pop) AS value
                MATCH (s:State {{abbr: st}})
                RETURN s.name AS label, round(value) AS value, null AS tier
                ORDER BY value {order} LIMIT {limit}
                """
            else:
                field = "newRiskScorePct"
                tier_expr = """CASE WHEN value >= 80 THEN 'Critical' WHEN value >= 60 THEN 'High' 
                              WHEN value >= 40 THEN 'Elevated' WHEN value >= 20 THEN 'Moderate' ELSE 'Low' END"""
                where = "z.newRiskScorePct IS NOT NULL"
            
            # Build generic cypher for all field-based metrics (not population which has its own)
            if not any(kw in q_lower for kw in ['population', 'largest', 'biggest', 'populated']):
                cypher = f"""
                MATCH (z:ZipCode) WHERE {where}
                WITH z.state AS st, avg(z.{field}) AS value
                MATCH (s:State {{abbr: st}})
                RETURN s.name AS label, value, {tier_expr} AS tier
                ORDER BY value {order} LIMIT {limit}
                """
    
    st.session_state["_rank_cypher"] = cypher.strip()[:300]
    
    try:
        driver = st.session_state.graph_rag.driver
        with driver.session() as session:
            records = [dict(r) for r in session.run(cypher)]
    except Exception as e:
        st.session_state["_rank_debug"] = f"Fallback error: {str(e)[:200]}"
        return []
    
    results = []
    for r in records:
        if r.get("label") is not None and r.get("value") is not None:
            results.append({"label": str(r["label"]), "value": round(float(r["value"]), 1), "tier": r.get("tier")})
    return results[:limit]


# ============================================================
# CHART CREATION FUNCTIONS
# ============================================================

def create_trend_chart(trend_data: list, question: str) -> go.Figure:
    """Create an interactive line chart showing trends over time"""
    metric_type = detect_metric_type(question)
    if not trend_data:
        return None
    
    fig = go.Figure()
    periods = trend_data[0]["periods"]
    x_labels = [PERIOD_LABELS.get(p, p) for p in periods]
    color_idx = 0
    
    for loc in trend_data:
        label = loc["label"]
        if metric_type in ["labor", "both", "risk"]:
            filtered = [(x, v) for x, v in zip(x_labels, loc["labor"]) if v > 0]
            if filtered:
                x_vals, y_vals = zip(*filtered)
                suffix = " — Labor" if metric_type == "both" else ""
                fig.add_trace(go.Scatter(x=list(x_vals), y=list(y_vals), mode='lines+markers',
                    name=f"{label}{suffix}", line=dict(color=CHART_COLORS[color_idx % len(CHART_COLORS)], width=3),
                    marker=dict(size=8), hovertemplate='%{x}<br>Index: %{y:.1f}<extra>' + label + '</extra>'))
                color_idx += 1
        if metric_type in ["living", "both"]:
            filtered = [(x, v) for x, v in zip(x_labels, loc["living"]) if v > 0]
            if filtered:
                x_vals, y_vals = zip(*filtered)
                suffix = " — Living" if metric_type == "both" else ""
                fig.add_trace(go.Scatter(x=list(x_vals), y=list(y_vals), mode='lines+markers',
                    name=f"{label}{suffix}", line=dict(color=CHART_COLORS[color_idx % len(CHART_COLORS)], width=3,
                    dash='dash' if metric_type == "both" else 'solid'), marker=dict(size=8),
                    hovertemplate='%{x}<br>Index: %{y:.1f}<extra>' + label + '</extra>'))
                color_idx += 1
    
    # Title
    if len(trend_data) == 1:
        loc_name = trend_data[0]["label"]
        title = {"labor": f"Cost of Labor Trend — {loc_name}", "living": f"Cost of Living Trend — {loc_name}"}.get(metric_type, f"ERI Cost Trends — {loc_name}")
    else:
        title = {"labor": "Cost of Labor — Trend Comparison", "living": "Cost of Living — Trend Comparison"}.get(metric_type, "ERI Cost Trends — Comparison")
    
    fig.update_layout(title=dict(text=title, font=dict(size=20, color='#1B4F5C', family='Arial')),
        xaxis_title="", yaxis_title="ERI Index (100 = National Average)", hovermode='x unified',
        template='plotly_white', height=500, plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(l=60, r=30, t=70, b=80),
        yaxis=dict(gridcolor='#E8E8E8', tickfont=dict(size=11)),
        xaxis=dict(gridcolor='#E8E8E8', tickangle=-45, tickfont=dict(size=11)))
    fig.add_hline(y=100, line_dash="dash", line_color="#E74C3C", line_width=1.5,
        annotation_text="National Avg (100)", annotation_position="bottom right",
        annotation_font=dict(color="#E74C3C", size=11))
    return fig

def create_bar_chart(trend_data: list, question: str) -> go.Figure:
    """Create a vertical bar chart comparing current values across locations"""
    metric_type = detect_metric_type(question)
    if not trend_data:
        return None
    
    labels, labor_values, living_values = [], [], []
    for loc in trend_data:
        labels.append(loc["label"])
        lv = [v for v in loc["labor"] if v > 0]
        cv = [v for v in loc["living"] if v > 0]
        labor_values.append(lv[-1] if lv else 0)
        living_values.append(cv[-1] if cv else 0)
    
    fig = go.Figure()
    if metric_type in ["labor", "both", "risk"]:
        fig.add_trace(go.Bar(x=labels, y=labor_values, name="Cost of Labor",
            marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(labels))] if metric_type == "labor" else [CHART_COLORS[0]] * len(labels),
            text=[f"{v:.1f}" for v in labor_values], textposition='outside',
            textfont=dict(size=14, color='#333333', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>Cost of Labor: %{y:.1f}<br>vs National Avg: %{customdata:+.1f}<extra></extra>',
            customdata=[v - 100 for v in labor_values]))
    if metric_type in ["living", "both"]:
        fig.add_trace(go.Bar(x=labels, y=living_values, name="Cost of Living",
            marker_color=[CHART_COLORS[(i+1) % len(CHART_COLORS)] for i in range(len(labels))] if metric_type == "living" else [CHART_COLORS[1]] * len(labels),
            text=[f"{v:.1f}" for v in living_values], textposition='outside',
            textfont=dict(size=14, color='#333333', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>Cost of Living: %{y:.1f}<br>vs National Avg: %{customdata:+.1f}<extra></extra>',
            customdata=[v - 100 for v in living_values]))
    
    title = {"labor": "Cost of Labor Comparison — Latest Period", "living": "Cost of Living Comparison — Latest Period"}.get(metric_type, "ERI Cost Comparison — Latest Period")
    all_vals = (labor_values if metric_type in ["labor", "both", "risk"] else []) + (living_values if metric_type in ["living", "both"] else [])
    y_max = max(all_vals) * 1.15 if all_vals else 150
    
    fig.update_layout(title=dict(text=title, font=dict(size=20, color='#1B4F5C', family='Arial')),
        xaxis_title="", yaxis_title="ERI Index (100 = National Average)", barmode='group',
        template='plotly_white', height=500, plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(l=60, r=30, t=70, b=60),
        yaxis=dict(gridcolor='#E8E8E8', range=[0, y_max]),
        xaxis=dict(tickfont=dict(size=13, color='#333333')))
    fig.add_hline(y=100, line_dash="dash", line_color="#E74C3C", line_width=2,
        annotation_text="National Avg (100)", annotation_position="bottom right",
        annotation_font=dict(color="#E74C3C", size=11))
    return fig

def create_hbar_chart(ranked_data: list, question: str) -> go.Figure:
    """Create a horizontal bar chart for ranked data — Power BI quality"""
    if not ranked_data:
        return None
    
    # Reverse so highest appears at top
    ranked_data = list(reversed(ranked_data))
    labels = [r["label"] for r in ranked_data]
    values = [r["value"] for r in ranked_data]
    tiers = [r.get("tier") for r in ranked_data]
    
    # Determine colors
    if any(tiers):
        colors = []
        for r in ranked_data:
            tier = r.get("tier")
            if tier and tier in RISK_COLORS:
                colors.append(RISK_COLORS[tier])
            else:
                v = r["value"]
                if v >= 80: colors.append(RISK_COLORS['Critical'])
                elif v >= 60: colors.append(RISK_COLORS['High'])
                elif v >= 40: colors.append(RISK_COLORS['Elevated'])
                elif v >= 20: colors.append(RISK_COLORS['Moderate'])
                else: colors.append(RISK_COLORS['Low'])
    else:
        max_val = max(values) if values else 1
        min_val = min(values) if values else 0
        colors = []
        for v in values:
            ratio = (v - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            if ratio > 0.75: colors.append('#CC0000')
            elif ratio > 0.5: colors.append('#E74C3C')
            elif ratio > 0.25: colors.append('#F39C12')
            else: colors.append('#2ECC71')
    
    fig = go.Figure()
    
    # Determine metric type for number formatting
    q_lower = question.lower()
    is_county = ('county' in q_lower or 'counties' in q_lower)
    geo = "Counties" if is_county else "States"
    
    # Detect metric for formatting
    if any(kw in q_lower for kw in ['income', 'wealthy', 'richest']):
        metric_fmt = "currency"
    elif any(kw in q_lower for kw in ['population', 'largest', 'biggest', 'populated', 'workforce']):
        metric_fmt = "population"
    else:
        metric_fmt = "decimal"  # risk %, unemployment %, ERI index, education %
    
    # Format bar labels based on metric type
    def fmt_value(v):
        if metric_fmt == "currency":
            return f"  ${v:,.0f}"
        elif metric_fmt == "population":
            if v >= 1_000_000_000:
                return f"  {v/1_000_000_000:.1f}B"
            elif v >= 1_000_000:
                return f"  {v/1_000_000:.1f}M"
            elif v >= 1_000:
                return f"  {v:,.0f}"
            else:
                return f"  {v:,.0f}"
        else:
            return f"  {v:.1f}"
    
    bar_texts = [fmt_value(v) for v in values]
    
    fig.add_trace(go.Bar(y=labels, x=values, orientation='h', marker_color=colors,
        text=bar_texts, textposition='outside',
        textfont=dict(size=13, color='#333333', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Value: %{x:,.1f}<extra></extra>'))
    
    if 'risk' in q_lower or 'score' in q_lower:
        title = f"{geo} by Average Risk Score (%)"
        x_label = "Risk Score (%)"
    elif 'labor' in q_lower:
        title, x_label = f"{geo} by Cost of Labor", "ERI Index (100 = National Avg)"
    elif 'living' in q_lower:
        title, x_label = f"{geo} by Cost of Living", "ERI Index (100 = National Avg)"
    elif 'expensive' in q_lower:
        title, x_label = f"Most Expensive {geo}", "ERI Index (100 = National Avg)"
    elif any(kw in q_lower for kw in ['unemployment', 'jobless']):
        title, x_label = f"{geo} by Unemployment Rate", "Unemployment Rate (%)"
    elif any(kw in q_lower for kw in ['educated', 'education', 'college', 'bachelor', 'degree']):
        title, x_label = f"{geo} by College Education Rate", "College-Educated (%)"
    elif any(kw in q_lower for kw in ['no diploma', 'uneducated', 'dropout']):
        title, x_label = f"{geo} by No Diploma Rate", "No Diploma (%)"
    elif any(kw in q_lower for kw in ['income', 'wealthy', 'richest']):
        title, x_label = f"{geo} by Median Household Income", "Median Income ($)"
    elif any(kw in q_lower for kw in ['population', 'largest', 'biggest', 'populated']):
        title, x_label = f"{geo} by Total Population", "Population"
    elif any(kw in q_lower for kw in ['workforce']):
        title, x_label = f"{geo} by Workforce Population", "Workforce (18-64)"
    else:
        title, x_label = f"{geo} Rankings", "Value"
    
    chart_height = max(400, len(ranked_data) * 40 + 120)
    
    # Dynamic right margin based on label length
    if metric_fmt == "currency":
        r_margin = 140
    elif metric_fmt == "population":
        r_margin = 100
    else:
        r_margin = 80
    
    fig.update_layout(title=dict(text=title, font=dict(size=20, color='#1B4F5C', family='Arial')),
        xaxis_title=x_label, yaxis_title="", template='plotly_white', height=chart_height,
        plot_bgcolor='white', showlegend=False,
        margin=dict(l=200, r=r_margin, t=70, b=50),
        xaxis=dict(gridcolor='#E8E8E8', tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12, color='#333333')))
    
    # Add x-axis padding so text labels don't overflow
    if metric_fmt in ("currency", "population") and values:
        max_val = max(values)
        fig.update_xaxes(range=[0, max_val * 1.25])
    
    if 'risk' in q_lower or 'score' in q_lower:
        fig.add_vline(x=80, line_dash="dash", line_color="#CC0000", line_width=1.5,
            annotation_text="Critical (80)", annotation_position="top",
            annotation_font=dict(color="#CC0000", size=10))
    elif any(kw in q_lower for kw in ['labor', 'living', 'cost', 'expensive']):
        fig.add_vline(x=100, line_dash="dash", line_color="#E74C3C", line_width=1.5,
            annotation_text="National Avg (100)", annotation_position="top",
            annotation_font=dict(color="#E74C3C", size=10))
    elif any(kw in q_lower for kw in ['unemployment', 'jobless']):
        fig.add_vline(x=4.0, line_dash="dash", line_color="#F39C12", line_width=1.5,
            annotation_text="Healthy (4%)", annotation_position="top",
            annotation_font=dict(color="#F39C12", size=10))
    elif any(kw in q_lower for kw in ['educated', 'education', 'college', 'bachelor', 'degree']):
        fig.add_vline(x=33, line_dash="dash", line_color="#3498DB", line_width=1.5,
            annotation_text="National Avg (~33%)", annotation_position="top",
            annotation_font=dict(color="#3498DB", size=10))
    
    return fig

def create_chart(chart_type: str, data: list, question: str) -> go.Figure:
    """Router: create the appropriate chart based on type"""
    if chart_type == "LINE_TREND":
        return create_trend_chart(data, question)
    elif chart_type == "BAR_COMPARE":
        return create_bar_chart(data, question)
    elif chart_type == "HBAR_RANK":
        return create_hbar_chart(data, question)
    return None


def generate_response(question: str) -> dict:
    """Generate comprehensive response with session memory. Returns dict with 'text' and optional 'chart'."""
    
    # Step 1: Resolve follow-up questions using conversation history
    resolved_question = resolve_follow_up(question)
    
    # Reset trend unavailable flag
    st.session_state["_trend_unavailable"] = False
    
    # Step 2: Intelligent chart generation
    chart_data = None
    chart_error = None
    chart_attempted = False  # Track if we tried to generate a chart
    chart_failed_reason = None  # User-friendly reason if chart fails
    
    if should_generate_chart(resolved_question):
        chart_attempted = True
        try:
            chart_type = classify_chart_type(resolved_question)
            
            # Safety guard: LINE_TREND only valid for ERI cost data
            if chart_type == "LINE_TREND":
                q_check = resolved_question.lower()
                has_eri_keywords = any(kw in q_check for kw in [
                    'cost of labor', 'cost of living', 'eri', 'col ', 'coliv',
                    'labor cost', 'living cost', 'labor trend', 'living trend'
                ])
                if not has_eri_keywords:
                    chart_type = "NONE"
                    st.session_state["_trend_unavailable"] = True
                    chart_failed_reason = "trend_unavailable"
            
            # Safety net: Force HBAR_RANK if classifier says NONE but question is clearly a ranking
            if chart_type == "NONE" and not chart_failed_reason:
                q_check = resolved_question.lower()
                has_rank_signal = any(kw in q_check for kw in [
                    'top ', 'highest', 'lowest', 'rank', 'most ', 'least ',
                    'best ', 'worst', 'safest', 'riskiest', 'cheapest', 'expensive',
                    'wealthiest', 'richest', 'poorest', 'bottom '
                ])
                has_metric = any(kw in q_check for kw in [
                    'risk', 'income', 'population', 'unemployment', 'education',
                    'labor', 'living', 'cost', 'populated', 'educated', 'workforce'
                ])
                if has_rank_signal and has_metric:
                    chart_type = "HBAR_RANK"
            
            chart_error = f"type={chart_type}"
            
            if chart_type == "LINE_TREND":
                data = fetch_trend_data(resolved_question)
                if data:
                    chart_data = {"chart_type": chart_type, "data": data, "question": resolved_question}
                    chart_error += f", OK: {len(data)} locations"
                else:
                    chart_error += ", no trend data found"
                    chart_failed_reason = chart_failed_reason or "no_data"
            
            elif chart_type == "BAR_COMPARE":
                data = fetch_trend_data(resolved_question)
                if data:
                    chart_data = {"chart_type": chart_type, "data": data, "question": resolved_question}
                    chart_error += f", OK: {len(data)} locations"
                else:
                    chart_error += ", no comparison data found"
                    chart_failed_reason = chart_failed_reason or "no_data"
            
            elif chart_type == "HBAR_RANK":
                data = fetch_ranked_data(resolved_question)
                if data:
                    chart_data = {"chart_type": chart_type, "data": data, "question": resolved_question}
                    chart_error += f", OK: {len(data)} items"
                else:
                    chart_error += ", no ranking data found"
                    chart_failed_reason = chart_failed_reason or "no_data"
            
        except Exception as e:
            chart_error = str(e)[:200]
            chart_failed_reason = "error"
    else:
        chart_error = "no chart needed"
    
    st.session_state["_chart_debug"] = f"q='{resolved_question[:60]}' | {chart_error}"
    
    # Step 3: Classify the resolved question
    q_type = classify_question(resolved_question)
    
    context_parts = []
    
    # Get database results if needed
    if q_type in ["DATABASE", "BOTH"]:
        try:
            db_result = st.session_state.graph_rag.answer_question(resolved_question)
            context_parts.append(f"**Database Results:**\n{db_result['answer']}")
        except Exception as e:
            context_parts.append(f"Database query error: {e}")
    
    # Get web results if needed
    if q_type in ["WEB_SEARCH", "BOTH"]:
        web_results = search_web(resolved_question)
        if web_results:
            web_text = "\n**Recent News:**\n"
            for r in web_results[:2]:
                web_text += f"- {r.get('title', 'No title')}\n"
            context_parts.append(web_text)
    
    # Build conversation context for the LLM
    conversation_context = get_conversation_context()
    
    # Generate final answer WITH conversation history
    if context_parts:
        context = "\n\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the MWR AI Assistant for Sodexo — a workforce market intelligence system.
You help executives analyze Minimum Wage Risk, labor market conditions, and workforce demographics 
to support contract bidding and strategic planning decisions.

DATA AVAILABLE IN THE DATABASE:
- Risk Scores: ZIP-level and county-level risk scores (0-100) with tiers (Critical/High/Elevated/Moderate/Low)
- Education (5 levels): No Diploma %, HS Diploma %, Some College %, Bachelor's %, Graduate %
- Workforce Population: Working-age population (18-64) by county
- Cost of Labor: ERI index where 100 = national average (higher = more expensive labor)
- Cost of Living: ERI index where 100 = national average (higher = more expensive area)
- Unemployment Rate: County-level BLS unemployment data
- Demographics: Total population, median income, median age, median home value
- Geography: 40,000+ ZIP codes linked to counties, states, and CBSAs

When answering, provide clear business insights. Use specific numbers from the data.
Explain what the numbers mean for Sodexo's business — contract pricing, talent competition, wage pressures.
When presenting education data, show all 5 levels to paint a complete workforce picture.
When discussing costs, explain what high/low indices mean for bidding strategy.

IMPORTANT: You have memory of this conversation. Use the conversation history 
to understand context and provide coherent follow-up answers. If the user 
refers to something discussed earlier, use that context to give a relevant answer.

Previous Conversation:
{conversation_history}"""),
            ("human", """Question: {question}

Context:
{context}

Provide a helpful answer:""")
        ])
        
        chain = prompt | st.session_state.llm | StrOutputParser()
        text = chain.invoke({
            "question": question,
            "context": context,
            "conversation_history": conversation_context if conversation_context else "No previous conversation."
        })
        # Append chart status note if applicable
        if chart_failed_reason == "trend_unavailable":
            text += "\n\n---\n📊 *Sorry, we don't have enough historical data to show a trend chart for this metric at this time. Trend charts are currently available for Cost of Labor and Cost of Living only.*"
        elif chart_failed_reason == "no_data" and chart_attempted:
            text += "\n\n---\n📊 *A chart was requested but could not be generated — the data query returned no results. Try rephrasing your question or specifying states/counties.*"
        elif chart_failed_reason == "error" and chart_attempted:
            text += "\n\n---\n📊 *A chart could not be generated due to a system error. The text analysis above is still accurate.*"
        return {"text": text, "chart_data": chart_data}
    else:
        # General question - answer directly with conversation history
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the MWR AI Assistant for Sodexo — a workforce market intelligence system.
You help executives analyze minimum wage risk, labor costs, education demographics, and workforce data.
Answer questions clearly and concisely with business context.

IMPORTANT: You have memory of this conversation. Use the conversation history 
to understand context and provide coherent follow-up answers.

Previous Conversation:
{conversation_history}"""),
            ("human", "{question}")
        ])
        chain = prompt | st.session_state.llm | StrOutputParser()
        text = chain.invoke({
            "question": question,
            "conversation_history": conversation_context if conversation_context else "No previous conversation."
        })
        # Append chart status note if applicable
        if chart_failed_reason == "trend_unavailable":
            text += "\n\n---\n📊 *Sorry, we don't have enough historical data to show a trend chart for this metric at this time. Trend charts are currently available for Cost of Labor and Cost of Living only.*"
        elif chart_failed_reason == "no_data" and chart_attempted:
            text += "\n\n---\n📊 *A chart was requested but could not be generated — the data query returned no results. Try rephrasing your question or specifying states/counties.*"
        elif chart_failed_reason == "error" and chart_attempted:
            text += "\n\n---\n📊 *A chart could not be generated due to a system error. The text analysis above is still accurate.*"
        return {"text": text, "chart_data": chart_data}

def process_question(question: str):
    """Process a question and update chat history"""
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Generate response
    response = generate_response(question)
    
    # Add assistant response to display (store both text and chart data)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["text"],
        "chart_data": response.get("chart_data")
    })
    
    # Add to conversation history (session memory) — text only
    st.session_state.conversation_history.append({"role": "user", "content": question})
    st.session_state.conversation_history.append({"role": "assistant", "content": response["text"]})

# ============================================================
# MAIN UI
# ============================================================
# Title with memory indicator
num_turns = len(st.session_state.conversation_history) // 2
if num_turns > 0:
    st.markdown(
        f'<p class="main-title">🤖 MWR AI Assistant '
        f'<span class="memory-badge">🧠 {num_turns} exchanges remembered</span></p>',
        unsafe_allow_html=True
    )
else:
    st.markdown('<p class="main-title">🤖 MWR AI Assistant</p>', unsafe_allow_html=True)

# Connection status
if st.session_state.get("connected"):
    st.success("✅ Connected to MWR Database", icon="✅")
else:
    st.error("❌ Not connected - check Neo4j")
    st.stop()

# ============================================================
# QUICK ACTION BUTTONS - FIXED VERSION
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Top 5 Risk States", use_container_width=True, key="btn_top5"):
        st.session_state.pending_question = "What are the top 5 highest risk states?"

with col2:
    if st.button("🔴 Critical ZIPs", use_container_width=True, key="btn_critical"):
        st.session_state.pending_question = "How many ZIP codes are in the Critical risk tier?"

with col3:
    if st.button("☀️ California Risk", use_container_width=True, key="btn_california"):
        st.session_state.pending_question = "What is California's risk score and why?"

with col4:
    if st.button("🗞️ Latest News", use_container_width=True, key="btn_news"):
        st.session_state.pending_question = "What are the latest minimum wage news and changes?"

# Process pending button question BEFORE displaying chat history
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None  # Clear it
    
    with st.spinner("🔍 Analyzing..."):
        process_question(question)
    
    st.rerun()  # Rerun to display the new messages

st.divider()

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Recreate and render chart from stored data if present
        if message.get("chart_data") is not None:
            try:
                cd = message["chart_data"]
                fig = create_chart(cd["chart_type"], cd["data"], cd["question"])
                if fig:
                    chart_key = f"chart_{id(message)}_{hash(message['content'][:50])}"
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                else:
                    st.warning("Chart function returned None")
            except Exception as e:
                st.error(f"Chart render error: {e}")

# ============================================================
# CHAT INPUT
# ============================================================
if prompt := st.chat_input("Ask me anything about Minimum Wage Risk..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            response = generate_response(prompt)
            st.markdown(response["text"])
            # Render chart if data present
            if response.get("chart_data") is not None:
                try:
                    cd = response["chart_data"]
                    fig = create_chart(cd["chart_type"], cd["data"], cd["question"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="chart_live")
                except Exception as e:
                    st.error(f"Chart error: {e}")
    
    # Update session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["text"],
        "chart_data": response.get("chart_data")
    })
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "assistant", "content": response["text"]})
    
    # Rerun to update memory badge
    st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **MWR AI Assistant** helps you analyze 
    workforce market intelligence data.
    
    **You can ask about:**
    - Risk scores by state/ZIP/county
    - Education demographics (5 levels)
    - Workforce population (ages 18-64)
    - Cost of labor & cost of living
    - **ERI cost trends with charts! 📈**
    - Unemployment rates
    - Market profiles for bidding
    - Latest MW news
    - **Follow-up questions!** 🧠
    
    **Data Sources:**
    - Neo4j MWR Database
    - Census Bureau (ACS)
    - ERI Economic Research
    - BLS Unemployment Data
    - Tavily Web Search
    
    **Session Memory:**
    The assistant remembers your 
    conversation in this session.
    Ask follow-ups naturally!
    """)
    
    st.divider()
    
    # Memory stats
    turns = len(st.session_state.conversation_history) // 2
    st.markdown(f"🧠 **Memory:** {turns} exchanges")
    st.markdown(f"📊 **Max Memory:** {MAX_MEMORY_TURNS} exchanges")
    
    if st.button("🗑️ Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.pending_question = None
        st.rerun()
    
    # Developer debug (hidden from main chat)
    st.divider()
    st.markdown("**🔧 Debug (dev only)**")
    st.caption(st.session_state.get("_chart_debug", "—"))
    st.caption(st.session_state.get("_rank_debug", "—"))
    rank_cypher = st.session_state.get("_rank_cypher", "")
    if rank_cypher:
        st.code(rank_cypher[:200], language="cypher")