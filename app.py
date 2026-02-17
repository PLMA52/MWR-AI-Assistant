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
- LINE_TREND: Time-series questions about how something changed over time (trends, history, trajectory)
- BAR_COMPARE: Side-by-side comparison of current values between 2-5 specific named locations
- HBAR_RANK: Ranked list of items (top/bottom states, counties, highest/lowest risk, most expensive)
- NONE: Question doesn't benefit from a chart (education breakdowns, general info, ZIP counts, news)

Rules:
- "top 5 risk states" → HBAR_RANK
- "highest risk counties" → HBAR_RANK
- "most expensive states for labor" → HBAR_RANK
- "which states have highest cost of labor" → HBAR_RANK
- "rank states by cost of living" → HBAR_RANK
- "cost of labor trend in SF" → LINE_TREND
- "how has cost changed over time" → LINE_TREND
- "compare cost of labor NY vs MD" → BAR_COMPARE
- "bar chart of costs in SF vs LA" → BAR_COMPARE
- "compare cost of living in San Francisco vs Los Angeles vs San Diego" → BAR_COMPARE
- "what is California's risk?" → NONE (single value, no chart needed)
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
    """Fetch ranked data from Neo4j for horizontal bar charts"""
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate a Neo4j Cypher query to fetch ranked data for a horizontal bar chart.

Available node properties on ZipCode:
- risk_score: numeric risk score (0-100)
- risk_tier: 'Critical', 'High', 'Elevated', 'Moderate', or 'Low'
- cost_of_labor: ERI labor cost index (100 = national average)
- cost_of_living: ERI living cost index (100 = national average)
- county, state: location fields
- unemployment_rate: county-level unemployment

Available node properties on State:
- state_name, state_abbr
- new_risk_score_pct: state-level risk score (0-100)
- state_risk_tier: 'Critical', 'High', 'Elevated', 'Moderate', or 'Low'

RULES:
1. For state rankings: use State nodes, ORDER BY the metric DESC or ASC
2. For county rankings: aggregate ZipCode data by county using avg()
3. Always return: label (name), value (numeric), tier (risk tier if available)
4. Use LIMIT to control count (default 10, or whatever the user asks)
5. ORDER BY value DESC for "top/highest/most", ASC for "bottom/lowest/least"

Examples:
- "top 5 risk states" →
  MATCH (s:State) WHERE s.new_risk_score_pct IS NOT NULL
  RETURN s.state_name AS label, s.new_risk_score_pct AS value, s.state_risk_tier AS tier
  ORDER BY s.new_risk_score_pct DESC LIMIT 5

- "top 10 highest risk counties" →
  MATCH (z:ZipCode) WHERE z.risk_score IS NOT NULL
  WITH z.county + ', ' + z.state AS label, avg(z.risk_score) AS value
  RETURN label, value, 
    CASE WHEN value >= 80 THEN 'Critical' WHEN value >= 60 THEN 'High' WHEN value >= 40 THEN 'Elevated' WHEN value >= 20 THEN 'Moderate' ELSE 'Low' END AS tier
  ORDER BY value DESC LIMIT 10

- "most expensive states for labor" →
  MATCH (z:ZipCode) WHERE z.cost_of_labor IS NOT NULL AND z.cost_of_labor > 0
  WITH z.state AS st, avg(z.cost_of_labor) AS avg_cost
  MATCH (s:State {state_abbr: st})
  RETURN s.state_name AS label, avg_cost AS value, null AS tier
  ORDER BY avg_cost DESC LIMIT 10

- "states with lowest cost of living" →
  MATCH (z:ZipCode) WHERE z.cost_of_living IS NOT NULL AND z.cost_of_living > 0
  WITH z.state AS st, avg(z.cost_of_living) AS avg_cost
  MATCH (s:State {state_abbr: st})
  RETURN s.state_name AS label, avg_cost AS value, null AS tier
  ORDER BY avg_cost ASC LIMIT 10

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
    
    results = []
    for r in records:
        if r.get("label") is not None and r.get("value") is not None:
            results.append({"label": str(r["label"]), "value": round(float(r["value"]), 1), "tier": r.get("tier")})
    return results[:15]

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
    fig.add_trace(go.Bar(y=labels, x=values, orientation='h', marker_color=colors,
        text=[f"  {v:.1f}" for v in values], textposition='outside',
        textfont=dict(size=13, color='#333333', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>'))
    
    q_lower = question.lower()
    if 'risk' in q_lower or 'score' in q_lower:
        title = "Counties by Average Risk Score (%)" if ('county' in q_lower or 'counties' in q_lower) else "States by Average Risk Score (%)"
        x_label = "Risk Score (%)"
    elif 'labor' in q_lower:
        title, x_label = "Cost of Labor Rankings", "ERI Index (100 = National Avg)"
    elif 'living' in q_lower:
        title, x_label = "Cost of Living Rankings", "ERI Index (100 = National Avg)"
    elif 'expensive' in q_lower:
        title, x_label = "Most Expensive Markets", "ERI Index (100 = National Avg)"
    else:
        title, x_label = "Rankings", "Value"
    
    chart_height = max(400, len(ranked_data) * 40 + 120)
    
    fig.update_layout(title=dict(text=title, font=dict(size=20, color='#1B4F5C', family='Arial')),
        xaxis_title=x_label, yaxis_title="", template='plotly_white', height=chart_height,
        plot_bgcolor='white', showlegend=False,
        margin=dict(l=200, r=80, t=70, b=50),
        xaxis=dict(gridcolor='#E8E8E8', tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12, color='#333333')))
    
    if 'risk' in q_lower or 'score' in q_lower:
        fig.add_vline(x=80, line_dash="dash", line_color="#CC0000", line_width=1.5,
            annotation_text="Critical (80)", annotation_position="top",
            annotation_font=dict(color="#CC0000", size=10))
    elif any(kw in q_lower for kw in ['labor', 'living', 'cost', 'expensive']):
        fig.add_vline(x=100, line_dash="dash", line_color="#E74C3C", line_width=1.5,
            annotation_text="National Avg (100)", annotation_position="top",
            annotation_font=dict(color="#E74C3C", size=10))
    
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
    
    # Step 2: Intelligent chart generation
    chart_data = None
    chart_error = None
    if should_generate_chart(resolved_question):
        try:
            chart_type = classify_chart_type(resolved_question)
            chart_error = f"type={chart_type}"
            
            if chart_type == "LINE_TREND":
                data = fetch_trend_data(resolved_question)
                if data:
                    chart_data = {"chart_type": chart_type, "data": data, "question": resolved_question}
                    chart_error += f", OK: {len(data)} locations"
                else:
                    chart_error += ", no data"
            
            elif chart_type == "BAR_COMPARE":
                data = fetch_trend_data(resolved_question)  # Uses latest period values
                if data:
                    chart_data = {"chart_type": chart_type, "data": data, "question": resolved_question}
                    chart_error += f", OK: {len(data)} locations"
                else:
                    chart_error += ", no data"
            
            elif chart_type == "HBAR_RANK":
                data = fetch_ranked_data(resolved_question)
                if data:
                    chart_data = {"chart_type": chart_type, "data": data, "question": resolved_question}
                    chart_error += f", OK: {len(data)} items"
                else:
                    chart_error += ", no data"
            
        except Exception as e:
            chart_error = str(e)[:200]
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
        # Show if chart_data exists for debugging
        if message["role"] == "assistant":
            has_chart = message.get("chart_data") is not None
            debug_msg = st.session_state.get("_chart_debug", "N/A")
            st.caption(f"🔧 chart_data: {has_chart} | {debug_msg}")
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