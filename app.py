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
load_dotenv()

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
# PLOTLY CHART GENERATION
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

PERIOD_LABELS = {
    '2024-05': 'May 2024', '2024-07': 'Jul 2024', '2024-10': 'Oct 2024',
    '2024-11': 'Nov 2024', '2025-01': 'Jan 2025', '2025-02': 'Feb 2025',
    '2025-04': 'Apr 2025', '2025-05': 'May 2025', '2025-07': 'Jul 2025',
    '2025-08': 'Aug 2025', '2025-10': 'Oct 2025', '2025-11': 'Nov 2025',
    '2026-01': 'Jan 2026'
}

def is_trend_question(question: str) -> bool:
    """Detect if a question is asking about trends/history that would benefit from a chart"""
    trend_keywords = [
        'trend', 'over time', 'history', 'historical', 'changed', 'change',
        'direction', 'compare trend', 'show trend', 'chart', 'graph', 'plot',
        'how has', 'what happened to', 'evolve', 'movement', 'trajectory'
    ]
    return any(kw in question.lower() for kw in trend_keywords)

def fetch_trend_data(question: str) -> list:
    """Fetch ERI time-series data from Neo4j for charting"""
    driver = st.session_state.graph_rag.driver
    
    # Use LLM to extract location(s) from the question
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract location(s) from this question about cost trends. 
Return a JSON array of objects with 'county' and 'state' (2-letter abbreviation).

Examples:
- "Cost of Labor trend in San Francisco" → [{"county": "San Francisco", "state": "CA"}]
- "Compare costs in San Francisco and Los Angeles" → [{"county": "San Francisco", "state": "CA"}, {"county": "Los Angeles", "state": "CA"}]
- "How has cost changed in Maryland" → [{"county": "__STATE__", "state": "MD"}]
- "Cost trend in Boulder County Colorado" → [{"county": "Boulder", "state": "CO"}]

If a STATE is mentioned (not a specific county), use "__STATE__" as county to indicate all counties.
Return ONLY the JSON array, nothing else."""),
        ("human", "{question}")
    ])
    
    chain = extract_prompt | st.session_state.llm | StrOutputParser()
    
    try:
        raw = chain.invoke({"question": question}).strip()
        # Clean JSON
        raw = raw.replace("```json", "").replace("```", "").strip()
        locations = json.loads(raw)
    except:
        return []
    
    results = []
    
    with driver.session() as session:
        for loc in locations:
            county = loc.get("county", "")
            state = loc.get("state", "")
            
            if county == "__STATE__":
                # State-level: average across all counties in the state
                query = """
                MATCH (z:ZipCode) 
                WHERE z.state = $state AND z.eri_periods IS NOT NULL AND z.eri_labor_history IS NOT NULL
                WITH z.eri_periods AS periods, 
                     z.eri_labor_history AS labor, 
                     z.eri_living_history AS living
                LIMIT 1
                RETURN periods AS periods
                """
                period_result = session.run(query, state=state).single()
                if not period_result:
                    continue
                periods = period_result["periods"]
                
                # Get averages per period for the state
                labor_avgs = []
                living_avgs = []
                for i in range(len(periods)):
                    avg_query = f"""
                    MATCH (z:ZipCode) 
                    WHERE z.state = $state AND z.eri_labor_history IS NOT NULL AND z.eri_labor_history[{i}] > 0
                    RETURN avg(z.eri_labor_history[{i}]) AS avg_labor, avg(z.eri_living_history[{i}]) AS avg_living
                    """
                    avg_result = session.run(avg_query, state=state).single()
                    labor_avgs.append(round(avg_result["avg_labor"], 2) if avg_result["avg_labor"] else 0)
                    living_avgs.append(round(avg_result["avg_living"], 2) if avg_result["avg_living"] else 0)
                
                results.append({
                    "label": state,
                    "periods": periods,
                    "labor": labor_avgs,
                    "living": living_avgs
                })
            else:
                # County-level
                query = """
                MATCH (z:ZipCode) 
                WHERE z.county = $county AND z.state = $state AND z.eri_periods IS NOT NULL
                RETURN DISTINCT z.county AS county, z.state AS state, 
                       z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living
                LIMIT 1
                """
                record = session.run(query, county=county, state=state).single()
                if record:
                    results.append({
                        "label": f"{record['county']}, {record['state']}",
                        "periods": record["periods"],
                        "labor": record["labor"],
                        "living": record["living"]
                    })
    
    return results

def detect_metric_type(question: str) -> str:
    """Detect whether the question is about labor, living, or both"""
    q_lower = question.lower()
    has_labor = any(kw in q_lower for kw in ['labor', 'col ', 'cost of labor'])
    has_living = any(kw in q_lower for kw in ['living', 'cost of living', 'coliv'])
    
    if has_labor and not has_living:
        return "labor"
    elif has_living and not has_labor:
        return "living"
    else:
        return "both"

def create_trend_chart(trend_data: list, question: str) -> go.Figure:
    """Create an interactive Plotly chart from trend data"""
    metric_type = detect_metric_type(question)
    
    if not trend_data:
        return None
    
    fig = go.Figure()
    
    # Get readable period labels
    periods = trend_data[0]["periods"]
    x_labels = [PERIOD_LABELS.get(p, p) for p in periods]
    
    color_idx = 0
    
    for loc in trend_data:
        label = loc["label"]
        labor = loc["labor"]
        living = loc["living"]
        
        if metric_type in ["labor", "both"]:
            # Filter out 0.0 values (missing data)
            labor_filtered = [(x, v) for x, v in zip(x_labels, labor) if v > 0]
            if labor_filtered:
                x_vals, y_vals = zip(*labor_filtered)
                suffix = " - Cost of Labor" if metric_type == "both" else ""
                fig.add_trace(go.Scatter(
                    x=list(x_vals), y=list(y_vals),
                    mode='lines+markers',
                    name=f"{label}{suffix}",
                    line=dict(color=CHART_COLORS[color_idx % len(CHART_COLORS)], width=3),
                    marker=dict(size=8),
                    hovertemplate='%{x}<br>Index: %{y:.1f}<extra>' + label + '</extra>'
                ))
                color_idx += 1
        
        if metric_type in ["living", "both"]:
            living_filtered = [(x, v) for x, v in zip(x_labels, living) if v > 0]
            if living_filtered:
                x_vals, y_vals = zip(*living_filtered)
                suffix = " - Cost of Living" if metric_type == "both" else ""
                fig.add_trace(go.Scatter(
                    x=list(x_vals), y=list(y_vals),
                    mode='lines+markers',
                    name=f"{label}{suffix}",
                    line=dict(color=CHART_COLORS[color_idx % len(CHART_COLORS)], width=3, dash='dash' if metric_type == "both" else 'solid'),
                    marker=dict(size=8),
                    hovertemplate='%{x}<br>Index: %{y:.1f}<extra>' + label + '</extra>'
                ))
                color_idx += 1
    
    # Determine title
    if len(trend_data) == 1:
        loc_name = trend_data[0]["label"]
        if metric_type == "labor":
            title = f"Cost of Labor Trend — {loc_name}"
        elif metric_type == "living":
            title = f"Cost of Living Trend — {loc_name}"
        else:
            title = f"ERI Cost Trends — {loc_name}"
    else:
        if metric_type == "labor":
            title = "Cost of Labor Comparison"
        elif metric_type == "living":
            title = "Cost of Living Comparison"
        else:
            title = "ERI Cost Trends Comparison"
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1B4F5C')),
        xaxis_title="Period",
        yaxis_title="ERI Index (100 = National Average)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=30, t=60, b=80),
        yaxis=dict(gridcolor='#E8E8E8'),
        xaxis=dict(gridcolor='#E8E8E8', tickangle=-45)
    )
    
    # Add reference line at 100 (national average)
    fig.add_hline(
        y=100, line_dash="dot", line_color="#999999", line_width=1,
        annotation_text="National Avg (100)", annotation_position="bottom right",
        annotation_font_color="#999999"
    )
    
    return fig

def generate_response(question: str) -> dict:
    """Generate comprehensive response with session memory. Returns dict with 'text' and optional 'chart'."""
    
    # Step 1: Resolve follow-up questions using conversation history
    resolved_question = resolve_follow_up(question)
    
    # Step 2: Check if this is a trend question that needs a chart
    chart_fig = None
    if is_trend_question(resolved_question):
        try:
            trend_data = fetch_trend_data(resolved_question)
            if trend_data:
                chart_fig = create_trend_chart(trend_data, resolved_question)
        except Exception as e:
            pass  # Chart generation failed, still provide text answer
    
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
        return {"text": text, "chart": chart_fig}
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
        return {"text": text, "chart": chart_fig}

def process_question(question: str):
    """Process a question and update chat history"""
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Generate response
    response = generate_response(question)
    
    # Add assistant response to display (store both text and chart)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["text"],
        "chart": response.get("chart")
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
        # Render chart if present
        if message.get("chart") is not None:
            st.plotly_chart(message["chart"], use_container_width=True)

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
            if response.get("chart") is not None:
                st.plotly_chart(response["chart"], use_container_width=True)
    
    # Update session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["text"],
        "chart": response.get("chart")
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