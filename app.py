"""
MWR AI Chat - Streamlit Interface with Session Memory & Cross-Session Memory
Secured with user authentication and persistent conversation memory
"""

import streamlit as st
import os
import json
import re
import hashlib
from datetime import datetime
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
# USER CREDENTIALS — Admin-controlled access codes
# ============================================================
# Michel can add/remove/change codes here at any time.
# Format: "access_code": {"name": "Display Name", "role": "admin|user"}
# Codes are hashed for security — the raw code never appears in the file.

def hash_code(code: str) -> str:
    """Hash an access code for secure storage"""
    return hashlib.sha256(code.encode()).hexdigest()

# To add a new user:  
#   1. Pick an access code (e.g., "MWR-DG2026")
#   2. Run: python -c "import hashlib; print(hashlib.sha256('MWR-DG2026'.encode()).hexdigest())"
#   3. Add the hash below with the user's name
# To revoke access: simply delete or comment out the user's line

AUTHORIZED_USERS = {
    hash_code(os.getenv("MWR_ADMIN_CODE", "MWR-ADMIN-2026")): {"name": "Michel Pierre-Louis", "role": "admin"},
    hash_code(os.getenv("MWR_DAN_CODE", "MWR-DAN-2026")): {"name": "Dan Green", "role": "user"},
    hash_code(os.getenv("MWR_RENEE_CODE", "MWR-RENEE-2026")): {"name": "Renee", "role": "user"},
    hash_code(os.getenv("MWR_DANM_CODE", "MWR-DANM-2026")): {"name": "Dan More", "role": "user"},
}

# ============================================================
# CROSS-SESSION MEMORY — Neo4j Persistence
# ============================================================
MAX_CROSS_SESSION_SUMMARIES = 5  # Keep last 5 session summaries for context

def save_session_summary(driver, user_name: str, summary: str):
    """Save a conversation summary to Neo4j for cross-session memory"""
    try:
        with driver.session() as session:
            session.run("""
                MERGE (u:MWRUser {name: $user_name})
                CREATE (s:SessionSummary {
                    summary: $summary,
                    timestamp: datetime(),
                    date: $date
                })
                CREATE (u)-[:HAD_SESSION]->(s)
            """, user_name=user_name, summary=summary, date=datetime.now().strftime("%Y-%m-%d %H:%M"))
    except Exception as e:
        pass  # Silent fail — don't break the app if memory save fails

def load_session_summaries(driver, user_name: str) -> str:
    """Load past session summaries from Neo4j for context"""
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (u:MWRUser {name: $user_name})-[:HAD_SESSION]->(s:SessionSummary)
                RETURN s.summary AS summary, s.date AS date
                ORDER BY s.timestamp DESC
                LIMIT $limit
            """, user_name=user_name, limit=MAX_CROSS_SESSION_SUMMARIES)
            
            summaries = []
            for record in result:
                summaries.append(f"[{record['date']}] {record['summary']}")
            
            if summaries:
                summaries.reverse()  # Chronological order
                return "\n".join(summaries)
            return ""
    except:
        return ""

def generate_session_summary(llm, conversation_history: list) -> str:
    """Use LLM to generate a concise summary of the current session"""
    if len(conversation_history) < 2:
        return ""
    
    # Build conversation text
    conv_text = ""
    for msg in conversation_history[-20:]:  # Last 10 exchanges max
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]
        conv_text += f"{role}: {content}\n"
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize this MWR AI Assistant conversation in 2-3 sentences. 
Focus on: what locations/topics were discussed, key data points mentioned, and any decisions or insights.
Keep it concise — this summary will be used to provide context in future sessions.
Return ONLY the summary, nothing else."""),
            ("human", "{conversation}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"conversation": conv_text}).strip()
    except:
        return ""

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MWR AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================================
# CUSTOM CSS FOR CLEAN LOOK
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide Fork/GitHub badge in top-right but keep sidebar toggle */
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    
    /* Hide ALL Streamlit Cloud branding — bottom-right badges, logos, manage button */
    [data-testid="manage-app-button"] {display: none !important;}
    .viewerBadge_container__r5tak {display: none !important;}
    .stApp [data-testid="stBottomBlockContainer"] iframe {display: none !important;}
    ._profileContainer_gzau3_53 {display: none !important;}
    #manage-app-button {display: none !important;}
    ._container_gzau3_1 {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    .reportview-container .main footer {display: none !important;}
    .stApp iframe[height="0"] {display: none !important;}
    /* Target the colored floating button and Streamlit logo in bottom-right */
    .st-emotion-cache-czk5ss {display: none !important;}
    .st-emotion-cache-164nlkn {display: none !important;}
    .stAppDeployButton {display: none !important;}
    button[kind="manage"] {display: none !important;}
    /* Catch-all for any remaining bottom-right fixed elements */
    .stApp > iframe {display: none !important;}
    a[href*="streamlit.io"] {display: none !important;}
    /* Hide "Relaunch to update" banner at top */
    [data-testid="stAppViewBlockContainer"] > div:first-child > div[data-testid="stAlert"] {display: none !important;}
    .stAlert[data-baseweb="notification"] {display: none !important;}
    /* Nuclear option: hide ALL fixed-position elements from Streamlit in corners */
    div[class*="StatusWidget"] {display: none !important;}
    div[class*="manage"] {display: none !important;}
    button[class*="manage"] {display: none !important;}
    div[class*="deploy"] {display: none !important;}
    header[data-testid="stHeader"] {visibility: hidden !important; height: 0 !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    
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
# AUTHENTICATION GATE
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_name = ""
    st.session_state.user_role = ""

if not st.session_state.authenticated:
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <h1 style='color: #1B4F5C; font-family: Arial;'>🔒 MWR AI Assistant</h1>
        <p style='color: #666; font-size: 16px;'>Sodexo Workforce Market Intelligence</p>
        <hr style='width: 200px; margin: 20px auto;'>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        with st.form("login_form"):
            name_input = st.text_input("Your Name", placeholder="Enter your name")
            code_input = st.text_input("Access Code", type="password", placeholder="Enter your access code")
            submitted = st.form_submit_button("🔑 Sign In", use_container_width=True)
            
            if submitted:
                if name_input and code_input:
                    code_hash = hash_code(code_input)
                    if code_hash in AUTHORIZED_USERS:
                        st.session_state.authenticated = True
                        st.session_state.user_name = AUTHORIZED_USERS[code_hash]["name"]
                        st.session_state.user_role = AUTHORIZED_USERS[code_hash]["role"]
                        st.rerun()
                    else:
                        st.error("❌ Invalid access code. Contact Michel for access.")
                else:
                    st.warning("Please enter both your name and access code.")
    
    st.stop()  # Block everything below until authenticated

# ============================================================
# INITIALIZE SESSION STATE (only after authentication)
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
            st.session_state.llm_fast = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=4096
            )
            st.session_state.connected = True
            
            # Load cross-session memory for the authenticated user
            if "cross_session_loaded" not in st.session_state:
                past_context = load_session_summaries(
                    st.session_state.graph_rag.driver,
                    st.session_state.user_name
                )
                st.session_state.cross_session_context = past_context
                st.session_state.cross_session_loaded = True
                st.session_state.session_saved = False  # Track if we've saved this session
                
        except Exception as e:
            st.session_state.connected = False
            st.error(f"❌ Connection failed: {e}")

# ============================================================
# SESSION MEMORY CONFIG
# ============================================================
MAX_MEMORY_TURNS = 10  # Keep last 10 exchanges (20 messages) for context

def get_conversation_context() -> str:
    """Build conversation context string from recent history AND cross-session memory"""
    parts = []
    
    # Include cross-session memory if available
    cross_session = st.session_state.get("cross_session_context", "")
    if cross_session:
        parts.append(f"Previous Sessions:\n{cross_session}")
    
    # Include current session history
    if st.session_state.conversation_history:
        recent = st.session_state.conversation_history[-(MAX_MEMORY_TURNS * 2):]
        context_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500] if len(msg["content"]) > 500 else msg["content"]
            context_lines.append(f"{role}: {content}")
        parts.append(f"Current Session:\n" + "\n".join(context_lines))
    
    return "\n\n".join(parts) if parts else ""

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
    
    chain = resolve_prompt | st.session_state.llm_fast | StrOutputParser()
    
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
    
    chain = prompt | st.session_state.llm_fast | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip().upper()
    except Exception:
        result = "DATABASE"  # Safe default on API error
    
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
    '2026-01': 'Jan 2026', '2026-02': 'Feb 2026'
}

def _county_label(county_name: str, state_abbr: str) -> str:
    """Format county labels for charts: 'Montgomery' -> 'Montgomery County, MD'
    Avoids double-labeling for names already containing 'County' or 'City'."""
    name = str(county_name).strip()
    if any(suffix in name for suffix in ['County', 'City', 'Parish', 'Borough']):
        return f"{name}, {state_abbr}"
    return f"{name} County, {state_abbr}"

# ── City-to-County Resolution ────────────────────────────────────
# Maps common US city names to their actual county + state for accurate queries.
# This prevents "Buffalo, New York" from matching "Buffalo County, WI" instead of Erie County, NY.

CITY_TO_COUNTY = {
    # New York State
    'buffalo': ('Erie', 'NY'),
    'new york city': ('New York', 'NY'),  # Manhattan; also Kings, Queens, Bronx, Richmond
    'nyc': ('New York', 'NY'),
    'manhattan': ('New York', 'NY'),
    'brooklyn': ('Kings', 'NY'),
    'queens': ('Queens', 'NY'),
    'bronx': ('Bronx', 'NY'),
    'staten island': ('Richmond', 'NY'),
    'rochester': ('Monroe', 'NY'),
    'syracuse': ('Onondaga', 'NY'),
    'albany': ('Albany', 'NY'),
    'yonkers': ('Westchester', 'NY'),
    'white plains': ('Westchester', 'NY'),
    # California
    'los angeles': ('Los Angeles', 'CA'),
    'san francisco': ('San Francisco', 'CA'),
    'san diego': ('San Diego', 'CA'),
    'san jose': ('Santa Clara', 'CA'),
    'sacramento': ('Sacramento', 'CA'),
    'oakland': ('Alameda', 'CA'),
    'fresno': ('Fresno', 'CA'),
    'long beach': ('Los Angeles', 'CA'),
    'anaheim': ('Orange', 'CA'),
    'irvine': ('Orange', 'CA'),
    'riverside': ('Riverside', 'CA'),
    'bakersfield': ('Kern', 'CA'),
    # Texas
    'houston': ('Harris', 'TX'),
    'dallas': ('Dallas', 'TX'),
    'san antonio': ('Bexar', 'TX'),
    'austin': ('Travis', 'TX'),
    'fort worth': ('Tarrant', 'TX'),
    'el paso': ('El Paso', 'TX'),
    'plano': ('Collin', 'TX'),
    'arlington': ('Tarrant', 'TX'),
    # Florida
    'miami': ('Miami-Dade', 'FL'),
    'orlando': ('Orange', 'FL'),
    'tampa': ('Hillsborough', 'FL'),
    'jacksonville': ('Duval', 'FL'),
    'st. petersburg': ('Pinellas', 'FL'),
    'fort lauderdale': ('Broward', 'FL'),
    # Illinois
    'chicago': ('Cook', 'IL'),
    'naperville': ('DuPage', 'IL'),
    'aurora': ('Kane', 'IL'),
    # Pennsylvania
    'philadelphia': ('Philadelphia', 'PA'),
    'pittsburgh': ('Allegheny', 'PA'),
    # Arizona
    'phoenix': ('Maricopa', 'AZ'),
    'tucson': ('Pima', 'AZ'),
    'scottsdale': ('Maricopa', 'AZ'),
    'mesa': ('Maricopa', 'AZ'),
    # Other major cities
    'seattle': ('King', 'WA'),
    'portland': ('Multnomah', 'OR'),
    'denver': ('Denver', 'CO'),
    'boulder': ('Boulder', 'CO'),
    'atlanta': ('Fulton', 'GA'),
    'boston': ('Suffolk', 'MA'),
    'detroit': ('Wayne', 'MI'),
    'minneapolis': ('Hennepin', 'MN'),
    'st. louis': ('St. Louis City', 'MO'),
    'kansas city': ('Jackson', 'MO'),
    'nashville': ('Davidson', 'TN'),
    'memphis': ('Shelby', 'TN'),
    'new orleans': ('Orleans', 'LA'),
    'las vegas': ('Clark', 'NV'),
    'reno': ('Washoe', 'NV'),
    'charlotte': ('Mecklenburg', 'NC'),
    'raleigh': ('Wake', 'NC'),
    'columbus': ('Franklin', 'OH'),
    'cleveland': ('Cuyahoga', 'OH'),
    'cincinnati': ('Hamilton', 'OH'),
    'indianapolis': ('Marion', 'IN'),
    'milwaukee': ('Milwaukee', 'WI'),
    'madison': ('Dane', 'WI'),
    'salt lake city': ('Salt Lake', 'UT'),
    'baltimore': ('Baltimore', 'MD'),  # Baltimore City vs Baltimore County — city maps to city
    'washington': ('District of Columbia', 'DC'),
    'washington dc': ('District of Columbia', 'DC'),
    'dc': ('District of Columbia', 'DC'),
    'richmond': ('Richmond City', 'VA'),
    'virginia beach': ('Virginia Beach City', 'VA'),
    'norfolk': ('Norfolk City', 'VA'),
    'honolulu': ('Honolulu', 'HI'),
    'anchorage': ('Anchorage', 'AK'),
    'omaha': ('Douglas', 'NE'),
    'des moines': ('Polk', 'IA'),
    'louisville': ('Jefferson', 'KY'),
    'oklahoma city': ('Oklahoma', 'OK'),
    'tulsa': ('Tulsa', 'OK'),
    'albuquerque': ('Bernalillo', 'NM'),
    'boise': ('Ada', 'ID'),
    'charleston': ('Charleston', 'SC'),
    'birmingham': ('Jefferson', 'AL'),
    'little rock': ('Pulaski', 'AR'),
    'hartford': ('Hartford', 'CT'),
    'providence': ('Providence', 'RI'),
    'newark': ('Essex', 'NJ'),
    'jersey city': ('Hudson', 'NJ'),
    'wilmington': ('New Castle', 'DE'),
    # Maryland specific (Dan's common queries)
    'gaithersburg': ('Montgomery', 'MD'),
    'bethesda': ('Montgomery', 'MD'),
    'silver spring': ('Montgomery', 'MD'),
    'columbia': ('Howard', 'MD'),
    'annapolis': ('Anne Arundel', 'MD'),
    'frederick': ('Frederick', 'MD'),
    'towson': ('Baltimore', 'MD'),
    'hagerstown': ('Washington', 'MD'),
}

# State name to abbreviation mapping
STATE_ABBR = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
    'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
    'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
    'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
    'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY'
}

def _resolve_city_to_county(question: str) -> list:
    """Resolve city names in a question to (county, state_abbr) pairs.
    Returns list of tuples: [(county_name, state_abbr), ...]
    Uses direct dictionary lookup — scans for known city names in the text."""
    q_lower = question.lower()
    resolved = []
    used_cities = set()
    
    # Sort city names by length descending to match longer names first
    # e.g., "new york city" before "new york" (which is a state name)
    sorted_cities = sorted(CITY_TO_COUNTY.keys(), key=len, reverse=True)
    
    for city_name in sorted_cities:
        if city_name in q_lower and city_name not in used_cities:
            county, default_state = CITY_TO_COUNTY[city_name]
            
            # Check if a state is specified after the city name to confirm/disambiguate
            # Find position of city in question
            pos = q_lower.find(city_name)
            after_city = q_lower[pos + len(city_name):pos + len(city_name) + 25]  # look ahead 25 chars
            
            # Check if a state name or abbreviation follows
            state_confirmed = False
            for state_name, state_abbr in STATE_ABBR.items():
                if state_name in after_city or state_abbr.lower() in after_city.split():
                    if state_abbr == default_state:
                        state_confirmed = True
                        break
            
            # If no state specified, or state matches, use the mapping
            # (If state is specified but doesn't match, skip — wrong city)
            resolved.append((county, default_state))
            used_cities.add(city_name)
            # Replace in q_lower to prevent partial re-matching
            q_lower = q_lower.replace(city_name, '___matched___', 1)
    
    return resolved


def _fetch_city_trend_comparison(question: str) -> list:
    """Fetch trend data for city-based queries by resolving cities to counties first.
    Handles queries like 'Compare Buffalo NY to New York City NY'."""
    cities = _resolve_city_to_county(question)
    if not cities:
        return []
    
    results = []
    driver = st.session_state.graph_rag.driver
    
    for county, state in cities:
        try:
            with driver.session() as session:
                records = list(session.run("""
                    MATCH (z:ZipCode)
                    WHERE z.county = $county AND z.state = $state AND z.eri_periods IS NOT NULL
                    RETURN DISTINCT z.county AS county, z.state AS state,
                           z.eri_periods AS periods, z.eri_labor_history AS labor,
                           z.eri_living_history AS living
                    LIMIT 1
                """, county=county, state=state))
                
                if records:
                    r = dict(records[0])
                    results.append({
                        "label": _county_label(r['county'], r['state']),
                        "county": r['county'],
                        "state": r['state'],
                        "periods": r['periods'],
                        "labor": r['labor'],
                        "living": r['living']
                    })
        except Exception:
            continue
    
    return results if len(results) >= 1 else []

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
- "compare the trend of cost of living between Maryland and New York" → LINE_TREND (trend + cost = LINE_TREND, even with "compare")
- "compare cost of labor trend NY vs CA" → LINE_TREND (trend comparison = time-series overlay)
- "show me the cost of living trend for Maryland and Virginia" → LINE_TREND
- "compare cost of labor NY vs MD" → BAR_COMPARE (no "trend" word = current snapshot comparison)
- "compare cost of living in San Francisco vs Los Angeles vs San Diego" → BAR_COMPARE
- "what is California's risk?" → NONE
- "unemployment trend in Montgomery County" → NONE (no time-series for unemployment)
- "unemployment rate in Boulder" → NONE
- "education breakdown in SF" → NONE
- "how many critical ZIP codes" → NONE
- "what are the latest minimum wage news" → NONE

IMPORTANT: If the question mentions BOTH "compare" AND "trend" for cost of labor or cost of living → always return LINE_TREND (not BAR_COMPARE).
The word "trend" is the deciding factor — it means the user wants to see change over time, not just current values.

Return ONLY one word: LINE_TREND, BAR_COMPARE, HBAR_RANK, or NONE"""),
        ("human", "{question}")
    ])
    
    chain = chart_prompt | st.session_state.llm_fast | StrOutputParser()
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
    """Fetch ERI time-series data from Neo4j for line charts and bar comparisons"""
    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate a Neo4j Cypher query to fetch ERI time-series data.
The ZipCode nodes have these properties:
- eri_periods: list of period labels like ['2024-05', '2024-07', ...]
- eri_labor_history: list of Cost of Labor values per period
- eri_living_history: list of Cost of Living values per period
- county: county name (e.g., 'San Francisco')
- state: state abbreviation (e.g., 'CA')

State nodes have: name (full name like 'New York'), abbr (like 'NY')
Relationships: (ZipCode)-[:IN_STATE]->(State)

CRITICAL RULES:
1. Always use DISTINCT to avoid duplicate rows
2. Always include: AND z.eri_periods IS NOT NULL
3. For a SINGLE county query, use LIMIT 1
4. For state-level queries, return max 5 representative counties (use LIMIT 5)
5. Return fields AS: county, state, periods, labor, living

IMPORTANT — STATE vs COUNTY detection:
- If the user mentions US STATES (like "New York", "California", "Texas", "Maryland"), 
  pick ONE representative county per state (the most populated or capital county).
  Use the state abbreviation to filter: z.state = 'NY', z.state = 'CA', etc.
- If the user mentions COUNTIES or CITIES (like "San Francisco", "Los Angeles", "Boulder"), 
  use the county name directly.
- "Compare New York and California" means compare NY STATE vs CA STATE, 
  NOT counties named "New York" within NY state.

Examples:
- "trend in San Francisco" → 
  MATCH (z:ZipCode) WHERE z.county = 'San Francisco' AND z.state = 'CA' AND z.eri_periods IS NOT NULL 
  RETURN DISTINCT z.county AS county, z.state AS state, z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living LIMIT 1

- "compare SF and LA" → 
  MATCH (z:ZipCode) WHERE z.county IN ['San Francisco', 'Los Angeles'] AND z.state = 'CA' AND z.eri_periods IS NOT NULL 
  WITH DISTINCT z.county AS county, z.state AS state, z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living 
  RETURN county, state, periods, labor, living

- "compare New York and California" → 
  MATCH (z:ZipCode) WHERE z.state IN ['NY', 'CA'] AND z.eri_periods IS NOT NULL 
  WITH z.state AS state, z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living 
  WITH state, head(collect(periods)) AS periods, head(collect(labor)) AS labor, head(collect(living)) AS living 
  MATCH (s:State {abbr: state}) 
  RETURN s.name AS county, state, periods, labor, living

- "compare Maryland and Virginia" → 
  MATCH (z:ZipCode) WHERE z.state IN ['MD', 'VA'] AND z.eri_periods IS NOT NULL 
  WITH z.state AS state, z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living 
  WITH state, head(collect(periods)) AS periods, head(collect(labor)) AS labor, head(collect(living)) AS living 
  MATCH (s:State {abbr: state}) 
  RETURN s.name AS county, state, periods, labor, living

Return ONLY the Cypher query, nothing else."""),
        ("human", "{question}")
    ])
    
    chain = cypher_prompt | st.session_state.llm_fast | StrOutputParser()
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
                results.append({"label": _county_label(r['county'], r['state']), "periods": r["periods"], "labor": r["labor"], "living": r["living"]})
    return results[:8]


def _fetch_state_bar_compare(question: str) -> list:
    """Fallback for BAR_COMPARE: fetch current cost_of_labor/cost_of_living averages per state.
    Returns data in the same format as fetch_trend_data so create_bar_chart can consume it."""
    
    # Common state name to abbreviation mapping
    state_map = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
        'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
        'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
        'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
        'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
        'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
        'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
        'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
        'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN',
        'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
        'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
    }
    
    # Extract state names from question
    q_lower = question.lower()
    found_states = []
    for state_name, abbr in sorted(state_map.items(), key=lambda x: -len(x[0])):
        if state_name in q_lower:
            found_states.append((state_name, abbr))
    
    if len(found_states) < 2:
        return []
    
    # Limit to first 5 states
    found_states = found_states[:5]
    abbrs = [s[1] for s in found_states]
    
    # Query current averages per state
    cypher = """
    MATCH (z:ZipCode) WHERE z.state IN $states AND z.cost_of_labor IS NOT NULL AND z.cost_of_labor > 0
    WITH z.state AS st, avg(z.cost_of_labor) AS labor_avg, avg(z.cost_of_living) AS living_avg
    MATCH (s:State {abbr: st})
    RETURN s.name AS name, st AS state, labor_avg, living_avg
    """
    
    try:
        driver = st.session_state.graph_rag.driver
        with driver.session() as session:
            records = [dict(r) for r in session.run(cypher, states=abbrs)]
    except:
        return []
    
    if not records:
        return []
    
    # Convert to trend_data format (with single-element lists for bar chart compatibility)
    results = []
    for r in records:
        results.append({
            "label": r["name"],
            "periods": ["Current"],
            "labor": [round(r["labor_avg"], 1)],
            "living": [round(r["living_avg"], 1)]
        })
    
    return results


def _fetch_state_trend_compare(question: str) -> list:
    """Fallback for LINE_TREND: compute state-level average trends for state-vs-state comparisons."""
    
    # Common state name to abbreviation mapping
    state_map = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
        'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
        'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
        'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
        'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
        'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
        'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
        'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
        'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN',
        'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
        'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
    }
    
    # Extract state names from question
    q_lower = question.lower()
    found_states = []
    for state_name, abbr in sorted(state_map.items(), key=lambda x: -len(x[0])):
        if state_name in q_lower and abbr not in [s[1] for s in found_states]:
            found_states.append((state_name, abbr))
    
    if not found_states:
        return []
    
    driver = st.session_state.graph_rag.driver
    results = []
    
    for state_name, abbr in found_states[:5]:
        try:
            with driver.session() as session:
                records = [dict(r) for r in session.run("""
                    MATCH (z:ZipCode) WHERE z.state = $state AND z.eri_periods IS NOT NULL
                    WITH z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living
                    WITH head(collect(periods)) AS periods, collect(labor) AS all_labor, collect(living) AS all_living
                    WITH periods, 
                         [i IN range(0, size(periods)-1) | 
                            round(reduce(s=0.0, arr IN all_labor | s + arr[i]) / size(all_labor), 1)] AS avg_labor,
                         [i IN range(0, size(periods)-1) | 
                            round(reduce(s=0.0, arr IN all_living | s + arr[i]) / size(all_living), 1)] AS avg_living
                    RETURN periods, avg_labor, avg_living
                """, state=abbr)]
            if records and records[0].get("periods"):
                r = records[0]
                results.append({
                    "label": state_name.title(),
                    "periods": r["periods"],
                    "labor": r["avg_labor"],
                    "living": r["avg_living"]
                })
        except:
            # Fallback: use representative county
            try:
                with driver.session() as session:
                    records = [dict(r) for r in session.run("""
                        MATCH (z:ZipCode) WHERE z.state = $state AND z.eri_periods IS NOT NULL
                        WITH z.county AS county, count(z) AS cnt
                        ORDER BY cnt DESC LIMIT 1
                        WITH county
                        MATCH (z2:ZipCode) WHERE z2.county = county AND z2.state = $state AND z2.eri_periods IS NOT NULL
                        RETURN z2.eri_periods AS periods, z2.eri_labor_history AS labor, z2.eri_living_history AS living
                        LIMIT 1
                    """, state=abbr)]
                if records and records[0].get("periods"):
                    r = records[0]
                    results.append({
                        "label": state_name.title(),
                        "periods": r["periods"],
                        "labor": r["labor"],
                        "living": r["living"]
                    })
            except:
                continue
    
    return results


def _fetch_county_trend_fallback(question: str) -> list:
    """Fallback for LINE_TREND when LLM Cypher fails for county/city queries.
    Uses fuzzy matching on common city/county names."""
    
    # Common city-to-county mappings and direct county names
    county_keywords = {
        'san francisco': ('San Francisco', 'CA'),
        'los angeles': ('Los Angeles', 'CA'),
        'san diego': ('San Diego', 'CA'),
        'boulder': ('Boulder', 'CO'),
        'denver': ('Denver', 'CO'),
        'montgomery county': ('Montgomery', 'MD'),
        'fairfax': ('Fairfax', 'VA'),
        'arlington': ('Arlington', 'VA'),
        'howard': ('Howard', 'MD'),
        'miami': ('Miami-Dade', 'FL'),
        'cook county': ('Cook', 'IL'),
        'chicago': ('Cook', 'IL'),
        'harris county': ('Harris', 'TX'),
        'houston': ('Harris', 'TX'),
        'king county': ('King', 'WA'),
        'seattle': ('King', 'WA'),
        'new york county': ('New York', 'NY'),
        'manhattan': ('New York', 'NY'),
        'brooklyn': ('Kings', 'NY'),
        'queens': ('Queens', 'NY'),
        'bronx': ('Bronx', 'NY'),
        'dallas': ('Dallas', 'TX'),
        'austin': ('Travis', 'TX'),
        'phoenix': ('Maricopa', 'AZ'),
        'westchester': ('Westchester', 'NY'),
        'nassau': ('Nassau', 'NY'),
        'suffolk county': ('Suffolk', 'NY'),
        'bergen': ('Bergen', 'NJ'),
        'essex': ('Essex', 'NJ'),
        'anne arundel': ('Anne Arundel', 'MD'),
        'prince george': ('Prince Georges', 'MD'),
        'alameda': ('Alameda', 'CA'),
    }
    
    q_lower = question.lower()
    found_counties = []
    
    for keyword, (county, state) in sorted(county_keywords.items(), key=lambda x: -len(x[0])):
        if keyword in q_lower:
            found_counties.append((county, state))
    
    if not found_counties:
        # Try a broader approach: search for any word that might be a county
        # This is a last resort — query Neo4j directly with a CONTAINS match
        words = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', question)
        for word in words:
            if word.lower() not in ['cost', 'labor', 'living', 'trend', 'compare', 'show', 'what', 'how', 'the']:
                try:
                    driver = st.session_state.graph_rag.driver
                    with driver.session() as session:
                        result = session.run("""
                            MATCH (z:ZipCode) WHERE z.county = $county AND z.eri_periods IS NOT NULL
                            RETURN DISTINCT z.county AS county, z.state AS state LIMIT 1
                        """, county=word)
                        record = result.single()
                        if record:
                            found_counties.append((record['county'], record['state']))
                            break
                except:
                    continue
    
    if not found_counties:
        return []
    
    results = []
    driver = st.session_state.graph_rag.driver
    
    for county, state in found_counties[:5]:
        cypher = """
        MATCH (z:ZipCode) WHERE z.county = $county AND z.state = $state AND z.eri_periods IS NOT NULL
        RETURN z.county AS county, z.state AS state, z.eri_periods AS periods, 
               z.eri_labor_history AS labor, z.eri_living_history AS living
        LIMIT 1
        """
        try:
            with driver.session() as session:
                records = [dict(r) for r in session.run(cypher, county=county, state=state)]
            if records:
                r = records[0]
                if r.get("periods") and r.get("labor"):
                    results.append({
                        "label": _county_label(r["county"], r["state"]),
                        "periods": r["periods"],
                        "labor": r["labor"],
                        "living": r["living"]
                    })
        except:
            continue
    
    return results


def _fetch_county_vs_state_trend(question: str) -> list:
    """Handle 'county vs state' trend comparisons.
    Detects when the question asks to compare specific counties against their parent state.
    Returns each county's actual trend data + the state's average trend data."""
    
    state_map = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
        'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
        'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
        'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
        'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
        'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
        'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
        'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
        'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN',
        'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
        'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
    }
    
    # County-to-Neo4j-name lookup — names must match EXACTLY what is in Neo4j
    county_keywords = {
        'san francisco': ('San Francisco', 'CA'), 'los angeles': ('Los Angeles', 'CA'),
        'san diego': ('San Diego', 'CA'), 'boulder': ('Boulder', 'CO'),
        'denver': ('Denver', 'CO'), 'montgomery county': ('Montgomery', 'MD'),
        'montgomery': ('Montgomery', 'MD'), 'fairfax': ('Fairfax', 'VA'),
        'arlington': ('Arlington', 'VA'), 'howard county': ('Howard', 'MD'),
        'howard': ('Howard', 'MD'), 'miami': ('Miami-Dade', 'FL'),
        'cook county': ('Cook', 'IL'), 'chicago': ('Cook', 'IL'),
        'harris county': ('Harris', 'TX'), 'houston': ('Harris', 'TX'),
        'king county': ('King', 'WA'), 'seattle': ('King', 'WA'),
        'manhattan': ('New York', 'NY'), 'brooklyn': ('Kings', 'NY'),
        'queens': ('Queens', 'NY'), 'bronx': ('Bronx', 'NY'),
        'dallas': ('Dallas', 'TX'), 'austin': ('Travis', 'TX'),
        'phoenix': ('Maricopa', 'AZ'), 'westchester': ('Westchester', 'NY'),
        'nassau': ('Nassau', 'NY'), 'bergen': ('Bergen', 'NJ'),
        'essex': ('Essex', 'NJ'), 'anne arundel': ('Anne Arundel', 'MD'),
        'prince george': ('Prince Georges', 'MD'),
        "prince george's": ('Prince Georges', 'MD'),
        'prince georges': ('Prince Georges', 'MD'),
        'alameda': ('Alameda', 'CA'),
        'baltimore county': ('Baltimore', 'MD'), 'baltimore city': ('Baltimore City', 'MD'),
        'baltimore': ('Baltimore', 'MD'), 'frederick': ('Frederick', 'MD'),
        'charles': ('Charles', 'MD'), 'harford': ('Harford', 'MD'),
        'saint marys': ('Saint Marys', 'MD'), "st. mary's": ('Saint Marys', 'MD'),
        'calvert': ('Calvert', 'MD'), 'cecil': ('Cecil', 'MD'),
    }
    
    q_lower = question.lower()
    
    # Step 1: Find ALL county mentions (support multiple)
    found_counties = []
    used_keywords = set()
    for keyword, (county, state) in sorted(county_keywords.items(), key=lambda x: -len(x[0])):
        if keyword in q_lower and keyword not in used_keywords:
            county_key = f"{county}|{state}"
            if county_key not in [f"{c}|{s}" for c, s in found_counties]:
                found_counties.append((county, state))
                used_keywords.add(keyword)
                for other_kw, (other_c, other_s) in county_keywords.items():
                    if other_c == county and other_s == state:
                        used_keywords.add(other_kw)
    
    # Step 2: Find state mentions
    found_state = None
    for state_name, abbr in sorted(state_map.items(), key=lambda x: -len(x[0])):
        if state_name in q_lower:
            found_state = (state_name, abbr)
            break
    
    if not found_counties or not found_state:
        return []
    
    state_full_name, state_abbr = found_state
    
    found_counties = [(c, s) for c, s in found_counties if c.lower() != state_full_name.lower()]
    if not found_counties:
        return []
    
    driver = st.session_state.graph_rag.driver
    results = []
    
    for county_name, county_state_abbr in found_counties[:4]:
        try:
            with driver.session() as session:
                records = [dict(r) for r in session.run("""
                    MATCH (z:ZipCode) WHERE z.county = $county AND z.state = $state AND z.eri_periods IS NOT NULL
                    RETURN z.county AS county, z.state AS state, z.eri_periods AS periods,
                           z.eri_labor_history AS labor, z.eri_living_history AS living
                    LIMIT 1
                """, county=county_name, state=county_state_abbr)]
            if records and records[0].get("periods"):
                r = records[0]
                results.append({
                    "label": _county_label(r["county"], r["state"]),
                    "periods": r["periods"],
                    "labor": r["labor"],
                    "living": r["living"]
                })
        except:
            continue
    
    try:
        with driver.session() as session:
            records = [dict(r) for r in session.run("""
                MATCH (z:ZipCode) WHERE z.state = $state AND z.eri_periods IS NOT NULL
                WITH z.eri_periods AS periods, z.eri_labor_history AS labor, z.eri_living_history AS living
                WITH head(collect(periods)) AS periods, collect(labor) AS all_labor, collect(living) AS all_living
                WITH periods, 
                     [i IN range(0, size(periods)-1) | 
                        round(reduce(s=0.0, arr IN all_labor | s + arr[i]) / size(all_labor), 1)] AS avg_labor,
                     [i IN range(0, size(periods)-1) | 
                        round(reduce(s=0.0, arr IN all_living | s + arr[i]) / size(all_living), 1)] AS avg_living
                RETURN periods, avg_labor, avg_living
            """, state=state_abbr)]
        if records and records[0].get("periods"):
            r = records[0]
            results.append({
                "label": f"{state_full_name.title()} (State Avg)",
                "periods": r["periods"],
                "labor": r["avg_labor"],
                "living": r["avg_living"]
            })
    except:
        try:
            with driver.session() as session:
                records = [dict(r) for r in session.run("""
                    MATCH (z:ZipCode) WHERE z.state = $state AND z.eri_periods IS NOT NULL
                    WITH z.county AS county, count(z) AS cnt
                    ORDER BY cnt DESC LIMIT 1
                    WITH county
                    MATCH (z2:ZipCode) WHERE z2.county = county AND z2.state = $state AND z2.eri_periods IS NOT NULL
                    RETURN z2.eri_periods AS periods, z2.eri_labor_history AS labor, z2.eri_living_history AS living
                    LIMIT 1
                """, state=state_abbr)]
            if records and records[0].get("periods"):
                r = records[0]
                results.append({
                    "label": f"{state_full_name.title()} (State Rep.)",
                    "periods": r["periods"],
                    "labor": r["labor"],
                    "living": r["living"]
                })
        except:
            pass
    
    if len(results) >= 2:
        return results
    return []


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
  pct_bachelors, pct_graduate, pct_no_diploma, total_population,
  cbsa_code (CBSA/Division number or ZIP if rural), cbsa_classification ('Division'=major market, 'CBSA'=mid-size, 'Non CBSA'=rural),
  population_density_sq_mi (people per square mile), preferred_city, fips
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
            label = str(r["label"])
            # Add "County" to county-level labels like "Montgomery, MD" -> "Montgomery County, MD"
            import re
            county_match = re.match(r'^([A-Za-z\s\.\'-]+),\s*([A-Z]{2})$', label)
            if county_match:
                county_name, state_abbr = county_match.groups()
                label = _county_label(county_name.strip(), state_abbr)
            results.append({"label": label, "value": round(float(r["value"]), 1), "tier": r.get("tier")})
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
            marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(labels))],
            text=[f"{v:.1f}" for v in labor_values], textposition='outside',
            textfont=dict(size=14, color='#333333', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>Cost of Labor: %{y:.1f}<br>vs National Avg: %{customdata:+.1f}<extra></extra>',
            customdata=[v - 100 for v in labor_values]))
    if metric_type in ["living", "both"]:
        fig.add_trace(go.Bar(x=labels, y=living_values, name="Cost of Living",
            marker_color=[CHART_COLORS[(i + len(labels)) % len(CHART_COLORS)] for i in range(len(labels))],
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
            
            # Reverse guard: LINE_TREND chosen but NO trend word in question?
            # If user is comparing locations without asking for trends, use BAR_COMPARE instead
            if chart_type == "LINE_TREND":
                q_check = resolved_question.lower()
                has_trend_word = any(kw in q_check for kw in [
                    'trend', 'over time', 'historical', 'history', 'trajectory', 
                    'changed over', 'evolve', 'how has', 'movement'
                ])
                has_compare_signal = any(kw in q_check for kw in [
                    'compare', 'comparison', 'versus', ' vs ', ' vs.', 'between'
                ])
                if not has_trend_word and has_compare_signal:
                    chart_type = "BAR_COMPARE"
                    chart_error = f"type=BAR_COMPARE (downgraded from LINE_TREND — no trend word, comparison detected)"
            
            # Also catch when classifier correctly returns NONE for non-ERI trend requests
            # (e.g. "unemployment trend" — classifier learned to return NONE, but user still expects feedback)
            if chart_type == "NONE" and not chart_failed_reason:
                q_check = resolved_question.lower()
                has_trend_word = any(kw in q_check for kw in ['trend', 'over time', 'historical', 'history', 'trajectory', 'evolve'])
                has_non_eri_metric = any(kw in q_check for kw in [
                    'unemployment', 'risk', 'education', 'income', 'population',
                    'wage', 'diploma', 'workforce'
                ])
                has_eri_keywords = any(kw in q_check for kw in [
                    'cost of labor', 'cost of living', 'eri', 'col ', 'coliv',
                    'labor cost', 'living cost'
                ])
                if has_trend_word and has_non_eri_metric and not has_eri_keywords:
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
            
            # Safety override: "compare trend" should be LINE_TREND, not BAR_COMPARE
            # ONLY fires when the word "trend" (or equivalent) is explicitly in the question
            if chart_type == "BAR_COMPARE":
                q_check = resolved_question.lower()
                has_trend_word = any(kw in q_check for kw in ['trend', 'over time', 'historical', 'history', 'trajectory', 'changed over'])
                if has_trend_word:
                    chart_type = "LINE_TREND"
                    chart_error = f"type=LINE_TREND (overridden from BAR_COMPARE — trend detected)"
            
            if chart_type == "LINE_TREND":
                # Detect pure state-vs-state comparison and route directly
                q_check_cmp = resolved_question.lower()
                state_names = ['alabama','alaska','arizona','arkansas','california','colorado','connecticut',
                    'delaware','florida','georgia','hawaii','idaho','illinois','indiana','iowa','kansas',
                    'kentucky','louisiana','maine','maryland','massachusetts','michigan','minnesota',
                    'mississippi','missouri','montana','nebraska','nevada','new hampshire','new jersey',
                    'new mexico','new york','north carolina','north dakota','ohio','oklahoma','oregon',
                    'pennsylvania','rhode island','south carolina','south dakota','tennessee','texas',
                    'utah','vermont','virginia','washington','west virginia','wisconsin','wyoming']
                county_words = ['county', 'borough', 'parish', 'montgomery', 'baltimore', 'fairfax',
                    'arlington', 'boulder', 'howard', 'san francisco', 'los angeles', 'cook',
                    'harris', 'king county', 'manhattan', 'brooklyn', 'queens', 'bronx',
                    'dallas', 'austin', 'phoenix', 'westchester', 'nassau', 'bergen',
                    'anne arundel', 'prince george', 'frederick', 'charles', 'harford']
                matched_states = [s for s in state_names if s in q_check_cmp]
                has_county_word = any(kw in q_check_cmp for kw in county_words)
                is_pure_state_compare = len(matched_states) >= 2 and not has_county_word
                
                data = None
                
                # Priority 0: Try city-to-county resolution FIRST
                # This handles "Buffalo, New York" → Erie County, NY correctly
                # and prevents wrong matches like Buffalo County, WI
                city_resolved = _resolve_city_to_county(resolved_question)
                if city_resolved and not has_county_word:
                    data = _fetch_city_trend_comparison(resolved_question)
                    if data:
                        chart_error += " (city→county resolved)"
                
                if not data and is_pure_state_compare:
                    # Skip LLM Cypher entirely — go straight to state average computation
                    data = _fetch_state_trend_compare(resolved_question)
                
                if not data:
                    data = fetch_trend_data(resolved_question)
                    # Validate: for comparison queries, LLM Cypher must return 2+ results
                    is_comparison = any(kw in q_check_cmp for kw in ['compare', ' vs ', ' vs.', 'versus', ' and '])
                    if data and is_comparison and len(data) < 2:
                        data = []  # Force fallback — LLM returned partial results
                if not data:
                    # Fallback 0: try county-vs-state comparison (e.g., "Montgomery County vs Maryland")
                    data = _fetch_county_vs_state_trend(resolved_question)
                if not data:
                    # Fallback 1: try county-level trend FIRST (handles "Montgomery County vs Westchester County")
                    data = _fetch_county_trend_fallback(resolved_question)
                if not data:
                    # Fallback 2: try state-level trend using representative counties
                    data = _fetch_state_trend_compare(resolved_question)
                if data:
                    chart_data = {"chart_type": chart_type, "data": data, "question": resolved_question}
                    chart_error += f", OK: {len(data)} locations"
                else:
                    chart_error += ", no trend data found"
                    chart_failed_reason = chart_failed_reason or "no_data"
            
            elif chart_type == "BAR_COMPARE":
                # Try city-to-county resolution first for bar comparisons too
                city_resolved = _resolve_city_to_county(resolved_question)
                data = None
                if city_resolved:
                    data = _fetch_city_trend_comparison(resolved_question)
                if not data:
                    data = fetch_trend_data(resolved_question)
                if not data:
                    # Fallback: try state-level comparison using current values instead of time-series
                    data = _fetch_state_bar_compare(resolved_question)
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
    
    # If chart data was successfully retrieved, inject it into context so the LLM narrative matches
    if chart_data is not None:
        cd = chart_data
        if cd["chart_type"] == "LINE_TREND" and cd["data"]:
            trend_summary_parts = []
            metric = detect_metric_type(resolved_question)
            for loc in cd["data"]:
                vals = loc.get("living" if metric == "living" else "labor", [])
                if vals:
                    valid = [v for v in vals if v > 0]
                    if valid:
                        trend_summary_parts.append(
                            f"- {loc['label']}: ERI {'Cost of Living' if metric == 'living' else 'Cost of Labor'} "
                            f"ranges from {min(valid):.1f} to {max(valid):.1f} over {len(valid)} periods "
                            f"(May 2024 to Jan 2026). Latest: {valid[-1]:.1f}, National avg = 100."
                        )
            if trend_summary_parts:
                context_parts.append(
                    "**Chart Data Successfully Retrieved (trend comparison):**\n"
                    "The interactive chart below shows the full time-series. Key data points:\n"
                    + "\n".join(trend_summary_parts)
                    + "\n\nIMPORTANT: The chart IS displaying successfully. Do NOT say data is missing or there was an error. "
                    "Provide business analysis of the trends shown."
                )
    
    # Get database results if needed
    if q_type in ["DATABASE", "BOTH"]:
        # ── Direct CBSA/population density lookup for ZIP codes ──
        # Bypass LLM Cypher for CBSA questions to ensure reliable results
        import re
        cbsa_zip_match = re.search(r'\b(\d{5})\b', resolved_question)
        is_cbsa_question = cbsa_zip_match and any(kw in resolved_question.lower() for kw in [
            'cbsa', 'classification', 'market class', 'division', 'rural', 
            'population density', 'density', 'major market', 'metro'
        ])
        
        if is_cbsa_question:
            try:
                zip_code = cbsa_zip_match.group(1)
                driver = st.session_state.graph_rag.driver
                with driver.session() as session:
                    result = session.run("""
                        MATCH (z:ZipCode {zip: $zip})
                        RETURN z.zip AS zip, z.county AS county, z.state AS state,
                               z.cbsa_classification AS cbsa_classification,
                               z.cbsa_code AS cbsa_code,
                               z.population_density_sq_mi AS population_density,
                               z.preferred_city AS preferred_city
                    """, zip=zip_code)
                    record = result.single()
                    if record and record["cbsa_classification"]:
                        cbsa_info = (
                            f"**CBSA Data for ZIP {zip_code}:**\n"
                            f"- County: {record['county']}, {record['state']}\n"
                            f"- CBSA Classification: {record['cbsa_classification']}\n"
                            f"- CBSA Code: {record['cbsa_code']}\n"
                            f"- Population Density: {record['population_density']} people per sq mile\n"
                            f"- Preferred City: {record['preferred_city']}"
                        )
                        context_parts.append(cbsa_info)
                    else:
                        # Fall through to normal LLM query
                        is_cbsa_question = False
            except Exception:
                is_cbsa_question = False
        
        if not is_cbsa_question:
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
- CBSA Classification: Each ZIP is classified as Division (major market, 1M+ population), CBSA (mid-size market), or Non CBSA (rural area)
- Population Density: People per square mile at the ZIP code level — helps assess urban vs rural labor supply dynamics
- Preferred City: The major city associated with each ZIP code area

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
    
    # Generate response with error handling
    try:
        response = generate_response(question)
    except Exception as e:
        error_msg = str(e)
        if "overloaded" in error_msg.lower() or "529" in error_msg or "OverloadedError" in error_msg:
            response = {"text": "⏳ The AI service is temporarily busy. Please try your question again in a few seconds.", "chart_data": None}
        elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
            response = {"text": "⏳ Rate limit reached. Please wait a moment and try again.", "chart_data": None}
        else:
            response = {"text": f"⚠️ Something went wrong. Please try again. If the issue persists, try refreshing the page.", "chart_data": None}
    
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
# Title with user name and memory indicator
num_turns = len(st.session_state.conversation_history) // 2
has_cross_memory = bool(st.session_state.get("cross_session_context", ""))
memory_parts = []
if num_turns > 0:
    memory_parts.append(f"🧠 {num_turns} exchanges")
if has_cross_memory:
    memory_parts.append("📚 Past sessions loaded")
memory_html = " · ".join(memory_parts)

if memory_parts:
    st.markdown(
        f'<p class="main-title">🤖 MWR AI Assistant — Welcome, {st.session_state.user_name} '
        f'<span class="memory-badge">{memory_html}</span></p>',
        unsafe_allow_html=True
    )
else:
    st.markdown(f'<p class="main-title">🤖 MWR AI Assistant — Welcome, {st.session_state.user_name}</p>', unsafe_allow_html=True)

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
            try:
                response = generate_response(prompt)
            except Exception as e:
                error_msg = str(e)
                if "overloaded" in error_msg.lower() or "529" in error_msg or "OverloadedError" in error_msg:
                    response = {"text": "⏳ The AI service is temporarily busy. Please try your question again in a few seconds.", "chart_data": None}
                elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
                    response = {"text": "⏳ Rate limit reached. Please wait a moment and try again.", "chart_data": None}
                else:
                    response = {"text": "⚠️ Something went wrong. Please try again. If the issue persists, try refreshing the page.", "chart_data": None}
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
    
    # AUTO-SAVE: Save session memory every 3 exchanges (silent, no user action needed)
    num_exchanges = len(st.session_state.conversation_history) // 2
    if num_exchanges >= 3 and num_exchanges % 3 == 0 and not st.session_state.get("session_saved", False):
        try:
            summary = generate_session_summary(
                st.session_state.llm,
                st.session_state.conversation_history
            )
            if summary:
                save_session_summary(
                    st.session_state.graph_rag.driver,
                    st.session_state.user_name,
                    summary
                )
                st.session_state.session_saved = True
        except:
            pass  # Silent fail — never interrupt the user
    
    # Rerun to update memory badge
    st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.user_name}")
    st.caption(f"Role: {st.session_state.user_role.title()}")
    
    st.divider()
    
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
    - CBSA market classification 🏙️
    - Population density
    - Market profiles for bidding
    - Latest MW news
    - **Follow-up questions!** 🧠
    
    **Data Sources:**
    - Bureau of Labor Statistics (BLS)
    - U.S. Census Bureau (ACS)
    - ERI Economic Research
    - Minimum Wage Legislation
    - Web Search (latest news)
    
    **Memory:**
    The assistant remembers your 
    conversation in this session AND
    across sessions! 🧠📚
    """)
    
    st.divider()
    
    # Memory stats
    turns = len(st.session_state.conversation_history) // 2
    st.markdown(f"🧠 **Session Memory:** {turns} exchanges")
    st.markdown(f"📊 **Max Memory:** {MAX_MEMORY_TURNS} exchanges")
    if st.session_state.get("cross_session_context"):
        st.markdown("📚 **Past Sessions:** Loaded ✅")
    else:
        st.markdown("📚 **Past Sessions:** None yet")
    
    st.divider()
    
    # Auto-save indicator for all users
    if st.session_state.get("session_saved"):
        st.caption("✅ Session auto-saved")
    elif turns >= 2:
        st.caption("💾 Session will auto-save at 3 exchanges")
    
    # Manual save button — admin only
    if st.session_state.user_role == "admin":
        if turns >= 2 and not st.session_state.get("session_saved", False):
            if st.button("💾 Save Session Memory", use_container_width=True):
                with st.spinner("Saving..."):
                    summary = generate_session_summary(
                        st.session_state.llm,
                        st.session_state.conversation_history
                    )
                    if summary:
                        save_session_summary(
                            st.session_state.graph_rag.driver,
                            st.session_state.user_name,
                            summary
                        )
                        st.session_state.session_saved = True
                        st.success("✅ Session saved! I'll remember this next time.")
                    else:
                        st.warning("Not enough conversation to save.")
        elif st.session_state.get("session_saved"):
            st.success("✅ Session saved")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        # Auto-save before clearing if there's meaningful conversation
        if turns >= 2 and not st.session_state.get("session_saved", False):
            summary = generate_session_summary(
                st.session_state.llm,
                st.session_state.conversation_history
            )
            if summary:
                save_session_summary(
                    st.session_state.graph_rag.driver,
                    st.session_state.user_name,
                    summary
                )
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.pending_question = None
        st.session_state.session_saved = False
        st.rerun()
    
    if st.button("🚪 Logout", use_container_width=True):
        # Auto-save before logout if there's meaningful conversation
        if turns >= 2 and not st.session_state.get("session_saved", False):
            summary = generate_session_summary(
                st.session_state.llm,
                st.session_state.conversation_history
            )
            if summary:
                save_session_summary(
                    st.session_state.graph_rag.driver,
                    st.session_state.user_name,
                    summary
                )
        # Clear everything
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Admin panel — only visible to admin users
    if st.session_state.user_role == "admin":
        st.divider()
        st.markdown("### 🔧 Admin Panel")
        
        # Show active users
        if st.button("👥 View Active Users", use_container_width=True):
            try:
                driver = st.session_state.graph_rag.driver
                with driver.session() as sess:
                    result = sess.run("""
                        MATCH (u:MWRUser)-[:HAD_SESSION]->(s:SessionSummary)
                        RETURN u.name AS user, count(s) AS sessions, 
                               max(s.date) AS last_active
                        ORDER BY last_active DESC
                    """)
                    for record in result:
                        st.caption(f"**{record['user']}** — {record['sessions']} sessions, last: {record['last_active']}")
            except:
                st.caption("No session data yet.")
        
        # Debug info
        st.divider()
        st.markdown("**🔧 Debug (dev only)**")
        st.caption(st.session_state.get("_chart_debug", "—"))
        st.caption(st.session_state.get("_rank_debug", "—"))
        rank_cypher = st.session_state.get("_rank_cypher", "")
        if rank_cypher:
            st.code(rank_cypher[:200], language="cypher")