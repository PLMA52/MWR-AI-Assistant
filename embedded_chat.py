"""
MWR Embedded AI Chat - Compact version for Power BI embedding
Connects to the same Neo4j database and uses session memory
"""

import streamlit as st
import os
from dotenv import load_dotenv
from neo4j_graphrag import MWRGraphRAG
from tavily import TavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(override=True)

# ============================================================
# PAGE CONFIG - Compact for embedding
# ============================================================
st.set_page_config(
    page_title="MWR Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# COMPACT CSS FOR EMBEDDED VIEW
# ============================================================
st.markdown("""
<style>
    /* Hide ALL Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Remove default padding */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Compact chat messages */
    .stChatMessage {
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 5px;
        font-size: 14px;
    }

    /* Hide sidebar completely */
    [data-testid="stSidebar"] {display: none;}

    /* Make chat input more compact */
    .stChatInput {
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "graph_rag" not in st.session_state:
    try:
        st.session_state.graph_rag = MWRGraphRAG()
        st.session_state.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        st.session_state.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2048
        )
        st.session_state.connected = True
    except Exception as e:
        st.session_state.connected = False
        st.error(f"❌ Connection failed: {e}")

# ============================================================
# SESSION MEMORY
# ============================================================
MAX_MEMORY_TURNS = 10


def get_conversation_context() -> str:
    if not st.session_state.conversation_history:
        return ""
    recent = st.session_state.conversation_history[-(MAX_MEMORY_TURNS * 2):]
    context_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500] if len(msg["content"]) > 500 else msg["content"]
        context_lines.append(f"{role}: {content}")
    return "\n".join(context_lines)


def resolve_follow_up(question: str) -> str:
    conversation_context = get_conversation_context()
    if not conversation_context:
        return question

    follow_up_indicators = [
        "it", "that", "those", "them", "this", "these",
        "the top one", "the first", "the same", "more detail",
        "drill", "expand", "compare", "also", "what about",
        "and", "now show", "now filter", "instead", "versus"
    ]

    is_follow_up = any(indicator in question.lower() for indicator in follow_up_indicators)
    if not is_follow_up:
        return question

    resolve_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question resolver. Given the conversation history and a follow-up question,
rewrite the follow-up as a STANDALONE question that includes all necessary context.
Return ONLY the rewritten question, nothing else."""),
        ("human", """Conversation History:
{history}

Follow-up Question: {question}

Rewritten standalone question:""")
    ])

    chain = resolve_prompt | st.session_state.llm | StrOutputParser()
    try:
        resolved = chain.invoke({"history": conversation_context, "question": question}).strip()
        return resolved if resolved else question
    except:
        return question


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def classify_question(question: str) -> str:
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
    try:
        response = st.session_state.tavily.search(
            query=f"minimum wage {query} 2025",
            max_results=2
        )
        return response.get("results", [])
    except:
        return []


def generate_response(question: str) -> str:
    resolved_question = resolve_follow_up(question)
    q_type = classify_question(resolved_question)
    context_parts = []

    if q_type in ["DATABASE", "BOTH"]:
        try:
            db_result = st.session_state.graph_rag.answer_question(resolved_question)
            context_parts.append(f"**Database Results:**\n{db_result['answer']}")
        except Exception as e:
            context_parts.append(f"Database query error: {e}")

    if q_type in ["WEB_SEARCH", "BOTH"]:
        web_results = search_web(resolved_question)
        if web_results:
            web_text = "\n**Recent News:**\n"
            for r in web_results[:2]:
                web_text += f"- {r.get('title', 'No title')}\n"
            context_parts.append(web_text)

    conversation_context = get_conversation_context()

    if context_parts:
        context = "\n\n".join(context_parts)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the MWR AI Assistant for Sodexo, embedded inside a Power BI dashboard.
You are a workforce market intelligence system that helps executives analyze minimum wage risk,
labor costs, education demographics, and workforce data for contract bidding and strategic planning.

DATA AVAILABLE: Risk scores, education (5 levels), workforce population (18-64), cost of labor,
cost of living (ERI indices, 100=national avg), unemployment, demographics, 40,000+ ZIP codes.

Keep answers CONCISE - executives want quick insights, not essays.
Use bullet points for clarity. Limit responses to 3-5 key points.
Include specific numbers when available.
Explain business implications for Sodexo's bidding strategy.

You have memory of this conversation. Use it for follow-ups.

Previous Conversation:
{conversation_history}"""),
            ("human", """Question: {question}

Context:
{context}

Provide a concise answer:""")
        ])
        chain = prompt | st.session_state.llm | StrOutputParser()
        return chain.invoke({
            "question": question,
            "context": context,
            "conversation_history": conversation_context if conversation_context else "No previous conversation."
        })
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the MWR AI Assistant embedded in Power BI — a workforce market intelligence system.
Be concise and business-friendly. You can answer about risk scores, education, labor costs, workforce data.

Previous Conversation:
{conversation_history}"""),
            ("human", "{question}")
        ])
        chain = prompt | st.session_state.llm | StrOutputParser()
        return chain.invoke({
            "question": question,
            "conversation_history": conversation_context if conversation_context else "No previous conversation."
        })


# ============================================================
# COMPACT UI
# ============================================================
# Header
turns = len(st.session_state.conversation_history) // 2
connected = st.session_state.get("connected", False)
status_dot = "🟢" if connected else "🔴"
status_text = "Connected" if connected else "Disconnected"
mem_text = f" | 🧠 {turns} exchanges" if turns > 0 else ""

st.markdown(
    f'<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);'
    f' color: white; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;">'
    f'<span style="font-size: 16px; font-weight: bold;">🤖 MWR AI Assistant</span>'
    f'<span style="font-size: 11px; float: right; color: #4ade80;">'
    f'{status_dot} {status_text}{mem_text}</span>'
    f'</div>',
    unsafe_allow_html=True
)

if not connected:
    st.error("❌ Not connected to database")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Welcome message if no history
if not st.session_state.messages:
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 20px; font-size: 13px;">'
        '💬 Ask me about minimum wage risk<br>'
        '<i>Try: "What\'s California\'s risk score?"</i>'
        '</div>',
        unsafe_allow_html=True
    )

# Chat input
if prompt := st.chat_input("Ask about MWR..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍"):
            response = generate_response(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "assistant", "content": response})
    st.rerun()

# Compact clear button at bottom
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
