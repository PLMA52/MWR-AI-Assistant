"""
MWR AI Chat - Streamlit Interface with Session Memory
Embedded chat for Power BI dashboard
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
        ("system", """Classify this question about Minimum Wage Risk:
- DATABASE: Questions about specific data, scores, states, ZIP codes
- WEB_SEARCH: Questions about current news, recent legislation
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

def generate_response(question: str) -> str:
    """Generate comprehensive response with session memory"""
    
    # Step 1: Resolve follow-up questions using conversation history
    resolved_question = resolve_follow_up(question)
    
    # Step 2: Classify the resolved question
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
            ("system", """You are the MWR AI Assistant for Sodexo. 
Provide clear, concise answers about Minimum Wage Risk.
Use the provided context to answer. Be business-friendly.
Format with bullet points and headers when helpful.

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
        return chain.invoke({
            "question": question,
            "context": context,
            "conversation_history": conversation_context if conversation_context else "No previous conversation."
        })
    else:
        # General question - answer directly with conversation history
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the MWR AI Assistant. Answer questions about minimum wage risk clearly and concisely.

IMPORTANT: You have memory of this conversation. Use the conversation history 
to understand context and provide coherent follow-up answers.

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

# Quick action buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📊 Top 5 Risk States", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "What are the top 5 highest risk states?"})
with col2:
    if st.button("🔴 Critical ZIPs", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "How many ZIP codes are in the Critical risk tier?"})
with col3:
    if st.button("☀️ California Risk", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "What is California's risk score and why?"})
with col4:
    if st.button("🗞️ Latest News", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "What are the latest minimum wage news and changes?"})

st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Minimum Wage Risk..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing..."):
            response = generate_response(prompt)
            st.markdown(response)
    
    # Add assistant response to display messages
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add to conversation history (session memory)
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "assistant", "content": response})
    
    # Rerun to update the memory badge count
    st.rerun()

# Process quick action buttons
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_msg = st.session_state.messages[-1]["content"]
    # Check if this message needs a response (from button click)
    if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant":
        pass  # Already handled by chat input
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing..."):
                response = generate_response(last_msg)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Add to conversation history (session memory)
        st.session_state.conversation_history.append({"role": "user", "content": last_msg})
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        
        st.rerun()

# Sidebar with info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **MWR AI Assistant** helps you analyze 
    Minimum Wage Risk data.
    
    **You can ask:**
    - Risk scores by state/ZIP
    - Critical risk areas
    - Latest MW news
    - What-if scenarios
    - **Follow-up questions!** 🧠
    
    **Data Sources:**
    - Neo4j MWR Database
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
        st.rerun()
