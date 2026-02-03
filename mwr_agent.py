"""
MWR AI Agent - Main orchestrator for Minimum Wage Risk analysis
Combines GraphRAG, Web Search, and LLM reasoning
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient
from neo4j_graphrag import MWRGraphRAG

# Load environment variables
load_dotenv(override=True)


class MWRAgent:
    """
    AI Agent for Minimum Wage Risk analysis
    Orchestrates GraphRAG, Web Search, and LLM reasoning
    """
    
    def __init__(self):
        print("🚀 Initializing MWR AI Agent...")
        
        # Initialize components
        self.graph_rag = MWRGraphRAG()
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=4096
        )
        
        print("✅ MWR AI Agent ready!\n")
    
    def search_web(self, query: str, max_results: int = 3) -> list:
        """Search the web for minimum wage news and updates"""
        try:
            response = self.tavily.search(
                query=query,
                max_results=max_results,
                search_depth="basic"
            )
            return response.get("results", [])
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return []
    
    def classify_question(self, question: str) -> dict:
        """Classify the question to determine which tools to use"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a question classifier for a Minimum Wage Risk (MWR) system.

Analyze the question and determine which tools are needed:

1. DATABASE - Questions about specific data, scores, states, ZIP codes, risk tiers
   Examples: "What is California's risk score?", "How many critical ZIP codes?", "Top 5 states?"

2. WEB_SEARCH - Questions about current news, recent legislation, future changes
   Examples: "What are the latest minimum wage changes?", "Any new laws in 2025?"

3. BOTH - Questions that need current data AND latest news
   Examples: "What's California's risk and any upcoming changes?", "Analyze NY with recent news"

4. GENERAL - General questions about minimum wage concepts, no specific data needed
   Examples: "What is minimum wage?", "How does risk scoring work?"

Respond with ONLY one of: DATABASE, WEB_SEARCH, BOTH, GENERAL"""),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        classification = chain.invoke({"question": question}).strip().upper()
        
        # Ensure valid classification
        valid_types = ["DATABASE", "WEB_SEARCH", "BOTH", "GENERAL"]
        if classification not in valid_types:
            classification = "DATABASE"  # Default to database
        
        return {"type": classification, "question": question}
    
    def process_question(self, question: str) -> str:
        """
        Main method to process a user question
        Orchestrates all components to provide a comprehensive answer
        """
        
        print(f"\n{'='*70}")
        print(f"🎯 USER QUESTION: {question}")
        print('='*70)
        
        # Step 1: Classify the question
        print("\n📋 Step 1: Classifying question...")
        classification = self.classify_question(question)
        print(f"   Type: {classification['type']}")
        
        # Step 2: Gather information based on classification
        db_results = None
        web_results = None
        
        if classification["type"] in ["DATABASE", "BOTH"]:
            print("\n📊 Step 2a: Querying database...")
            db_response = self.graph_rag.answer_question(question)
            db_results = {
                "cypher": db_response["cypher"],
                "data": db_response["results"],
                "summary": db_response["answer"]
            }
        
        if classification["type"] in ["WEB_SEARCH", "BOTH"]:
            print("\n🌐 Step 2b: Searching web...")
            # Create a search query focused on minimum wage
            search_query = f"minimum wage {question} 2025 2026"
            web_results = self.search_web(search_query)
            if web_results:
                print(f"   Found {len(web_results)} web results")
                for i, result in enumerate(web_results[:3], 1):
                    print(f"   {i}. {result.get('title', 'No title')[:50]}...")
        
        # Step 3: Generate comprehensive answer
        print("\n💡 Step 3: Generating comprehensive answer...")
        
        answer = self._generate_final_answer(
            question=question,
            classification=classification["type"],
            db_results=db_results,
            web_results=web_results
        )
        
        return answer
    
    def _generate_final_answer(self, question: str, classification: str, 
                                db_results: dict = None, web_results: list = None) -> str:
        """Generate the final comprehensive answer"""
        
        # Build context from available sources
        context_parts = []
        
        if db_results:
            context_parts.append(f"""
DATABASE RESULTS:
Query: {db_results['cypher']}
Data: {str(db_results['data'][:10]) if db_results['data'] else 'No data found'}
Summary: {db_results['summary']}
""")
        
        if web_results:
            web_context = "\nWEB SEARCH RESULTS:\n"
            for i, result in enumerate(web_results[:3], 1):
                web_context += f"""
{i}. {result.get('title', 'No title')}
   URL: {result.get('url', 'No URL')}
   Summary: {result.get('content', 'No content')[:300]}...
"""
            context_parts.append(web_context)
        
        context = "\n".join(context_parts) if context_parts else "No specific data available."
        
        # Generate final answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI assistant for Sodexo's Minimum Wage Risk (MWR) system.

Your role is to provide comprehensive, actionable insights about minimum wage risks.

Guidelines:
- Be concise but thorough
- Include specific numbers when available
- Highlight key risks and recommendations
- Use business-friendly language
- Format with headers and bullet points for readability
- If data is from the database, mention it's from our MWR system
- If data is from web search, cite the sources

Question Type: {classification}"""),
            ("human", """Question: {question}

Available Information:
{context}

Provide a comprehensive answer:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "question": question,
            "classification": classification,
            "context": context
        })
        
        return answer
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*70)
        print("🤖 MWR AI AGENT - Interactive Mode")
        print("="*70)
        print("Ask me anything about Minimum Wage Risk!")
        print("Type 'quit' or 'exit' to end the session.")
        print("="*70 + "\n")
        
        while True:
            try:
                question = input("👤 You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! MWR Agent shutting down...")
                    break
                
                answer = self.process_question(question)
                print(f"\n🤖 MWR Agent:\n{answer}\n")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def close(self):
        """Clean up resources"""
        self.graph_rag.close()
        print("✅ MWR Agent closed.")


# ============================================================
# MAIN - Interactive Chat
# ============================================================
if __name__ == "__main__":
    # Initialize agent
    agent = MWRAgent()
    
    # Start interactive chat
    agent.chat()
    
    # Clean up
    agent.close()