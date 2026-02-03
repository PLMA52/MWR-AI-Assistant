"""
Neo4j GraphRAG Module for MWR AI Automation
Query the MWR graph database using natural language
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

class MWRGraphRAG:
    """
    GraphRAG system for querying MWR Neo4j database
    """
    
    def __init__(self):
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=4096
        )
        
        # Get database schema for context
        self.schema = self._get_schema()
        
        print("✅ MWR GraphRAG initialized successfully!")
    
    def _get_schema(self):
        """Get the database schema for LLM context"""
        schema_query = """
        CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes
        RETURN nodeType, collect({property: propertyName, types: propertyTypes}) as properties
        """
        
        with self.driver.session() as session:
            result = session.run(schema_query)
            schema_info = []
            for record in result:
                node_type = record["nodeType"]
                properties = record["properties"]
                prop_list = [f"{p['property']}" for p in properties[:15]]  # Limit to 15 properties
                schema_info.append(f"{node_type}: {', '.join(prop_list)}")
            
            return "\n".join(schema_info)
    
    def _run_cypher(self, query: str):
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]
        except Exception as e:
            return f"Error executing query: {e}"
    
    def generate_cypher(self, question: str) -> str:
        """Use LLM to generate Cypher query from natural language"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Neo4j Cypher expert for a Minimum Wage Risk (MWR) database.

DATABASE SCHEMA:
{schema}

KEY NODE TYPES:
- ZipCode: Contains zip, county, state, fips, New_Risk_Score_Pct (0-100 risk score), New_Risk_Tier (Critical/High/Elevated/Moderate/Low), New_Combined_Risk_Score (0-500 scale), County_Risk_Score_Pct, HCF, MHJ, Current_Min_Wage, City_Jurisdictions, Industry_Carveouts, unemployment_rate (county unemployment % from BLS), unemployment_updated (date of last update)
- State: Contains name, abbr (state abbreviation like 'CA', 'NY')
- County: Contains name, fips
- CBSA: Contains name, cbsa_code

KEY RELATIONSHIPS:
- (ZipCode)-[:IN_STATE]->(State)
- (ZipCode)-[:IN_COUNTY]->(County)
- (ZipCode)-[:IN_CBSA]->(CBSA)

RISK TIERS (based on New_Risk_Score_Pct):
- Critical: 70-100
- High: 55-70
- Elevated: 40-55
- Moderate: 25-40
- Low: 0-25

UNEMPLOYMENT DATA:
- unemployment_rate: The county's unemployment rate as a percentage (e.g., 3.4 means 3.4%)
- unemployment_updated: Date when the data was last updated (e.g., "2026-02-03")
- To get unemployment for a county: MATCH (z:ZipCode) WHERE z.county = "Boulder" AND z.state = "CO" RETURN DISTINCT z.county, z.state, z.unemployment_rate LIMIT 1
- To find high unemployment counties: MATCH (z:ZipCode) WHERE z.unemployment_rate > 5 RETURN DISTINCT z.county, z.state, z.unemployment_rate ORDER BY z.unemployment_rate DESC

IMPORTANT:
- State abbreviations are stored in State.abbr (e.g., 'CA', 'NY', 'TX')
- The state field on ZipCode uses abbreviations like 'CO', 'CA', 'NY'
- Always use LIMIT to prevent returning too many results
- For state queries, use: MATCH (s:State {{abbr: 'CA'}})<-[:IN_STATE]-(z:ZipCode)
- For county unemployment queries, filter by county name AND state: WHERE z.county = "Boulder" AND z.state = "CO"
- When asked about unemployment, always include unemployment_rate in the RETURN

Generate ONLY the Cypher query, no explanations."""),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        cypher = chain.invoke({
            "schema": self.schema,
            "question": question
        })
        
        # Clean up the response
        cypher = cypher.strip()
        if cypher.startswith("```"):
            cypher = cypher.split("```")[1]
            if cypher.startswith("cypher"):
                cypher = cypher[6:]
        cypher = cypher.strip()
        
        return cypher
    
    def answer_question(self, question: str) -> dict:
        """
        Answer a natural language question about MWR data
        Returns: dict with cypher, results, and answer
        """
        
        print(f"\n🔍 Question: {question}")
        
        # Step 1: Generate Cypher
        print("📝 Generating Cypher query...")
        cypher = self.generate_cypher(question)
        print(f"   Query: {cypher[:100]}...")
        
        # Step 2: Execute query
        print("⚡ Executing query...")
        results = self._run_cypher(cypher)
        
        if isinstance(results, str) and results.startswith("Error"):
            print(f"   ❌ {results}")
            return {
                "question": question,
                "cypher": cypher,
                "results": None,
                "answer": f"Query failed: {results}"
            }
        
        print(f"   ✅ Found {len(results)} results")
        
        # Step 3: Generate natural language answer
        print("💬 Generating answer...")
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert analyst for Minimum Wage Risk (MWR) data at Sodexo.
            
Given the user's question and the database results, provide a clear, concise answer.
Include specific numbers and insights. Format nicely for business users.

If the results are empty, explain what that means.
If there are many results, summarize the key findings.

For unemployment data:
- The unemployment_rate is a percentage (e.g., 3.4 means 3.4%)
- Provide context: rates below 4% are generally considered low, 4-6% moderate, above 6% high
- Explain business implications when relevant"""),
            ("human", """Question: {question}

Database Results:
{results}

Provide a clear, business-friendly answer:""")
        ])
        
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        
        # Limit results for context
        results_for_llm = results[:20] if len(results) > 20 else results
        
        answer = answer_chain.invoke({
            "question": question,
            "results": str(results_for_llm)
        })
        
        return {
            "question": question,
            "cypher": cypher,
            "results": results,
            "answer": answer
        }
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()


# ============================================================
# TEST THE GRAPHRAG
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MWR GRAPHRAG - TEST")
    print("=" * 60)
    
    # Initialize
    graph_rag = MWRGraphRAG()
    
    # Test questions
    test_questions = [
        "What is California's average risk score?",
        "How many ZIP codes are in the Critical risk tier?",
        "What are the top 5 highest risk states?",
        "What is the unemployment rate in Boulder County, Colorado?",
        "Which counties have unemployment above 5%?"
    ]
    
    for question in test_questions:
        result = graph_rag.answer_question(question)
        print(f"\n{'='*60}")
        print(f"📊 ANSWER:\n{result['answer']}")
        print("=" * 60)
    
    # Close connection
    graph_rag.close()
    print("\n✅ GraphRAG test complete!")
