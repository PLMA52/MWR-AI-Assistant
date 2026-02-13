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
- ZipCode: Contains zip, county, state, fips, New_Risk_Score_Pct (0-100 risk score), New_Risk_Tier (Critical/High/Elevated/Moderate/Low), New_Combined_Risk_Score (0-500 scale), County_Risk_Score_Pct, HCF, MHJ, Current_Min_Wage, City_Jurisdictions, Industry_Carveouts, unemployment_rate (county unemployment % from BLS), unemployment_updated (date of last update), total_population (county population), median_household_income (median income in $), median_age (median age in years), median_home_value (median home value in $), college_educated_count (people with bachelor's or higher), census_updated (date of census data update), pct_no_diploma (% of 25+ population with no high school diploma), pct_hs_diploma (% with high school diploma or GED), pct_some_college (% with some college or associate's degree), pct_bachelors (% with bachelor's degree), pct_graduate (% with graduate/professional degree), workforce_population (estimated population ages 18-64), cost_of_labor (ERI Cost of Labor index, 100 = national average), cost_of_living (ERI Cost of Living index, 100 = national average), pct_age_0_to_9 (% of population aged 0-9), pct_age_10_to_19 (% aged 10-19), pct_age_20_to_29 (% aged 20-29), pct_age_30_to_39 (% aged 30-39), pct_age_40_to_49 (% aged 40-49), pct_age_50_to_59 (% aged 50-59), pct_age_60_to_69 (% aged 60-69), pct_age_70_plus (% aged 70+)
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

DEMOGRAPHICS DATA (from Census):
- total_population: County population count
- median_household_income: Median household income in dollars
- median_age: Median age in years
- median_home_value: Median home value in dollars
- college_educated_count: Number of people with bachelor's degree or higher
- To get demographics for a county: MATCH (z:ZipCode) WHERE z.county = "Boulder" AND z.state = "CO" RETURN DISTINCT z.county, z.state, z.total_population, z.median_household_income, z.median_age LIMIT 1
- To find wealthy counties: MATCH (z:ZipCode) WHERE z.median_household_income > 100000 RETURN DISTINCT z.county, z.state, z.median_household_income ORDER BY z.median_household_income DESC LIMIT 10

EDUCATION DATA (from Census B15003, for population 25+):
- pct_no_diploma: Percentage with no high school diploma (includes all grade levels below HS completion)
- pct_hs_diploma: Percentage with high school diploma or GED equivalent
- pct_some_college: Percentage with some college or associate's degree
- pct_bachelors: Percentage with bachelor's degree
- pct_graduate: Percentage with graduate or professional degree (master's, doctorate, professional)
- These 5 percentages sum to approximately 100% for each county
- All education data is stored at the county level (same value for all ZIPs in a county)
- To get education profile: MATCH (z:ZipCode) WHERE z.county = "San Francisco" AND z.state = "CA" RETURN DISTINCT z.county, z.state, z.pct_no_diploma, z.pct_hs_diploma, z.pct_some_college, z.pct_bachelors, z.pct_graduate LIMIT 1
- To find highly educated counties: MATCH (z:ZipCode) WHERE z.pct_bachelors > 40 RETURN DISTINCT z.county, z.state, z.pct_bachelors, z.pct_graduate ORDER BY z.pct_bachelors DESC LIMIT 10
- To find counties with low education: MATCH (z:ZipCode) WHERE z.pct_no_diploma > 20 RETURN DISTINCT z.county, z.state, z.pct_no_diploma ORDER BY z.pct_no_diploma DESC LIMIT 10

WORKFORCE POPULATION DATA:
- workforce_population: Estimated working-age population (ages 18-64) for the county
- Calculated from Census B01001 (Sex by Age): Total Population - Under 18 - 65 and over
- All ZIPs in a county share the same workforce_population value
- To get workforce for a county: MATCH (z:ZipCode) WHERE z.county = "San Francisco" AND z.state = "CA" RETURN DISTINCT z.county, z.state, z.workforce_population, z.total_population LIMIT 1
- To find large workforce markets: MATCH (z:ZipCode) WHERE z.workforce_population > 500000 RETURN DISTINCT z.county, z.state, z.workforce_population ORDER BY z.workforce_population DESC LIMIT 10

ERI COST DATA (from Economic Research Institute):
- cost_of_labor: ERI Cost of Labor index where 100 = national average. Values above 100 mean labor costs are higher than average. Example: 139.9 means labor costs are 39.9% above national average.
- cost_of_living: ERI Cost of Living index where 100 = national average. Values above 100 mean cost of living is higher than average. Example: 221.6 means cost of living is 121.6% above national average.
- Both are at the county level (same value for all ZIPs in a county)
- To get cost data: MATCH (z:ZipCode) WHERE z.county = "San Francisco" AND z.state = "CA" RETURN DISTINCT z.county, z.state, z.cost_of_labor, z.cost_of_living LIMIT 1
- To find expensive labor markets: MATCH (z:ZipCode) WHERE z.cost_of_labor > 120 RETURN DISTINCT z.county, z.state, z.cost_of_labor, z.cost_of_living ORDER BY z.cost_of_labor DESC LIMIT 10
- To find affordable markets: MATCH (z:ZipCode) WHERE z.cost_of_living < 90 RETURN DISTINCT z.county, z.state, z.cost_of_living, z.cost_of_labor ORDER BY z.cost_of_living ASC LIMIT 10

AGE BREAKOUT DATA (8 decade groups, from demographic data):
- pct_age_0_to_9: Percentage of population aged 0-9
- pct_age_10_to_19: Percentage of population aged 10-19
- pct_age_20_to_29: Percentage of population aged 20-29
- pct_age_30_to_39: Percentage of population aged 30-39
- pct_age_40_to_49: Percentage of population aged 40-49
- pct_age_50_to_59: Percentage of population aged 50-59
- pct_age_60_to_69: Percentage of population aged 60-69
- pct_age_70_plus: Percentage of population aged 70+
- These 8 percentages sum to approximately 100% for each county
- All age data is stored at the county level (same value for all ZIPs in a county)
- To get age profile: MATCH (z:ZipCode) WHERE z.county = "San Francisco" AND z.state = "CA" RETURN DISTINCT z.county, z.state, z.pct_age_0_to_9, z.pct_age_10_to_19, z.pct_age_20_to_29, z.pct_age_30_to_39, z.pct_age_40_to_49, z.pct_age_50_to_59, z.pct_age_60_to_69, z.pct_age_70_plus LIMIT 1
- To find young workforce counties: MATCH (z:ZipCode) WHERE z.pct_age_20_to_29 > 18 RETURN DISTINCT z.county, z.state, z.pct_age_20_to_29 ORDER BY z.pct_age_20_to_29 DESC LIMIT 10
- To find aging population counties: MATCH (z:ZipCode) WHERE z.pct_age_70_plus > 15 RETURN DISTINCT z.county, z.state, z.pct_age_70_plus ORDER BY z.pct_age_70_plus DESC LIMIT 10

COMPREHENSIVE MARKET PROFILE:
- When asked for a full market profile or "tell me everything about" a county, include: risk score, population, workforce, age breakout, education, unemployment, income, cost of labor, cost of living
- Example: MATCH (z:ZipCode) WHERE z.county = "San Francisco" AND z.state = "CA" RETURN DISTINCT z.county, z.state, z.New_Risk_Score_Pct, z.New_Risk_Tier, z.total_population, z.workforce_population, z.median_age, z.pct_age_0_to_9, z.pct_age_10_to_19, z.pct_age_20_to_29, z.pct_age_30_to_39, z.pct_age_40_to_49, z.pct_age_50_to_59, z.pct_age_60_to_69, z.pct_age_70_plus, z.pct_no_diploma, z.pct_hs_diploma, z.pct_some_college, z.pct_bachelors, z.pct_graduate, z.unemployment_rate, z.median_household_income, z.cost_of_labor, z.cost_of_living LIMIT 1

IMPORTANT:
- State abbreviations are stored in State.abbr (e.g., 'CA', 'NY', 'TX')
- The state field on ZipCode uses abbreviations like 'CO', 'CA', 'NY'
- Always use LIMIT to prevent returning too many results
- For state queries, use: MATCH (s:State {{abbr: 'CA'}})<-[:IN_STATE]-(z:ZipCode)
- For county queries, filter by county name AND state: WHERE z.county = "Boulder" AND z.state = "CO"
- When asked about unemployment, always include unemployment_rate in the RETURN
- When asked about education, include all 5 education percentage fields
- When asked about costs, include both cost_of_labor and cost_of_living
- Education, workforce, and cost data are county-level — use DISTINCT to avoid duplicates
- When asked about age, age distribution, age breakout, or demographics by age, include all 8 pct_age fields
- CRITICAL DISAMBIGUATION: When users ask about "cost of labor" or "cost of living", ALWAYS use the ERI index properties (z.cost_of_labor, z.cost_of_living), NOT Current_Min_Wage or median_household_income. The ERI indices are the correct fields for cost questions. Current_Min_Wage is the hourly minimum wage rate. median_household_income is household income. These are DIFFERENT from cost indices.
- When users ask about "wages" or "minimum wage", use Current_Min_Wage
- When users ask about "income" or "salary", use median_household_income
- When users ask about "cost", "expensive", "affordable", "labor costs", "cost index", use cost_of_labor and cost_of_living

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
- Explain business implications when relevant

For demographics data:
- Population helps understand market size
- Median income indicates economic strength and wage expectations
- Median age helps understand workforce demographics
- Home values correlate with cost of living

For education data:
- Five education levels: No Diploma, HS Diploma, Some College, Bachelor's, Graduate
- High pct_bachelors + pct_graduate indicates a highly educated workforce (higher wage expectations)
- High pct_no_diploma may indicate vulnerable worker populations and potential compliance challenges
- Compare to national averages when helpful: ~11% no diploma, ~27% HS, ~29% some college, ~21% bachelor's, ~13% graduate

For workforce population:
- workforce_population is the estimated population aged 18-64
- This indicates the labor pool size available in a county
- Compare to total_population to understand the working-age share (typically 60-70%)

For ERI cost data:
- cost_of_labor: Index where 100 = national average. Above 100 means more expensive labor market.
- cost_of_living: Index where 100 = national average. Above 100 means higher cost of living.
- High cost of labor (>120) means Sodexo likely needs higher wages to compete for talent
- High cost of living (>150) means minimum wage may be insufficient for workers, increasing retention risk
- These are critical for contract bidding — they indicate how much Sodexo needs to budget for labor
- When both indices are high, it signals a very competitive and expensive market

For comprehensive market profiles:
- Synthesize all available data into a business narrative
- Highlight factors that affect Sodexo's bidding strategy: risk level, labor costs, education mix, workforce size
- Identify risks and opportunities: e.g., high education + high cost = competitive talent market

For age breakout data:
- 8 decade groups: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
- High pct_age_20_to_29 and pct_age_30_to_39 indicates a young, active workforce — higher turnover but larger labor pool
- High pct_age_50_to_59 and pct_age_60_to_69 indicates an aging workforce — potential retirement wave, harder to fill positions
- High pct_age_70_plus indicates a retirement community — different service needs (healthcare, senior living)
- High pct_age_0_to_9 indicates family-oriented community — may need family-friendly benefits to attract workers
- Compare to business context: young workforce areas have more entry-level labor but higher turnover; aging areas need retention strategies"""),
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
        "Which counties have unemployment above 5%?",
        "What is the population and median income in Los Angeles County?",
        "Which counties have median household income above $100,000?",
        "What is the education breakdown in San Francisco County, California?",
        "Which counties have the highest percentage of bachelor's degrees?",
        "What is the cost of labor and cost of living in San Francisco?",
        "Which counties have cost of labor above 130?",
        "Give me a full market profile for Boulder County, Colorado",
        "What is the workforce population in Los Angeles County?"
    ]
    
    for question in test_questions:
        result = graph_rag.answer_question(question)
        print(f"\n{'='*60}")
        print(f"📊 ANSWER:\n{result['answer']}")
        print("=" * 60)
    
    # Close connection
    graph_rag.close()
    print("\n✅ GraphRAG test complete!")
