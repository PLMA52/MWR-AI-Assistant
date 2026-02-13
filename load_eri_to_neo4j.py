"""
Load ERI Cost of Labor and Cost of Living data into Neo4j
Reads from the ERI CSV and writes cost_of_labor and cost_of_living 
properties to ZipCode nodes, matched by county and state.
"""

import os
import csv
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

# ERI CSV file path
ERI_FILE = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data/ERI_County_Summary_Jan2026_v2.csv")


def load_eri_to_neo4j():
    print("=" * 60)
    print("LOAD ERI COST DATA INTO NEO4J")
    print("=" * 60)
    
    # Step 1: Read the ERI CSV
    print(f"\n   Reading ERI data from: {ERI_FILE}")
    
    if not os.path.exists(ERI_FILE):
        print(f"   ERROR: File not found: {ERI_FILE}")
        print("   Please check the file path.")
        return
    
    eri_data = {}
    with open(ERI_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"   CSV headers: {headers}")
        
        for row in reader:
            # Get the County_State_Key (e.g., "San Francisco|CA")
            key = row.get("County_State_Key", "")
            if not key or "|" not in key:
                continue
            
            county, state = key.split("|", 1)
            
            # Get cost of labor and cost of living - use Avg columns to match Power BI
            col_avg = row.get("Cost_of_Labor_Avg", "")
            coliv_avg = row.get("Cost_of_Living_Avg", "")
            
            try:
                cost_of_labor = float(col_avg) if col_avg else None
                cost_of_living = float(coliv_avg) if coliv_avg else None
            except ValueError:
                continue
            
            if cost_of_labor is not None and cost_of_living is not None:
                eri_data[(county, state)] = {
                    "cost_of_labor": cost_of_labor,
                    "cost_of_living": cost_of_living
                }
    
    print(f"   Loaded {len(eri_data)} county ERI records")
    
    # Show San Francisco sample
    sf_key = ("San Francisco", "CA")
    if sf_key in eri_data:
        print(f"\n   Sample - San Francisco, CA:")
        print(f"      Cost of Labor: {eri_data[sf_key]['cost_of_labor']}")
        print(f"      Cost of Living: {eri_data[sf_key]['cost_of_living']}")
    
    # Step 2: Write to Neo4j
    print(f"\n   Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    updated_total = 0
    not_found = 0
    
    try:
        with driver.session() as session:
            for (county, state), costs in eri_data.items():
                result = session.run('''
                    MATCH (z:ZipCode)
                    WHERE z.county = $county AND z.state = $state
                    SET z.cost_of_labor = $col,
                        z.cost_of_living = $coliv
                    RETURN count(z) as cnt
                ''', county=county, state=state, 
                   col=costs["cost_of_labor"], 
                   coliv=costs["cost_of_living"])
                
                cnt = result.single()["cnt"]
                if cnt > 0:
                    updated_total += cnt
                else:
                    not_found += 1
                
                if updated_total % 5000 == 0 and updated_total > 0:
                    print(f"      Updated {updated_total} ZIPs...")
    
    finally:
        driver.close()
    
    print(f"\n   RESULTS:")
    print(f"      Updated {updated_total} ZIPs with ERI cost data")
    print(f"      Counties not found in Neo4j: {not_found}")
    
    # Step 3: Verify
    print(f"\n   VERIFICATION:")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run('''
                MATCH (z:ZipCode)
                WHERE z.county = "San Francisco" AND z.state = "CA"
                RETURN DISTINCT z.county, z.state, z.cost_of_labor, z.cost_of_living
                LIMIT 1
            ''')
            record = result.single()
            if record:
                print(f"      San Francisco, CA:")
                print(f"         Cost of Labor: {record['z.cost_of_labor']}")
                print(f"         Cost of Living: {record['z.cost_of_living']}")
            else:
                print(f"      San Francisco not found!")
            
            # Check coverage
            result2 = session.run('''
                MATCH (z:ZipCode)
                WHERE z.cost_of_labor IS NOT NULL
                RETURN count(z) as cnt
            ''')
            cnt = result2.single()["cnt"]
            print(f"      Total ZIPs with ERI data: {cnt}")
    finally:
        driver.close()
    
    print(f"\n{'=' * 60}")
    print("DONE! ERI cost data now available in Neo4j for AI Assistant.")
    print("=" * 60)


if __name__ == "__main__":
    load_eri_to_neo4j()
