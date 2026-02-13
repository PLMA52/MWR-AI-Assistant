"""
Diagnose county name mismatches between Dan's age data and Neo4j
"""
import os
import csv
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

CSV_FILE = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data/Census_Age_Breakout_Jan2026.csv")

def diagnose():
    # Read counties from CSV (Dan's data)
    csv_counties = set()
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_counties.add((row["County"], row["State"]))
    
    print(f"Counties in CSV: {len(csv_counties)}")
    
    # Read counties from Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    neo4j_counties = set()
    with driver.session() as session:
        result = session.run("MATCH (z:ZipCode) RETURN DISTINCT z.county as county, z.state as state")
        for record in result:
            neo4j_counties.add((record["county"], record["state"]))
    driver.close()
    
    print(f"Counties in Neo4j: {len(neo4j_counties)}")
    
    # Find matches and mismatches
    matched = csv_counties & neo4j_counties
    csv_only = csv_counties - neo4j_counties
    neo4j_only = neo4j_counties - csv_counties
    
    print(f"\nMatched: {matched.__len__()}")
    print(f"In CSV but NOT in Neo4j: {len(csv_only)}")
    print(f"In Neo4j but NOT in CSV: {len(neo4j_only)}")
    
    # Show sample mismatches by state
    print(f"\n--- Sample CSV-only counties (first 30) ---")
    for county, state in sorted(csv_only)[:30]:
        print(f"   {county}, {state}")
    
    print(f"\n--- Sample Neo4j-only counties (first 30) ---")
    for county, state in sorted(neo4j_only)[:30]:
        print(f"   {county}, {state}")
    
    # Look for near-matches (same state, similar name)
    print(f"\n--- Possible name mismatches (same state) ---")
    count = 0
    for csv_county, csv_state in sorted(csv_only):
        for neo_county, neo_state in neo4j_only:
            if csv_state == neo_state:
                # Check if names are similar
                csv_lower = csv_county.lower().replace(".", "").replace("'", "").replace(" ", "")
                neo_lower = neo_county.lower().replace(".", "").replace("'", "").replace(" ", "")
                if csv_lower == neo_lower or csv_lower in neo_lower or neo_lower in csv_lower:
                    print(f"   {csv_state}: CSV='{csv_county}' <-> Neo4j='{neo_county}'")
                    count += 1
                    if count >= 30:
                        break
        if count >= 30:
            break
    
    if count == 0:
        print("   No obvious near-matches found")

if __name__ == "__main__":
    diagnose()
