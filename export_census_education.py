"""
Export Census Education & Workforce data from Neo4j to CSV for Power BI
Fixed: uses avg instead of sum for workforce_population (county values repeated per ZIP)
"""

import os
import csv
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

OUTPUT_DIR = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Census_Education_Workforce_Jan2026.csv")


def export_county_education():
    print("=" * 60)
    print("EXPORT CENSUS EDUCATION & WORKFORCE DATA (FIXED)")
    print("=" * 60)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Use avg for ALL fields since county-level Census data is repeated per ZIP
            result = session.run('''
                MATCH (z:ZipCode)
                WHERE z.pct_no_diploma IS NOT NULL AND z.state IS NOT NULL AND z.county IS NOT NULL
                WITH z.state AS State, z.county AS County,
                     avg(z.pct_no_diploma) AS Pct_No_Diploma,
                     avg(z.pct_hs_diploma) AS Pct_HS_Diploma,
                     avg(z.pct_some_college) AS Pct_Some_College,
                     avg(z.pct_bachelors) AS Pct_Bachelors,
                     avg(z.pct_graduate) AS Pct_Graduate,
                     avg(z.workforce_population) AS Workforce_Population,
                     count(z) AS ZIP_Count
                RETURN State, County, 
                       round(Pct_No_Diploma, 1) AS Pct_No_Diploma,
                       round(Pct_HS_Diploma, 1) AS Pct_HS_Diploma,
                       round(Pct_Some_College, 1) AS Pct_Some_College,
                       round(Pct_Bachelors, 1) AS Pct_Bachelors,
                       round(Pct_Graduate, 1) AS Pct_Graduate,
                       toInteger(round(Workforce_Population)) AS Workforce_Population,
                       ZIP_Count
                ORDER BY State, County
            ''')
            
            records = list(result)
            print(f"   Retrieved {len(records)} county records")
            
            if not records:
                print("   No data returned.")
                return
            
            rows = []
            for r in records:
                state = r["State"]
                county = r["County"]
                if state and county:
                    rows.append({
                        "County_State_Key": f"{county}|{state}",
                        "Pct_No_Diploma": r["Pct_No_Diploma"],
                        "Pct_HS_Diploma": r["Pct_HS_Diploma"],
                        "Pct_Some_College": r["Pct_Some_College"],
                        "Pct_Bachelors": r["Pct_Bachelors"],
                        "Pct_Graduate": r["Pct_Graduate"],
                        "Workforce_Population": r["Workforce_Population"],
                    })
            
            if rows:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                fieldnames = ["County_State_Key", "Pct_No_Diploma", "Pct_HS_Diploma", 
                              "Pct_Some_College", "Pct_Bachelors", "Pct_Graduate", 
                              "Workforce_Population"]
                
                with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                
                print(f"\n   Saved {len(rows)} counties to:")
                print(f"      {OUTPUT_FILE}")
                
                # Show SF sample to verify
                print(f"\n   Verification - San Francisco:")
                for row in rows:
                    if "San Francisco" in row["County_State_Key"]:
                        print(f"      {row}")
                        break
                
                print(f"\n   Sample data:")
                for row in rows[:5]:
                    print(f"      {row['County_State_Key']}: "
                          f"NoDip={row['Pct_No_Diploma']}%, "
                          f"Workforce={row['Workforce_Population']:,}")
                
    finally:
        driver.close()


if __name__ == "__main__":
    export_county_education()
