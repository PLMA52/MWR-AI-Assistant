"""
Export Census Age Breakout data from Neo4j to CSV for Power BI
Aggregates ZIP-level age percentages to county-level averages
Outputs to OneDrive MWR_Automation_Data folder

Author: Michel Pierre-Louis
Date: February 2026
"""

import os
import csv
from neo4j import GraphDatabase

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

# Output path - OneDrive MWR_Automation_Data folder
OUTPUT_DIR = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Census_Age_Breakout_Feb2026.csv")


def export_county_age_breakout():
    """Query Neo4j for county-level age distribution percentages"""
    
    print("=" * 60)
    print("EXPORT CENSUS AGE BREAKOUT DATA")
    print("=" * 60)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # First check which property name is used for state
            print("\n   Checking property names...")
            check = session.run('''
                MATCH (z:ZipCode) 
                WHERE z.pct_age_0_to_9 IS NOT NULL
                RETURN z.state AS state_val, z.county AS county_val
                LIMIT 1
            ''')
            rec = check.single()
            
            if not rec:
                print("   ERROR: No ZipCode nodes found with age data!")
                print("   Check that age data was loaded into Neo4j.")
                return
            
            print(f"   Sample: state={rec['state_val']}, county={rec['county_val']}")
            
            # Aggregate ZIP-level data to county level using averages
            print("\n   Querying county-level age averages...")
            result = session.run('''
                MATCH (z:ZipCode)
                WHERE z.pct_age_0_to_9 IS NOT NULL 
                  AND z.state IS NOT NULL 
                  AND z.county IS NOT NULL
                WITH z.state AS State, z.county AS County,
                     avg(z.pct_age_0_to_9) AS Pct_Age_0_to_9,
                     avg(z.pct_age_10_to_19) AS Pct_Age_10_to_19,
                     avg(z.pct_age_20_to_29) AS Pct_Age_20_to_29,
                     avg(z.pct_age_30_to_39) AS Pct_Age_30_to_39,
                     avg(z.pct_age_40_to_49) AS Pct_Age_40_to_49,
                     avg(z.pct_age_50_to_59) AS Pct_Age_50_to_59,
                     avg(z.pct_age_60_to_69) AS Pct_Age_60_to_69,
                     avg(z.pct_age_70_plus) AS Pct_Age_70_Plus,
                     count(z) AS ZIP_Count
                RETURN State, County, 
                       round(Pct_Age_0_to_9, 1) AS Pct_Age_0_to_9,
                       round(Pct_Age_10_to_19, 1) AS Pct_Age_10_to_19,
                       round(Pct_Age_20_to_29, 1) AS Pct_Age_20_to_29,
                       round(Pct_Age_30_to_39, 1) AS Pct_Age_30_to_39,
                       round(Pct_Age_40_to_49, 1) AS Pct_Age_40_to_49,
                       round(Pct_Age_50_to_59, 1) AS Pct_Age_50_to_59,
                       round(Pct_Age_60_to_69, 1) AS Pct_Age_60_to_69,
                       round(Pct_Age_70_Plus, 1) AS Pct_Age_70_Plus,
                       ZIP_Count
                ORDER BY State, County
            ''')
            
            records = list(result)
            print(f"   Retrieved {len(records)} county records")
            
            if not records:
                print("   ERROR: No data returned. Check Neo4j properties.")
                return
            
            # Build CSV with County_State_Key for Power BI join
            rows = []
            for r in records:
                state = r["State"]
                county = r["County"]
                if state and county:
                    rows.append({
                        "County_State_Key": f"{county}|{state}",
                        "Pct_Age_0_to_9": r["Pct_Age_0_to_9"],
                        "Pct_Age_10_to_19": r["Pct_Age_10_to_19"],
                        "Pct_Age_20_to_29": r["Pct_Age_20_to_29"],
                        "Pct_Age_30_to_39": r["Pct_Age_30_to_39"],
                        "Pct_Age_40_to_49": r["Pct_Age_40_to_49"],
                        "Pct_Age_50_to_59": r["Pct_Age_50_to_59"],
                        "Pct_Age_60_to_69": r["Pct_Age_60_to_69"],
                        "Pct_Age_70_Plus": r["Pct_Age_70_Plus"],
                    })
            
            # Write CSV
            if rows:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                fieldnames = [
                    "County_State_Key", 
                    "Pct_Age_0_to_9", "Pct_Age_10_to_19", 
                    "Pct_Age_20_to_29", "Pct_Age_30_to_39",
                    "Pct_Age_40_to_49", "Pct_Age_50_to_59",
                    "Pct_Age_60_to_69", "Pct_Age_70_Plus"
                ]
                
                with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                
                print(f"\n   Saved {len(rows)} counties to:")
                print(f"   {OUTPUT_FILE}")
                
                # Show sample data
                print(f"\n   Sample data (first 3 counties):")
                print(f"   {'County':<35} {'0-9':>5} {'10-19':>6} {'20-29':>6} {'30-39':>6} {'40-49':>6} {'50-59':>6} {'60-69':>6} {'70+':>5}")
                print(f"   {'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
                for row in rows[:3]:
                    key = row['County_State_Key'][:35]
                    print(f"   {key:<35} {row['Pct_Age_0_to_9']:>5} {row['Pct_Age_10_to_19']:>6} {row['Pct_Age_20_to_29']:>6} {row['Pct_Age_30_to_39']:>6} {row['Pct_Age_40_to_49']:>6} {row['Pct_Age_50_to_59']:>6} {row['Pct_Age_60_to_69']:>6} {row['Pct_Age_70_Plus']:>5}")
                
                # Verify a known county
                for row in rows:
                    if "San Francisco" in row["County_State_Key"]:
                        print(f"\n   VERIFICATION - San Francisco:")
                        print(f"      0-9: {row['Pct_Age_0_to_9']}%")
                        print(f"      10-19: {row['Pct_Age_10_to_19']}%")
                        print(f"      20-29: {row['Pct_Age_20_to_29']}%")
                        print(f"      30-39: {row['Pct_Age_30_to_39']}%")
                        print(f"      40-49: {row['Pct_Age_40_to_49']}%")
                        print(f"      50-59: {row['Pct_Age_50_to_59']}%")
                        print(f"      60-69: {row['Pct_Age_60_to_69']}%")
                        print(f"      70+: {row['Pct_Age_70_Plus']}%")
                        total = sum([
                            row['Pct_Age_0_to_9'], row['Pct_Age_10_to_19'],
                            row['Pct_Age_20_to_29'], row['Pct_Age_30_to_39'],
                            row['Pct_Age_40_to_49'], row['Pct_Age_50_to_59'],
                            row['Pct_Age_60_to_69'], row['Pct_Age_70_Plus']
                        ])
                        print(f"      TOTAL: {total:.1f}% (should be ~100%)")
                        break
            else:
                print("   ERROR: No valid rows to write")
                
    finally:
        driver.close()
    
    print(f"\n{'=' * 60}")
    print("DONE!")
    print("=" * 60)
    print("\nNEXT STEPS IN POWER BI:")
    print("1. Open Power BI Desktop")
    print("2. Get Data > CSV > Census_Age_Breakout_Feb2026.csv")
    print("3. Create relationship: County_State_Key > County_Summary.County_State_Key")
    print("4. Add age columns to Inform tab (Table 2 or new Table 3)")
    print("5. Publish to Power BI Service")


if __name__ == "__main__":
    export_county_age_breakout()
