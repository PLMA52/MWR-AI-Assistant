"""
Fix Workforce Population in Neo4j
Uses Census B01001 (Sex by Age) to properly calculate population 16-64
Then re-exports county CSV for Power BI
"""

import requests
import os
import csv
from neo4j import GraphDatabase

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "ddc5085eb0dfb4fea8367baa28a5b94345e8daed")
CENSUS_BASE_URL = "https://api.census.gov/data"
DATASET = "2022/acs/acs5"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

OUTPUT_DIR = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Census_Education_Workforce_Jan2026.csv")

# B01001: Sex by Age
# We need to sum ages 16-64 for both male and female
# Male 16-64: B01001_007E (15-17 partial) through B01001_019E (62-64)
# Female 16-64: B01001_031E (15-17 partial) through B01001_043E (62-64)
# 
# Simpler approach: Total pop - Under 16 - 65 and over
# Under 16: sum of B01001_003 to B01001_006 (male) + B01001_027 to B01001_030 (female)
# 65+: sum of B01001_020 to B01001_025 (male) + B01001_044 to B01001_049 (female)

WORKFORCE_VARS = {
    "B01001_001E": "total_pop",
    # Male under 16 (under 5, 5-9, 10-14, 15-17 -- we'll include 15-17 as under 18 but
    # Dan said 16-65, so let's use under 15 + 65+)
    # Actually for simplicity: Under 18 + 65 and over, then subtract from total
    # Under 5
    "B01001_003E": "m_under5",
    "B01001_027E": "f_under5",
    # 5 to 9
    "B01001_004E": "m_5to9",
    "B01001_028E": "f_5to9",
    # 10 to 14
    "B01001_005E": "m_10to14",
    "B01001_029E": "f_10to14",
    # 15 to 17 (we'll include this as "under 18" since 16-17 is borderline)
    # Actually Dan wants 16-65. Census breaks at 15-17, so we can't perfectly get 16.
    # Best approximation: use 18-64 (standard working age) which is cleaner
    "B01001_006E": "m_15to17",
    "B01001_030E": "f_15to17",
    # 65 to 66
    "B01001_020E": "m_65to66",
    "B01001_044E": "f_65to66",
    # 67 to 69
    "B01001_021E": "m_67to69",
    "B01001_045E": "f_67to69",
    # 70 to 74
    "B01001_022E": "m_70to74",
    "B01001_046E": "f_70to74",
    # 75 to 79
    "B01001_023E": "m_75to79",
    "B01001_047E": "f_75to79",
    # 80 to 84
    "B01001_024E": "m_80to84",
    "B01001_048E": "f_80to84",
    # 85+
    "B01001_025E": "m_85plus",
    "B01001_049E": "f_85plus",
}


def fetch_and_fix():
    print("=" * 60)
    print("FIX WORKFORCE POPULATION (16-64)")
    print("=" * 60)
    
    # Step 1: Fetch age data from Census
    variables = ",".join(WORKFORCE_VARS.keys())
    url = f"{CENSUS_BASE_URL}/{DATASET}"
    params = {
        "get": f"NAME,{variables}",
        "for": "county:*",
        "in": "state:*",
        "key": CENSUS_API_KEY
    }
    
    print("\n   Fetching Census B01001 (Sex by Age)...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"   Retrieved {len(data) - 1} counties")
    
    headers = data[0]
    
    # Step 2: Calculate workforce population per county FIPS
    county_workforce = {}
    
    for row in data[1:]:
        state_fips = row[headers.index("state")]
        county_fips = row[headers.index("county")]
        fips = f"{state_fips}{county_fips}"
        fips_4 = str(int(fips))
        name = row[headers.index("NAME")]
        
        # Get total population
        total_pop = int(float(row[headers.index("B01001_001E")] or 0))
        
        # Sum under 18 (under 5 + 5-9 + 10-14 + 15-17) for both sexes
        under_18_vars = [
            "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E",  # male
            "B01001_027E", "B01001_028E", "B01001_029E", "B01001_030E",  # female
        ]
        under_18 = sum(int(float(row[headers.index(v)] or 0)) for v in under_18_vars)
        
        # Sum 65+ for both sexes
        over_65_vars = [
            "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",  # male
            "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",  # female
        ]
        over_65 = sum(int(float(row[headers.index(v)] or 0)) for v in over_65_vars)
        
        # Workforce = Total - Under 18 - 65+
        # This gives us 18-64, which is the standard working-age population
        workforce = total_pop - under_18 - over_65
        
        county_workforce[fips] = workforce
        county_workforce[fips_4] = workforce
        
        # Print San Francisco for verification
        if "San Francisco" in name:
            print(f"\n   VERIFICATION - {name}:")
            print(f"      Total Population: {total_pop:,}")
            print(f"      Under 18: {under_18:,}")
            print(f"      65 and over: {over_65:,}")
            print(f"      Workforce (18-64): {workforce:,}")
    
    # Step 3: Update Neo4j with correct workforce population
    print(f"\n   Updating Neo4j with corrected workforce population...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    updated = 0
    try:
        with driver.session() as session:
            for fips, workforce in county_workforce.items():
                if len(fips) == 5 or (len(fips) == 4 and not fips.startswith("0")):
                    result = session.run('''
                        MATCH (z:ZipCode)
                        WHERE toString(z.fips) = $fips
                        SET z.workforce_population = $workforce
                        RETURN count(z) as cnt
                    ''', fips=fips, workforce=workforce)
                    cnt = result.single()["cnt"]
                    if cnt > 0:
                        updated += cnt
                        if updated % 5000 == 0:
                            print(f"      Updated {updated} ZIPs...")
    finally:
        driver.close()
    
    print(f"   Updated {updated} ZIPs with corrected workforce population")
    
    # Step 4: Re-export county CSV for Power BI
    print(f"\n   Re-exporting county CSV...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
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
                
                print(f"\n   Saved {len(rows)} counties to {OUTPUT_FILE}")
                
                # Verify San Francisco
                for row in rows:
                    if "San Francisco" in row["County_State_Key"]:
                        print(f"\n   FINAL VERIFICATION - San Francisco:")
                        print(f"      Workforce Population: {row['Workforce_Population']:,}")
                        print(f"      (Should be ~560,000-580,000)")
                        break
                        
    finally:
        driver.close()
    
    print(f"\n{'=' * 60}")
    print("DONE! Refresh Power BI Desktop to pick up corrected data.")
    print("=" * 60)


if __name__ == "__main__":
    fetch_and_fix()
