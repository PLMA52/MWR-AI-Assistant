"""
Census API Integration for MWR Tool
Fetches demographic data (population, education, income) by county
and updates Neo4j database
"""

import requests
import json
import time
import os
from neo4j import GraphDatabase

# ============================================================
# CONFIGURATION
# ============================================================
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "ddc5085eb0dfb4fea8367baa28a5b94345e8daed")
CENSUS_BASE_URL = "https://api.census.gov/data"

# ACS 5-Year estimates (most reliable for county-level data)
DATASET = "2022/acs/acs5"  # Latest available 5-year estimates

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

# Census variables to fetch
# See: https://api.census.gov/data/2022/acs/acs5/variables.html
CENSUS_VARIABLES = {
    "B01003_001E": "total_population",           # Total Population
    "B19013_001E": "median_household_income",    # Median Household Income
    "B23025_005E": "unemployed_count",           # Unemployed (civilian labor force)
    "B23025_002E": "labor_force_count",          # In Labor Force
    "B15003_022E": "bachelors_degree_count",     # Bachelor's degree
    "B15003_023E": "masters_degree_count",       # Master's degree
    "B15003_024E": "professional_degree_count",  # Professional school degree
    "B15003_025E": "doctorate_degree_count",     # Doctorate degree
    "B01002_001E": "median_age",                 # Median Age
    "B25077_001E": "median_home_value",          # Median Home Value
}


def fetch_census_data():
    """Fetch demographic data for all US counties from Census API"""
    
    print("=" * 60)
    print("CENSUS API - Fetching Demographic Data")
    print("=" * 60)
    
    # Build variable list for API call
    variables = ",".join(CENSUS_VARIABLES.keys())
    
    # Fetch all counties (county:* in state:*)
    url = f"{CENSUS_BASE_URL}/{DATASET}"
    params = {
        "get": f"NAME,{variables}",
        "for": "county:*",
        "in": "state:*",
        "key": CENSUS_API_KEY
    }
    
    print(f"\n📡 Calling Census API...")
    print(f"   Dataset: ACS 5-Year ({DATASET})")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"   ✅ Retrieved {len(data) - 1} counties")  # -1 for header row
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ API Error: {e}")
        return None


def parse_census_data(raw_data):
    """Parse Census API response into county records"""
    
    if not raw_data or len(raw_data) < 2:
        return []
    
    # First row is headers
    headers = raw_data[0]
    records = []
    
    for row in raw_data[1:]:
        record = {}
        
        # Get FIPS code (state + county)
        state_fips = row[headers.index("state")]
        county_fips = row[headers.index("county")]
        record["fips"] = f"{state_fips}{county_fips}"
        record["fips_4"] = str(int(record["fips"]))  # Remove leading zeros for matching
        
        # Get county name
        record["name"] = row[headers.index("NAME")]
        
        # Get all variables
        for census_var, our_name in CENSUS_VARIABLES.items():
            idx = headers.index(census_var)
            value = row[idx]
            
            # Handle null/missing values
            if value is None or value == "" or value == "-":
                record[our_name] = None
            else:
                try:
                    # Convert to appropriate type
                    if "count" in our_name or "population" in our_name:
                        record[our_name] = int(float(value))
                    elif "income" in our_name or "value" in our_name:
                        record[our_name] = int(float(value)) if float(value) > 0 else None
                    else:
                        record[our_name] = float(value) if float(value) > 0 else None
                except (ValueError, TypeError):
                    record[our_name] = None
        
        # Calculate derived metrics
        if record.get("labor_force_count") and record.get("unemployed_count"):
            if record["labor_force_count"] > 0:
                record["census_unemployment_rate"] = round(
                    (record["unemployed_count"] / record["labor_force_count"]) * 100, 1
                )
        
        # Calculate education rate (% with bachelor's or higher)
        college_educated = sum(filter(None, [
            record.get("bachelors_degree_count"),
            record.get("masters_degree_count"),
            record.get("professional_degree_count"),
            record.get("doctorate_degree_count")
        ]))
        
        if record.get("total_population") and record["total_population"] > 0 and college_educated:
            # Note: This is approximate - should be % of 25+ population
            record["college_educated_count"] = college_educated
        
        records.append(record)
    
    return records


def update_neo4j(records):
    """Update Neo4j ZipCode nodes with Census demographic data"""
    
    print(f"\n📝 Updating Neo4j with demographic data...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    update_date = "2026-02-04"
    updated_counties = 0
    updated_zips = 0
    
    try:
        with driver.session() as session:
            for record in records:
                # Try both FIPS formats (with and without leading zeros)
                result = session.run('''
                    MATCH (z:ZipCode)
                    WHERE toString(z.fips) = $fips_5 OR toString(z.fips) = $fips_4
                    SET z.total_population = $total_population,
                        z.median_household_income = $median_income,
                        z.median_age = $median_age,
                        z.median_home_value = $median_home_value,
                        z.college_educated_count = $college_educated,
                        z.census_updated = $update_date
                    RETURN count(z) as cnt
                ''', 
                    fips_5=record["fips"],
                    fips_4=record["fips_4"],
                    total_population=record.get("total_population"),
                    median_income=record.get("median_household_income"),
                    median_age=record.get("median_age"),
                    median_home_value=record.get("median_home_value"),
                    college_educated=record.get("college_educated_count"),
                    update_date=update_date
                )
                
                cnt = result.single()["cnt"]
                if cnt > 0:
                    updated_counties += 1
                    updated_zips += cnt
                    
                    if updated_counties % 500 == 0:
                        print(f"   Updated {updated_counties} counties ({updated_zips} ZIPs)...")
            
            # Create update tracking node
            session.run('''
                MERGE (u:CensusUpdate {date: $date})
                SET u.counties_updated = $counties,
                    u.zips_updated = $zips,
                    u.dataset = $dataset
            ''', date=update_date, counties=updated_counties, zips=updated_zips, dataset=DATASET)
    
    finally:
        driver.close()
    
    return updated_counties, updated_zips


def main():
    print("\n" + "=" * 60)
    print("CENSUS DEMOGRAPHICS INTEGRATION")
    print("=" * 60)
    
    # Step 1: Fetch data from Census API
    raw_data = fetch_census_data()
    
    if not raw_data:
        print("\n❌ Failed to fetch Census data")
        return
    
    # Step 2: Parse the data
    print(f"\n🔄 Parsing Census data...")
    records = parse_census_data(raw_data)
    print(f"   Parsed {len(records)} county records")
    
    # Show sample
    print(f"\n📊 Sample data:")
    for record in records[:3]:
        print(f"   {record['name']}")
        print(f"      Population: {record.get('total_population', 'N/A'):,}")
        print(f"      Median Income: ${record.get('median_household_income', 'N/A'):,}")
        print(f"      Median Age: {record.get('median_age', 'N/A')}")
    
    # Step 3: Update Neo4j
    updated_counties, updated_zips = update_neo4j(records)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Counties processed: {len(records)}")
    print(f"   Counties matched:   {updated_counties}")
    print(f"   ZIP codes updated:  {updated_zips}")
    print(f"   Update date:        2026-02-04")
    print(f"\n✅ Census demographics integration complete!")


if __name__ == "__main__":
    main()
