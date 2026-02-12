"""
Census API Integration for MWR Tool
Fetches demographic data (population, education, income) by county
and updates Neo4j database

Updated: 2026-02-12
- Added 5-level educational detail (Dan's Feb 10 request)
- Added workforce population ages 16-65 (Dan's Feb 10 request)
- Median age already included
- Calculates education percentages for Power BI display
"""

import requests
import json
import time
import os
from neo4j import GraphDatabase

# ================================================================
# CONFIGURATION
# ================================================================
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "ddc5085eb0dfb4fea8367baa28a5b94345e8daed")
CENSUS_BASE_URL = "https://api.census.gov/data"

# ACS 5-Year estimates (most reliable for county-level data)
DATASET = "2022/acs/acs5"  # Latest available 5-year estimates

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

# ================================================================
# Census variables to fetch
# See: https://api.census.gov/data/2022/acs/acs5/variables.html
# ================================================================
CENSUS_VARIABLES = {
    # --- Population & Demographics ---
    "B01003_001E": "total_population",          # Total Population
    "B01002_001E": "median_age",                # Median Age
    "B19013_001E": "median_household_income",   # Median Household Income
    "B25077_001E": "median_home_value",         # Median Home Value

    # --- Employment ---
    "B23025_002E": "labor_force_count",         # In Labor Force
    "B23025_005E": "unemployed_count",          # Unemployed (civilian labor force)

    # --- Education (B15003 = Educational Attainment, 25+ population) ---
    "B15003_001E": "edu_total_25plus",          # Total population 25+
    # No high school diploma (sum of B15003_002 through B15003_016)
    "B15003_002E": "edu_no_school",             # No schooling completed
    "B15003_003E": "edu_nursery",               # Nursery school
    "B15003_004E": "edu_kindergarten",          # Kindergarten
    "B15003_005E": "edu_1st_grade",             # 1st grade
    "B15003_006E": "edu_2nd_grade",             # 2nd grade
    "B15003_007E": "edu_3rd_grade",             # 3rd grade
    "B15003_008E": "edu_4th_grade",             # 4th grade
    "B15003_009E": "edu_5th_grade",             # 5th grade
    "B15003_010E": "edu_6th_grade",             # 6th grade
    "B15003_011E": "edu_7th_grade",             # 7th grade
    "B15003_012E": "edu_8th_grade",             # 8th grade
    "B15003_013E": "edu_9th_grade",             # 9th grade
    "B15003_014E": "edu_10th_grade",            # 10th grade
    "B15003_015E": "edu_11th_grade",            # 11th grade
    "B15003_016E": "edu_12th_no_diploma",       # 12th grade, no diploma
    # High school diploma
    "B15003_017E": "edu_hs_diploma",            # Regular high school diploma
    "B15003_018E": "edu_ged",                   # GED or alternative credential
    # Some college / Associate's
    "B15003_019E": "edu_some_college_lt1yr",    # Some college, less than 1 year
    "B15003_020E": "edu_some_college_1yr_plus", # Some college, 1+ years, no degree
    "B15003_021E": "edu_associates",            # Associate's degree
    # Bachelor's
    "B15003_022E": "edu_bachelors",             # Bachelor's degree
    # Graduate
    "B15003_023E": "edu_masters",               # Master's degree
    "B15003_024E": "edu_professional",          # Professional school degree
    "B15003_025E": "edu_doctorate",             # Doctorate degree

    # --- Workforce Population by Age (B23001) ---
    # Male 16-64 in labor force + Female 16-64 in labor force
    # We'll use total population 16+ and subtract 65+
    "B23001_001E": "pop_16_plus_total",         # Total population 16+
    # We need 65+ to subtract for 16-64 workforce
    "B09021_001E": "pop_65_plus",               # Population 65 and over (alternative)
}

# We'll also make a separate API call for age 16-64 workforce
# Using B23001 detailed table
WORKFORCE_VARIABLES = {
    "B23001_001E": "pop_16_plus_total",
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
    """Parse Census API response into county records with 5-level education"""

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
                    if "count" in our_name or "population" in our_name or "edu_" in our_name or "pop_" in our_name:
                        record[our_name] = int(float(value))
                    elif "income" in our_name or "value" in our_name:
                        record[our_name] = int(float(value)) if float(value) > 0 else None
                    else:
                        record[our_name] = float(value) if float(value) > 0 else None
                except (ValueError, TypeError):
                    record[our_name] = None

        # ============================================================
        # Calculate derived metrics
        # ============================================================

        # --- Unemployment Rate ---
        if record.get("labor_force_count") and record.get("unemployed_count"):
            if record["labor_force_count"] > 0:
                record["census_unemployment_rate"] = round(
                    (record["unemployed_count"] / record["labor_force_count"]) * 100, 1
                )

        # --- 5-Level Education Breakdown ---
        edu_total = record.get("edu_total_25plus")
        if edu_total and edu_total > 0:
            # Level 1: No diploma (B15003_002 through B15003_016)
            no_diploma_fields = [
                "edu_no_school", "edu_nursery", "edu_kindergarten",
                "edu_1st_grade", "edu_2nd_grade", "edu_3rd_grade",
                "edu_4th_grade", "edu_5th_grade", "edu_6th_grade",
                "edu_7th_grade", "edu_8th_grade", "edu_9th_grade",
                "edu_10th_grade", "edu_11th_grade", "edu_12th_no_diploma"
            ]
            no_diploma_count = sum(filter(None, [record.get(f) for f in no_diploma_fields]))

            # Level 2: High school diploma/GED (B15003_017 + B15003_018)
            hs_count = sum(filter(None, [record.get("edu_hs_diploma"), record.get("edu_ged")]))

            # Level 3: Some college / Associate's (B15003_019 + B15003_020 + B15003_021)
            some_college_count = sum(filter(None, [
                record.get("edu_some_college_lt1yr"),
                record.get("edu_some_college_1yr_plus"),
                record.get("edu_associates")
            ]))

            # Level 4: Bachelor's degree (B15003_022)
            bachelors_count = record.get("edu_bachelors") or 0

            # Level 5: Graduate degree (B15003_023 + B15003_024 + B15003_025)
            graduate_count = sum(filter(None, [
                record.get("edu_masters"),
                record.get("edu_professional"),
                record.get("edu_doctorate")
            ]))

            # Store counts
            record["edu_no_diploma_count"] = no_diploma_count
            record["edu_hs_diploma_count"] = hs_count
            record["edu_some_college_count"] = some_college_count
            record["edu_bachelors_count"] = bachelors_count
            record["edu_graduate_count"] = graduate_count

            # Calculate percentages
            record["pct_no_diploma"] = round((no_diploma_count / edu_total) * 100, 1)
            record["pct_hs_diploma"] = round((hs_count / edu_total) * 100, 1)
            record["pct_some_college"] = round((some_college_count / edu_total) * 100, 1)
            record["pct_bachelors"] = round((bachelors_count / edu_total) * 100, 1)
            record["pct_graduate"] = round((graduate_count / edu_total) * 100, 1)

            # Also keep legacy college_educated_count for backward compatibility
            record["college_educated_count"] = bachelors_count + graduate_count

        # --- Workforce Population (16-64 estimate) ---
        pop_16_plus = record.get("pop_16_plus_total")
        pop_65_plus = record.get("pop_65_plus")
        if pop_16_plus and pop_65_plus:
            record["workforce_population"] = pop_16_plus - pop_65_plus
        elif pop_16_plus:
            # If we don't have 65+ data, use 16+ as approximation
            record["workforce_population"] = pop_16_plus

        records.append(record)

    return records


def update_neo4j(records):
    """Update Neo4j ZipCode nodes with Census demographic data"""

    print(f"\n📊 Updating Neo4j with demographic data...")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    update_date = "2026-02-12"
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
                        z.census_updated = $update_date,
                        z.edu_no_diploma_count = $edu_no_diploma_count,
                        z.edu_hs_diploma_count = $edu_hs_diploma_count,
                        z.edu_some_college_count = $edu_some_college_count,
                        z.edu_bachelors_count = $edu_bachelors_count,
                        z.edu_graduate_count = $edu_graduate_count,
                        z.pct_no_diploma = $pct_no_diploma,
                        z.pct_hs_diploma = $pct_hs_diploma,
                        z.pct_some_college = $pct_some_college,
                        z.pct_bachelors = $pct_bachelors,
                        z.pct_graduate = $pct_graduate,
                        z.workforce_population = $workforce_population
                    RETURN count(z) as cnt
                ''',
                    fips_5=record["fips"],
                    fips_4=record["fips_4"],
                    total_population=record.get("total_population"),
                    median_income=record.get("median_household_income"),
                    median_age=record.get("median_age"),
                    median_home_value=record.get("median_home_value"),
                    college_educated=record.get("college_educated_count"),
                    update_date=update_date,
                    edu_no_diploma_count=record.get("edu_no_diploma_count"),
                    edu_hs_diploma_count=record.get("edu_hs_diploma_count"),
                    edu_some_college_count=record.get("edu_some_college_count"),
                    edu_bachelors_count=record.get("edu_bachelors_count"),
                    edu_graduate_count=record.get("edu_graduate_count"),
                    pct_no_diploma=record.get("pct_no_diploma"),
                    pct_hs_diploma=record.get("pct_hs_diploma"),
                    pct_some_college=record.get("pct_some_college"),
                    pct_bachelors=record.get("pct_bachelors"),
                    pct_graduate=record.get("pct_graduate"),
                    workforce_population=record.get("workforce_population"),
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
                    u.dataset = $dataset,
                    u.education_levels = 5,
                    u.has_workforce_population = true
            ''', date=update_date, counties=updated_counties, zips=updated_zips, dataset=DATASET)

    finally:
        driver.close()

    return updated_counties, updated_zips


def main():
    print("\n" + "=" * 60)
    print("CENSUS DEMOGRAPHICS INTEGRATION")
    print("=" * 60)
    print("  New fields: 5-level education, workforce population")

    # Step 1: Fetch data from Census API
    raw_data = fetch_census_data()

    if not raw_data:
        print("\n❌ Failed to fetch Census data")
        return

    # Step 2: Parse the data
    print(f"\n📊 Parsing Census data...")
    records = parse_census_data(raw_data)
    print(f"   Parsed {len(records)} county records")

    # Show sample
    print(f"\n🔍 Sample data:")
    for record in records[:3]:
        print(f"   {record['name']}")
        print(f"      Population: {record.get('total_population', 'N/A'):,}")
        print(f"      Median Income: ${record.get('median_household_income', 'N/A'):,}")
        print(f"      Median Age: {record.get('median_age', 'N/A')}")
        print(f"      Workforce Pop (16-64): {record.get('workforce_population', 'N/A'):,}")
        print(f"      Education (25+ pop):")
        print(f"        No Diploma:    {record.get('pct_no_diploma', 'N/A')}%")
        print(f"        HS Diploma:    {record.get('pct_hs_diploma', 'N/A')}%")
        print(f"        Some College:  {record.get('pct_some_college', 'N/A')}%")
        print(f"        Bachelor's:    {record.get('pct_bachelors', 'N/A')}%")
        print(f"        Graduate:      {record.get('pct_graduate', 'N/A')}%")

    # Step 3: Update Neo4j
    updated_counties, updated_zips = update_neo4j(records)

    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Counties processed: {len(records)}")
    print(f"   Counties matched:   {updated_counties}")
    print(f"   ZIP codes updated:  {updated_zips}")
    print(f"   Update date:        2026-02-12")
    print(f"   New fields added:")
    print(f"     - pct_no_diploma, pct_hs_diploma, pct_some_college")
    print(f"     - pct_bachelors, pct_graduate")
    print(f"     - workforce_population (ages 16-64)")
    print(f"\n✅ Census demographics integration complete!")


if __name__ == "__main__":
    main()
