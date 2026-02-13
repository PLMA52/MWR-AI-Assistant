"""
Census Age Breakout by County
Fetches B01001 (Sex by Age) from Census API at ZIP level,
aggregates to county level, calculates percentage for 8 age groups
matching Dan's format, writes to Neo4j and exports CSV for Power BI.

Age Groups (matching Dan's file):
  0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
"""

import os
import csv
import json
import time
import requests
from neo4j import GraphDatabase

# ============================================================
# CONFIGURATION
# ============================================================
CENSUS_API_KEY = "ceaeb94491153e84470b75899d8367e93debb205"
CENSUS_BASE_URL = "https://api.census.gov/data/2023/acs/acs5"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

OUTPUT_CSV = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data/Census_Age_Breakout_Jan2026.csv")

# ============================================================
# CENSUS B01001 VARIABLE MAPPING
# ============================================================
# B01001: Sex by Age
# Male variables: 003-025, Female variables: 027-049
# We combine male + female into Dan's 8 decade groups

AGE_GROUPS = {
    "age_0_to_9": {
        "male": ["003", "004"],        # Under 5, 5-9
        "female": ["027", "028"]
    },
    "age_10_to_19": {
        "male": ["005", "006", "007"],  # 10-14, 15-17, 18-19
        "female": ["029", "030", "031"]
    },
    "age_20_to_29": {
        "male": ["008", "009", "010", "011"],  # 20, 21, 22-24, 25-29
        "female": ["032", "033", "034", "035"]
    },
    "age_30_to_39": {
        "male": ["012", "013"],         # 30-34, 35-39
        "female": ["036", "037"]
    },
    "age_40_to_49": {
        "male": ["014", "015"],         # 40-44, 45-49
        "female": ["038", "039"]
    },
    "age_50_to_59": {
        "male": ["016", "017"],         # 50-54, 55-59
        "female": ["040", "041"]
    },
    "age_60_to_69": {
        "male": ["018", "019", "020", "021"],  # 60-61, 62-64, 65-66, 67-69
        "female": ["042", "043", "044", "045"]
    },
    "age_70_plus": {
        "male": ["022", "023", "024", "025"],  # 70-74, 75-79, 80-84, 85+
        "female": ["046", "047", "048", "049"]
    }
}

# Build flat list of all Census variables needed
ALL_VARS = ["B01001_001E"]  # Total population
for group_name, sexes in AGE_GROUPS.items():
    for var_num in sexes["male"]:
        ALL_VARS.append(f"B01001_{var_num}E")
    for var_num in sexes["female"]:
        ALL_VARS.append(f"B01001_{var_num}E")

# State FIPS codes
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR", "78": "VI", "66": "GU", "69": "MP",
    "60": "AS"
}


def fetch_age_data():
    """Fetch B01001 age data from Census API at ZIP level"""
    print("=" * 60)
    print("CENSUS AGE BREAKOUT - FETCH & PROCESS")
    print("=" * 60)
    
    # Census API has a 50-variable limit per call
    # We need 47 variables, so we can do it in one call
    var_string = ",".join(ALL_VARS)
    print(f"\n   Variables to fetch: {len(ALL_VARS)}")
    
    all_zip_data = []
    
    for state_fips, state_abbr in sorted(STATE_FIPS.items()):
        url = f"{CENSUS_BASE_URL}?get=NAME,{var_string}&for=zip%20code%20tabulation%20area:*&in=state:{state_fips}&key={CENSUS_API_KEY}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                headers = data[0]
                rows = data[1:]
                all_zip_data.extend(rows)
                print(f"   {state_abbr}: {len(rows)} ZIPs")
            else:
                print(f"   {state_abbr}: ERROR {response.status_code}")
        except Exception as e:
            print(f"   {state_abbr}: FAILED - {e}")
        
        time.sleep(0.3)  # Rate limiting
    
    print(f"\n   Total ZIPs fetched: {len(all_zip_data)}")
    
    # Parse headers to get variable positions
    # First call to get headers
    test_url = f"{CENSUS_BASE_URL}?get=NAME,{var_string}&for=zip%20code%20tabulation%20area:*&in=state:01&key={CENSUS_API_KEY}"
    test_response = requests.get(test_url, timeout=30)
    headers = test_response.json()[0]
    
    return headers, all_zip_data


def process_to_county(headers, zip_data):
    """
    We need county mapping. Since Census ZCTA queries don't return county,
    we'll use Neo4j to get the ZIP-to-county mapping, then aggregate.
    """
    print("\n   Getting ZIP-to-county mapping from Neo4j...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    zip_to_county = {}
    
    with driver.session() as session:
        result = session.run("""
            MATCH (z:ZipCode)
            RETURN z.zip as zip, z.county as county, z.state as state
        """)
        for record in result:
            zip_to_county[str(record["zip"])] = {
                "county": record["county"],
                "state": record["state"]
            }
    driver.close()
    
    print(f"   ZIP-to-county mappings loaded: {len(zip_to_county)}")
    
    # Build variable index from headers
    var_index = {}
    for i, h in enumerate(headers):
        var_index[h] = i
    
    # Parse ZIP-level data and aggregate to county
    county_data = {}  # key: (county, state) -> {age_group: sum, ...}
    matched = 0
    unmatched = 0
    
    for row in zip_data:
        zip_code = row[var_index.get("zip code tabulation area", len(row)-1)]
        
        # Pad ZIP to 5 digits
        zip_code = str(zip_code).zfill(5)
        
        if zip_code not in zip_to_county:
            unmatched += 1
            continue
        
        matched += 1
        county = zip_to_county[zip_code]["county"]
        state = zip_to_county[zip_code]["state"]
        key = (county, state)
        
        if key not in county_data:
            county_data[key] = {
                "total_pop": 0,
                "age_0_to_9": 0,
                "age_10_to_19": 0,
                "age_20_to_29": 0,
                "age_30_to_39": 0,
                "age_40_to_49": 0,
                "age_50_to_59": 0,
                "age_60_to_69": 0,
                "age_70_plus": 0
            }
        
        # Get total population
        total_val = row[var_index["B01001_001E"]]
        try:
            total_pop = int(total_val) if total_val and int(total_val) >= 0 else 0
        except (ValueError, TypeError):
            total_pop = 0
        
        county_data[key]["total_pop"] += total_pop
        
        # Sum each age group
        for group_name, sexes in AGE_GROUPS.items():
            group_sum = 0
            for var_num in sexes["male"]:
                var_key = f"B01001_{var_num}E"
                if var_key in var_index:
                    val = row[var_index[var_key]]
                    try:
                        v = int(val) if val and int(val) >= 0 else 0
                    except (ValueError, TypeError):
                        v = 0
                    group_sum += v
            for var_num in sexes["female"]:
                var_key = f"B01001_{var_num}E"
                if var_key in var_index:
                    val = row[var_index[var_key]]
                    try:
                        v = int(val) if val and int(val) >= 0 else 0
                    except (ValueError, TypeError):
                        v = 0
                    group_sum += v
            
            county_data[key][group_name] += group_sum
    
    print(f"   ZIPs matched to counties: {matched}")
    print(f"   ZIPs unmatched: {unmatched}")
    print(f"   Counties with data: {len(county_data)}")
    
    # Calculate percentages
    county_results = {}
    for (county, state), data in county_data.items():
        total = data["total_pop"]
        if total == 0:
            continue
        
        county_results[(county, state)] = {
            "total_pop": total,
            "age_0_to_9": data["age_0_to_9"],
            "age_10_to_19": data["age_10_to_19"],
            "age_20_to_29": data["age_20_to_29"],
            "age_30_to_39": data["age_30_to_39"],
            "age_40_to_49": data["age_40_to_49"],
            "age_50_to_59": data["age_50_to_59"],
            "age_60_to_69": data["age_60_to_69"],
            "age_70_plus": data["age_70_plus"],
            "pct_0_to_9": round(data["age_0_to_9"] / total * 100, 1),
            "pct_10_to_19": round(data["age_10_to_19"] / total * 100, 1),
            "pct_20_to_29": round(data["age_20_to_29"] / total * 100, 1),
            "pct_30_to_39": round(data["age_30_to_39"] / total * 100, 1),
            "pct_40_to_49": round(data["age_40_to_49"] / total * 100, 1),
            "pct_50_to_59": round(data["age_50_to_59"] / total * 100, 1),
            "pct_60_to_69": round(data["age_60_to_69"] / total * 100, 1),
            "pct_70_plus": round(data["age_70_plus"] / total * 100, 1),
        }
    
    return county_results


def validate_against_dan(county_results):
    """Quick validation against Dan's file for a known county"""
    print("\n   VALIDATION - Sample counties:")
    test_counties = [
        ("Jefferson", "AL"),
        ("Los Angeles", "CA"),
        ("San Francisco", "CA"),
        ("Boulder", "CO"),
    ]
    for county, state in test_counties:
        if (county, state) in county_results:
            data = county_results[(county, state)]
            print(f"\n   {county}, {state}:")
            print(f"      Total Pop: {data['total_pop']:,}")
            print(f"      0-9: {data['pct_0_to_9']}%  |  10-19: {data['pct_10_to_19']}%  |  20-29: {data['pct_20_to_29']}%  |  30-39: {data['pct_30_to_39']}%")
            print(f"      40-49: {data['pct_40_to_49']}%  |  50-59: {data['pct_50_to_59']}%  |  60-69: {data['pct_60_to_69']}%  |  70+: {data['pct_70_plus']}%")
            pct_sum = (data['pct_0_to_9'] + data['pct_10_to_19'] + data['pct_20_to_29'] + 
                       data['pct_30_to_39'] + data['pct_40_to_49'] + data['pct_50_to_59'] + 
                       data['pct_60_to_69'] + data['pct_70_plus'])
            print(f"      Sum of %: {pct_sum}%")


def export_csv(county_results):
    """Export county-level age data to CSV for Power BI"""
    print(f"\n   Exporting CSV to: {OUTPUT_CSV}")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "County", "State", "Total_Population",
            "Age_0_to_9", "Age_10_to_19", "Age_20_to_29", "Age_30_to_39",
            "Age_40_to_49", "Age_50_to_59", "Age_60_to_69", "Age_70_Plus",
            "Pct_0_to_9", "Pct_10_to_19", "Pct_20_to_29", "Pct_30_to_39",
            "Pct_40_to_49", "Pct_50_to_59", "Pct_60_to_69", "Pct_70_Plus"
        ])
        
        for (county, state), data in sorted(county_results.items()):
            writer.writerow([
                county, state, data["total_pop"],
                data["age_0_to_9"], data["age_10_to_19"], data["age_20_to_29"], data["age_30_to_39"],
                data["age_40_to_49"], data["age_50_to_59"], data["age_60_to_69"], data["age_70_plus"],
                data["pct_0_to_9"], data["pct_10_to_19"], data["pct_20_to_29"], data["pct_30_to_39"],
                data["pct_40_to_49"], data["pct_50_to_59"], data["pct_60_to_69"], data["pct_70_plus"]
            ])
    
    print(f"   Exported {len(county_results)} counties")


def write_to_neo4j(county_results):
    """Write age percentages to Neo4j ZipCode nodes"""
    print(f"\n   Writing age data to Neo4j...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    updated_total = 0
    
    try:
        with driver.session() as session:
            for (county, state), data in county_results.items():
                result = session.run('''
                    MATCH (z:ZipCode)
                    WHERE z.county = $county AND z.state = $state
                    SET z.pct_age_0_to_9 = $p1,
                        z.pct_age_10_to_19 = $p2,
                        z.pct_age_20_to_29 = $p3,
                        z.pct_age_30_to_39 = $p4,
                        z.pct_age_40_to_49 = $p5,
                        z.pct_age_50_to_59 = $p6,
                        z.pct_age_60_to_69 = $p7,
                        z.pct_age_70_plus = $p8
                    RETURN count(z) as cnt
                ''', county=county, state=state,
                   p1=data["pct_0_to_9"], p2=data["pct_10_to_19"],
                   p3=data["pct_20_to_29"], p4=data["pct_30_to_39"],
                   p5=data["pct_40_to_49"], p6=data["pct_50_to_59"],
                   p7=data["pct_60_to_69"], p8=data["pct_70_plus"])
                
                cnt = result.single()["cnt"]
                updated_total += cnt
                
                if updated_total % 5000 == 0 and updated_total > 0:
                    print(f"      Updated {updated_total} ZIPs...")
    finally:
        driver.close()
    
    print(f"   Updated {updated_total} ZIPs with age breakout data")
    
    # Verify
    print(f"\n   VERIFICATION:")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run('''
                MATCH (z:ZipCode)
                WHERE z.county = "San Francisco" AND z.state = "CA"
                RETURN DISTINCT z.county, z.state,
                    z.pct_age_0_to_9, z.pct_age_10_to_19, z.pct_age_20_to_29,
                    z.pct_age_30_to_39, z.pct_age_40_to_49, z.pct_age_50_to_59,
                    z.pct_age_60_to_69, z.pct_age_70_plus
                LIMIT 1
            ''')
            record = result.single()
            if record:
                print(f"   San Francisco, CA:")
                print(f"      0-9: {record['z.pct_age_0_to_9']}%  |  10-19: {record['z.pct_age_10_to_19']}%")
                print(f"      20-29: {record['z.pct_age_20_to_29']}%  |  30-39: {record['z.pct_age_30_to_39']}%")
                print(f"      40-49: {record['z.pct_age_40_to_49']}%  |  50-59: {record['z.pct_age_50_to_59']}%")
                print(f"      60-69: {record['z.pct_age_60_to_69']}%  |  70+: {record['z.pct_age_70_plus']}%")
            
            # Count coverage
            result2 = session.run('''
                MATCH (z:ZipCode) WHERE z.pct_age_0_to_9 IS NOT NULL
                RETURN count(z) as cnt
            ''')
            cnt = result2.single()["cnt"]
            print(f"   Total ZIPs with age data: {cnt}")
    finally:
        driver.close()


def main():
    print("\n" + "=" * 60)
    print("STEP 1: FETCH AGE DATA FROM CENSUS API")
    print("=" * 60)
    headers, zip_data = fetch_age_data()
    
    print("\n" + "=" * 60)
    print("STEP 2: AGGREGATE TO COUNTY LEVEL")
    print("=" * 60)
    county_results = process_to_county(headers, zip_data)
    
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATE")
    print("=" * 60)
    validate_against_dan(county_results)
    
    print("\n" + "=" * 60)
    print("STEP 4: EXPORT CSV FOR POWER BI")
    print("=" * 60)
    export_csv(county_results)
    
    print("\n" + "=" * 60)
    print("STEP 5: WRITE TO NEO4J")
    print("=" * 60)
    write_to_neo4j(county_results)
    
    print("\n" + "=" * 60)
    print("DONE! Age breakout data loaded into Neo4j and exported to CSV.")
    print("Next steps:")
    print("  1. Add CSV to Power BI Inform tab")
    print("  2. Update AI Assistant prompt with age properties")
    print("  3. Push updates to GitHub")
    print("=" * 60)


if __name__ == "__main__":
    main()
