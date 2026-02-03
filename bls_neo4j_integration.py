"""
BLS Unemployment Data → Neo4j Integration
Fetches county-level unemployment rates from BLS API and updates AuraDB

Author: Michel Pierre-Louis
Date: February 2026
"""

import requests
import json
import os
from datetime import datetime
from neo4j import GraphDatabase
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# BLS API Configuration
BLS_API_KEY = os.getenv("BLS_API_KEY", "32f25c394fc94565a5f2af71abc3e62e")
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# AuraDB Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

# =============================================================================
# STATE FIPS CODE MAPPING
# =============================================================================

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56", "PR": "72"
}

# =============================================================================
# BLS API FUNCTIONS
# =============================================================================

def get_county_unemployment_series_id(state_fips: str, county_fips: str) -> str:
    """
    Generate BLS series ID for county unemployment rate.
    Format: LAUCN + state_fips(2) + county_fips(3) + 0000000003
    """
    return f"LAUCN{state_fips}{county_fips}0000000003"


def fetch_unemployment_batch(series_ids: list, start_year: int = 2024, end_year: int = 2026) -> dict:
    """
    Fetch unemployment data from BLS API for a batch of series (max 50).
    """
    headers = {"Content-type": "application/json"}
    
    data = json.dumps({
        "seriesid": series_ids[:50],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": BLS_API_KEY
    })
    
    try:
        response = requests.post(BLS_API_URL, data=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "REQUEST_SUCCEEDED":
            return result
        else:
            print(f"  ⚠️ BLS API warning: {result.get('message', 'Unknown')}")
            return result  # May still have partial data
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request failed: {e}")
        return None


def parse_latest_rates(api_response: dict) -> dict:
    """
    Parse BLS API response and return dict of series_id -> latest unemployment rate.
    """
    rates = {}
    
    if not api_response or "Results" not in api_response:
        return rates
    
    for series in api_response["Results"]["series"]:
        series_id = series["seriesID"]
        
        # Find most recent valid data point
        for data_point in series["data"]:
            value_str = data_point["value"]
            if value_str != "-" and value_str != "":
                try:
                    rates[series_id] = float(value_str)
                    break  # Got the latest, move to next series
                except ValueError:
                    continue
    
    return rates


# =============================================================================
# NEO4J FUNCTIONS
# =============================================================================

def get_all_counties_from_neo4j(driver) -> list:
    """
    Get all unique county + state combinations from Neo4j ZipCode nodes.
    Returns list of dicts with county, state, and fips info.
    """
    query = """
    MATCH (z:ZipCode)
    WHERE z.county IS NOT NULL AND z.state IS NOT NULL AND z.fips IS NOT NULL
    WITH DISTINCT z.county AS county, z.state AS state, z.fips AS fips
    RETURN county, state, fips
    ORDER BY state, county
    """
    
    with driver.session() as session:
        result = session.run(query)
        counties = []
        seen = set()
        
        for record in result:
            county = record["county"]
            state = record["state"]
            fips = record["fips"]
            
            # Extract county FIPS (last 3 digits of 5-digit FIPS)
            if fips and len(fips) >= 5:
                county_fips = fips[-3:]
                state_fips = fips[:2]
                
                key = f"{state_fips}-{county_fips}"
                if key not in seen:
                    seen.add(key)
                    counties.append({
                        "county": county,
                        "state": state,
                        "state_fips": state_fips,
                        "county_fips": county_fips,
                        "full_fips": fips
                    })
        
        return counties


def update_county_unemployment_in_neo4j(driver, state_fips: str, county_fips: str, 
                                         unemployment_rate: float, update_date: str):
    """
    Update all ZipCode nodes in a county with the unemployment rate.
    """
    full_fips = f"{state_fips}{county_fips}"
    
    query = """
    MATCH (z:ZipCode)
    WHERE z.fips = $fips OR z.fips STARTS WITH $fips_prefix
    SET z.unemployment_rate = $rate,
        z.unemployment_updated = $update_date
    RETURN count(z) as updated_count
    """
    
    with driver.session() as session:
        result = session.run(query, 
                           fips=full_fips,
                           fips_prefix=full_fips,
                           rate=unemployment_rate,
                           update_date=update_date)
        record = result.single()
        return record["updated_count"] if record else 0


def create_unemployment_summary_in_neo4j(driver, update_date: str, total_counties: int, 
                                          successful_updates: int):
    """
    Create or update a summary node tracking unemployment data updates.
    """
    query = """
    MERGE (u:UnemploymentUpdate {type: 'BLS_LAUS'})
    SET u.last_updated = $update_date,
        u.total_counties = $total_counties,
        u.successful_updates = $successful_updates,
        u.data_source = 'Bureau of Labor Statistics - Local Area Unemployment Statistics'
    RETURN u
    """
    
    with driver.session() as session:
        session.run(query, 
                   update_date=update_date,
                   total_counties=total_counties,
                   successful_updates=successful_updates)


# =============================================================================
# MAIN INTEGRATION
# =============================================================================

def update_all_unemployment_data():
    """
    Main function: Fetch unemployment data from BLS and update Neo4j.
    """
    print("=" * 70)
    print("BLS Unemployment Data → Neo4j Integration")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Connect to Neo4j
    print("📊 Connecting to AuraDB...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        print("  ✅ Connected to AuraDB")
        print()
        
        # Get all counties from Neo4j
        print("📍 Fetching counties from Neo4j...")
        counties = get_all_counties_from_neo4j(driver)
        print(f"  ✅ Found {len(counties)} unique counties")
        print()
        
        # Build series IDs for BLS API
        series_mapping = {}  # series_id -> county info
        for county in counties:
            series_id = get_county_unemployment_series_id(
                county["state_fips"], 
                county["county_fips"]
            )
            series_mapping[series_id] = county
        
        all_series_ids = list(series_mapping.keys())
        
        # Fetch data in batches (BLS API limit: 50 series per request)
        print(f"📡 Fetching unemployment data from BLS API...")
        print(f"  Processing {len(all_series_ids)} series in batches of 50...")
        
        all_rates = {}
        batch_size = 50
        
        for i in range(0, len(all_series_ids), batch_size):
            batch = all_series_ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_series_ids) + batch_size - 1) // batch_size
            
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} series)...", end=" ")
            
            response = fetch_unemployment_batch(batch)
            if response:
                rates = parse_latest_rates(response)
                all_rates.update(rates)
                print(f"✅ Got {len(rates)} rates")
            else:
                print("❌ Failed")
            
            # Rate limiting: wait between batches
            if i + batch_size < len(all_series_ids):
                time.sleep(1)  # 1 second delay between batches
        
        print()
        print(f"  ✅ Retrieved unemployment rates for {len(all_rates)} counties")
        print()
        
        # Update Neo4j with unemployment data
        print("📝 Updating Neo4j with unemployment data...")
        update_date = datetime.now().strftime("%Y-%m-%d")
        successful_updates = 0
        total_zips_updated = 0
        
        for series_id, rate in all_rates.items():
            county_info = series_mapping.get(series_id)
            if county_info:
                zips_updated = update_county_unemployment_in_neo4j(
                    driver,
                    county_info["state_fips"],
                    county_info["county_fips"],
                    rate,
                    update_date
                )
                if zips_updated > 0:
                    successful_updates += 1
                    total_zips_updated += zips_updated
        
        # Create summary node
        create_unemployment_summary_in_neo4j(driver, update_date, len(counties), successful_updates)
        
        print(f"  ✅ Updated {successful_updates} counties")
        print(f"  ✅ Updated {total_zips_updated} ZIP code nodes")
        print()
        
        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Total counties in database:    {len(counties)}")
        print(f"  Unemployment rates retrieved:  {len(all_rates)}")
        print(f"  Counties successfully updated: {successful_updates}")
        print(f"  ZIP codes updated:             {total_zips_updated}")
        print(f"  Update date:                   {update_date}")
        print()
        print("✅ Unemployment data integration complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
        
    finally:
        driver.close()


if __name__ == "__main__":
    update_all_unemployment_data()
