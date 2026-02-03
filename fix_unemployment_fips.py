"""
Fix: Update unemployment data for counties with 4-digit FIPS codes
Handles leading zeros properly (e.g., 8013 -> 08013 for Colorado)
"""
from neo4j import GraphDatabase
import requests
import json
import time

# Configuration
BLS_API_KEY = "32f25c394fc94565a5f2af71abc3e62e"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

NEO4J_URI = "neo4j+s://551c1b37.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g"

def fetch_unemployment_batch(series_ids, start_year=2024, end_year=2026):
    """Fetch unemployment data from BLS API."""
    headers = {"Content-type": "application/json"}
    data = json.dumps({
        "seriesid": series_ids[:50],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": BLS_API_KEY
    })
    
    try:
        response = requests.post(BLS_API_URL, data=data, headers=headers)
        return response.json()
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def parse_latest_rates(api_response):
    """Parse BLS response to get latest rates."""
    rates = {}
    if not api_response or "Results" not in api_response:
        return rates
    
    for series in api_response["Results"]["series"]:
        series_id = series["seriesID"]
        for data_point in series["data"]:
            value_str = data_point["value"]
            if value_str != "-" and value_str != "":
                try:
                    rates[series_id] = float(value_str)
                    break
                except ValueError:
                    continue
    return rates

def main():
    print("=" * 60)
    print("Fixing Unemployment Data for 4-digit FIPS Counties")
    print("=" * 60)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Find counties WITHOUT unemployment data (4-digit FIPS)
            print("\n📍 Finding counties missing unemployment data...")
            result = session.run('''
                MATCH (z:ZipCode)
                WHERE z.unemployment_rate IS NULL 
                  AND z.fips IS NOT NULL
                  AND size(toString(z.fips)) <= 4
                WITH DISTINCT toString(z.fips) AS fips, z.county AS county, z.state AS state
                RETURN fips, county, state
                ORDER BY state, county
            ''')
            
            missing_counties = []
            for r in result:
                missing_counties.append({
                    'fips': str(r['fips']),
                    'county': r['county'],
                    'state': r['state']
                })
            
            print(f"  Found {len(missing_counties)} counties with 4-digit FIPS codes")
            
            if not missing_counties:
                print("  ✅ No missing counties found!")
                return
            
            # Build BLS series IDs with proper 5-digit FIPS
            series_mapping = {}
            for county in missing_counties:
                fips_4 = county['fips']
                # Pad to 5 digits: 8013 -> 08013
                fips_5 = fips_4.zfill(5)
                state_fips = fips_5[:2]
                county_fips = fips_5[2:]
                
                series_id = f"LAUCN{state_fips}{county_fips}0000000003"
                series_mapping[series_id] = {
                    'fips_4': fips_4,
                    'fips_5': fips_5,
                    'county': county['county'],
                    'state': county['state']
                }
            
            # Fetch from BLS in batches
            print(f"\n📡 Fetching unemployment data from BLS...")
            all_series = list(series_mapping.keys())
            all_rates = {}
            batch_size = 50
            
            for i in range(0, len(all_series), batch_size):
                batch = all_series[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(all_series) + batch_size - 1) // batch_size
                
                print(f"  Batch {batch_num}/{total_batches}...", end=" ")
                response = fetch_unemployment_batch(batch)
                
                if response:
                    rates = parse_latest_rates(response)
                    all_rates.update(rates)
                    print(f"✅ Got {len(rates)} rates")
                else:
                    print("❌ Failed")
                
                time.sleep(1)  # Rate limiting
            
            print(f"\n  Retrieved {len(all_rates)} unemployment rates")
            
            # Update Neo4j
            print(f"\n📝 Updating Neo4j...")
            update_date = "2026-02-03"
            updated_count = 0
            zip_count = 0
            
            for series_id, rate in all_rates.items():
                info = series_mapping.get(series_id)
                if info:
                    # Update using the 4-digit FIPS that's in the database
                    result = session.run('''
                        MATCH (z:ZipCode)
                        WHERE toString(z.fips) = $fips
                        SET z.unemployment_rate = $rate,
                            z.unemployment_updated = $update_date
                        RETURN count(z) as cnt
                    ''', fips=info['fips_4'], rate=rate, update_date=update_date)
                    
                    cnt = result.single()['cnt']
                    if cnt > 0:
                        updated_count += 1
                        zip_count += cnt
                        print(f"  ✅ {info['county']}, {info['state']}: {rate}% ({cnt} ZIPs)")
            
            print(f"\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"  Counties updated: {updated_count}")
            print(f"  ZIP codes updated: {zip_count}")
            print(f"\n✅ Fix complete!")
            
    finally:
        driver.close()

if __name__ == "__main__":
    main()
