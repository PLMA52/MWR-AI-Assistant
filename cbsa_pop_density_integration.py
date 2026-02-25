"""
Load CBSA classification and Population Density data into Neo4j ZipCode nodes.

Data sources:
- CBSA: April_2025_Zip_Code_to_CBSA_w_Preferred_City_(Edit_Version).xlsx
  Adds: cbsa_code, cbsa_classification, preferred_city, fips
- Population Density: Pop_per_Sq_Footage_-_US_IncomeByZipDemographics.xlsb
  Adds: population_density_sq_mi

Run: python cbsa_pop_density_integration.py
"""

import os
import pandas as pd
from neo4j import GraphDatabase

# ── Config ──────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

# File paths — update these to match your local paths
CBSA_FILE = os.path.join(os.path.dirname(__file__),
    "April_2025_Zip_Code_to_CBSA_w_Preferred_City_(Edit_Version).xlsx")
POP_FILE = os.path.join(os.path.dirname(__file__),
    "Pop_per_Sq_Footage_-_US_IncomeByZipDemographics_(Table_format_with_FIPS).xlsb")

# Alternative paths (OneDrive)
CBSA_FILE_ALT = os.path.expanduser(
    "~/OneDrive - SODEXO/MWR_Automation_Data/April_2025_Zip_Code_to_CBSA_w_Preferred_City_(Edit_Version).xlsx")
POP_FILE_ALT = os.path.expanduser(
    "~/OneDrive - SODEXO/MWR_Automation_Data/Pop_per_Sq_Footage_-_US_IncomeByZipDemographics_(Table_format_with_FIPS).xlsb")


def find_file(primary, alt, description):
    """Try primary path, then alternate, then search Downloads"""
    for p in [primary, alt]:
        if os.path.exists(p):
            print(f"  ✅ Found {description}: {p}")
            return p
    
    # Search in Downloads folder
    downloads = os.path.expanduser("~/Downloads")
    if os.path.isdir(downloads):
        for f in os.listdir(downloads):
            if "CBSA" in f and "Edit" in f and f.endswith(".xlsx") and description == "CBSA":
                path = os.path.join(downloads, f)
                print(f"  ✅ Found {description} in Downloads: {f}")
                return path
            if "Pop_per" in f and f.endswith(".xlsb") and description == "Population":
                path = os.path.join(downloads, f)
                print(f"  ✅ Found {description} in Downloads: {f}")
                return path
    
    # Search current directory
    cwd = os.getcwd()
    for f in os.listdir(cwd):
        if "CBSA" in f and f.endswith(".xlsx") and description == "CBSA":
            path = os.path.join(cwd, f)
            print(f"  ✅ Found {description} in cwd: {f}")
            return path
        if ("Pop_per" in f or "IncomeByZip" in f) and description == "Population":
            path = os.path.join(cwd, f)
            print(f"  ✅ Found {description} in cwd: {f}")
            return path
    
    print(f"  ❌ {description} file not found!")
    return None


def load_cbsa_data(filepath):
    """Load and clean CBSA data"""
    print(f"\n📂 Loading CBSA data...")
    df = pd.read_excel(filepath)
    
    # Clean ZIP codes
    df['zip'] = df['ZIP'].astype(str).str.zfill(5)
    
    # Key columns Dan wants:
    # - CBSA Final or Zip Code (the highest applicable number)
    # - Top Group (Division / CBSA / Non CBSA)
    # Plus useful extras: Preferred City, FIPS
    
    result = df[['zip']].copy()
    result['cbsa_code'] = df['CBSA Final or Zip Code'].astype(str)
    result['cbsa_classification'] = df['Top Group'].fillna('Unknown')
    result['preferred_city'] = df['Preferred City to Zip Code'].fillna('')
    result['fips'] = df['FIPS'].astype(str).str.zfill(5) if 'FIPS' in df.columns else ''
    
    print(f"  Loaded {len(result):,} ZIPs with CBSA data")
    print(f"  Classification breakdown:")
    for cls, cnt in result['cbsa_classification'].value_counts().items():
        print(f"    {cls}: {cnt:,}")
    
    return result


def load_pop_density(filepath):
    """Load population density from the Income by Zip file"""
    print(f"\n📂 Loading population density data...")
    
    try:
        df = pd.read_excel(filepath, sheet_name='IncomeByZipCodeReport', 
                          engine='pyxlsb', usecols=[0, 32])
    except Exception:
        # Try without specifying engine
        df = pd.read_excel(filepath, sheet_name='IncomeByZipCodeReport',
                          usecols=[0, 32])
    
    df.columns = ['zip_raw', 'population_density_sq_mi']
    df['zip'] = df['zip_raw'].apply(lambda x: str(int(x)).zfill(5) if pd.notna(x) else None)
    df = df.dropna(subset=['zip'])
    df['population_density_sq_mi'] = pd.to_numeric(df['population_density_sq_mi'], errors='coerce').fillna(0)
    
    print(f"  Loaded {len(df):,} ZIPs with population density")
    print(f"  Density stats: min={df['population_density_sq_mi'].min():.1f}, "
          f"max={df['population_density_sq_mi'].max():.1f}, "
          f"mean={df['population_density_sq_mi'].mean():.1f}")
    
    return df[['zip', 'population_density_sq_mi']]


def update_neo4j(cbsa_data, pop_data):
    """Write CBSA and population density data to Neo4j ZipCode nodes"""
    print(f"\n🔗 Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # Merge CBSA and population data
    merged = cbsa_data.merge(pop_data, on='zip', how='left')
    merged['population_density_sq_mi'] = merged['population_density_sq_mi'].fillna(0)
    
    print(f"  Merged dataset: {len(merged):,} ZIPs")
    print(f"  With pop density: {(merged['population_density_sq_mi'] > 0).sum():,}")
    
    # Batch update
    batch_size = 500
    total_updated = 0
    
    try:
        with driver.session() as session:
            for start in range(0, len(merged), batch_size):
                batch = merged.iloc[start:start + batch_size]
                
                # Build batch params
                params = []
                for _, row in batch.iterrows():
                    params.append({
                        'zip': row['zip'],
                        'cbsa_code': str(row['cbsa_code']),
                        'cbsa_classification': str(row['cbsa_classification']),
                        'preferred_city': str(row['preferred_city']),
                        'fips': str(row['fips']),
                        'pop_density': round(float(row['population_density_sq_mi']), 1)
                    })
                
                result = session.run("""
                    UNWIND $params AS p
                    MATCH (z:ZipCode {zip: p.zip})
                    SET z.cbsa_code = p.cbsa_code,
                        z.cbsa_classification = p.cbsa_classification,
                        z.preferred_city = p.preferred_city,
                        z.fips = p.fips,
                        z.population_density_sq_mi = p.pop_density
                    RETURN count(z) AS updated
                """, params=params)
                
                count = result.single()["updated"]
                total_updated += count
                
                if (start // batch_size) % 10 == 0:
                    pct = (start + batch_size) / len(merged) * 100
                    print(f"  Progress: {min(pct, 100):.0f}% ({total_updated:,} ZIPs updated)")
        
        print(f"\n✅ Done! Updated {total_updated:,} ZipCode nodes")
        
        # Verify
        with driver.session() as session:
            verify = session.run("""
                MATCH (z:ZipCode) WHERE z.cbsa_classification IS NOT NULL
                RETURN z.cbsa_classification AS cls, count(z) AS cnt
                ORDER BY cnt DESC
            """)
            print(f"\n📊 Verification:")
            for r in verify:
                print(f"  {r['cls']}: {r['cnt']:,} ZIPs")
            
            pop_check = session.run("""
                MATCH (z:ZipCode) WHERE z.population_density_sq_mi > 0
                RETURN count(z) AS cnt, round(avg(z.population_density_sq_mi), 1) AS avg_density
            """).single()
            print(f"  Population density: {pop_check['cnt']:,} ZIPs, avg = {pop_check['avg_density']}")
    
    finally:
        driver.close()


def main():
    print("=" * 60)
    print("CBSA + POPULATION DENSITY → NEO4J INTEGRATION")
    print("=" * 60)
    
    # Find files
    cbsa_path = find_file(CBSA_FILE, CBSA_FILE_ALT, "CBSA")
    pop_path = find_file(POP_FILE, POP_FILE_ALT, "Population")
    
    if not cbsa_path:
        print("\n❌ CBSA file not found. Please place it in this folder or Downloads.")
        print("   Expected name contains: 'CBSA' and 'Edit' and ends with .xlsx")
        return
    
    if not pop_path:
        print("\n⚠️  Population density file not found. Will proceed with CBSA only.")
        pop_data = pd.DataFrame(columns=['zip', 'population_density_sq_mi'])
    else:
        pop_data = load_pop_density(pop_path)
    
    cbsa_data = load_cbsa_data(cbsa_path)
    update_neo4j(cbsa_data, pop_data)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("  1. Push to GitHub: git add cbsa_pop_density_integration.py && git commit -m 'Add CBSA + pop density integration' && git push")
    print("  2. The AI Assistant can now answer questions like:")
    print("     - 'What is the CBSA classification for Goshen County Wyoming?'")
    print("     - 'What is the population density in Montgomery County Maryland?'")
    print("     - 'Show me rural areas (Non CBSA) with high risk scores'")
    print("=" * 60)


if __name__ == "__main__":
    main()
