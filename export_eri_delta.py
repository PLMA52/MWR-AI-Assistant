"""
ERI Cost of Labor & Cost of Living — Period-over-Period Delta
Compares August 2025 (prior) vs January 2026 (current) ERI data
Aggregates ZIP-level deltas to county level for Power BI Inform tab

Author: Michel Pierre-Louis
Date: February 2026
"""

import os
import csv
import pandas as pd
import warnings
from neo4j import GraphDatabase

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# August 2025 ERI file (prior period)
AUGUST_ERI_FILE = r"C:\Users\mpierrelouis1\August 2025 Cost of Labor and Cost of Living ERI.xlsx"
AUGUST_ERI_FALLBACK = r"C:\Users\mpierrelouis1\MWR-AI-Assistant\August 2025 Cost of Labor and Cost of Living ERI.xlsx"

# Neo4j (current period — January 2026 data already loaded)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

# Output
OUTPUT_DIR = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ERI_Delta_Aug2025_Jan2026.csv")


def load_august_2025_eri(filepath):
    """Parse August 2025 ERI file for Cost of Labor and Cost of Living"""
    
    if not os.path.exists(filepath):
        print(f"   File not found: {filepath}")
        return None, None
    
    print(f"   Reading: {filepath}")
    
    # Cost of Labor — skip ERI header rows
    df_labor = pd.read_excel(filepath, sheet_name='Comparison List - Labor', 
                              header=None, skiprows=7)
    df_labor.columns = ['City_State', 'State', 'ZIP', 'COL_30K', 'COL_40K', 'COL_50K',
                         'Cost_of_Labor_Avg', 'CBSA_Div', 'CBSA_or_ZP', 'Prior_SDX_Rate',
                         'COL_Delta', 'COL_Pct_Change', 'COL_Market_Measure', 'COL_Risk_Ranking']
    
    # Cost of Living
    df_living = pd.read_excel(filepath, sheet_name='Comparison List - Cost of Livin',
                               header=None, skiprows=7)
    df_living.columns = ['City_State', 'State', 'ZIP', 'COLiv_30K', 'COLiv_40K', 'COLiv_50K',
                          'Cost_of_Living_Avg', 'CBSA', 'CBSA_or_ZP']
    
    # Clean ZIPs
    for df in [df_labor, df_living]:
        df['ZIP'] = pd.to_numeric(df['ZIP'], errors='coerce')
        df.dropna(subset=['ZIP'], inplace=True)
        df['ZIP'] = df['ZIP'].astype(int).astype(str).str.zfill(5)
    
    print(f"   August 2025 Labor: {len(df_labor):,} ZIPs")
    print(f"   August 2025 Living: {len(df_living):,} ZIPs")
    
    return df_labor[['ZIP', 'Cost_of_Labor_Avg']], df_living[['ZIP', 'Cost_of_Living_Avg']]


def load_january_2026_from_neo4j():
    """Query Neo4j for current (January 2026) ERI values"""
    
    print("   Querying Neo4j for January 2026 ERI data...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            result = session.run('''
                MATCH (z:ZipCode)
                WHERE z.cost_of_labor IS NOT NULL 
                  AND z.cost_of_living IS NOT NULL
                  AND z.state IS NOT NULL
                  AND z.county IS NOT NULL
                RETURN z.zip AS zip, z.state AS state, z.county AS county,
                       z.cost_of_labor AS cost_of_labor,
                       z.cost_of_living AS cost_of_living
            ''')
            
            records = [dict(r) for r in result]
            df = pd.DataFrame(records)
            df['zip'] = df['zip'].astype(str).str.zfill(5)
            
            print(f"   January 2026 from Neo4j: {len(df):,} ZIPs with ERI data")
            return df
    finally:
        driver.close()


def calculate_deltas_and_export():
    """Main function: load both periods, calculate deltas, aggregate to county, export"""
    
    print("=" * 60)
    print("ERI COST OF LABOR & LIVING — PERIOD DELTA")
    print("August 2025 → January 2026")
    print("=" * 60)
    
    # Step 1: Load August 2025 data
    print("\n[1/4] Loading August 2025 ERI data...")
    # Try primary path first, then fallback
    if os.path.exists(AUGUST_ERI_FILE):
        filepath = AUGUST_ERI_FILE
    elif os.path.exists(AUGUST_ERI_FALLBACK):
        filepath = AUGUST_ERI_FALLBACK
    else:
        print(f"\n   ERROR: Cannot find August 2025 ERI file at:")
        print(f"   {AUGUST_ERI_FILE}")
        print(f"   {AUGUST_ERI_FALLBACK}")
        print(f"   Please update the path in the script.")
        return
    
    aug_labor, aug_living = load_august_2025_eri(filepath)
    
    if aug_labor is None:
        return
    
    # Step 2: Load January 2026 data from Neo4j
    print("\n[2/4] Loading January 2026 ERI data from Neo4j...")
    jan_df = load_january_2026_from_neo4j()
    
    # Step 3: Calculate ZIP-level deltas
    print("\n[3/4] Calculating deltas...")
    
    # Merge August labor with January
    merged = jan_df.merge(aug_labor, left_on='zip', right_on='ZIP', how='inner', suffixes=('', '_aug'))
    merged = merged.merge(aug_living, left_on='zip', right_on='ZIP', how='inner', suffixes=('', '_aug_liv'))
    
    print(f"   Matched ZIPs: {len(merged):,}")
    
    # Calculate deltas
    merged['labor_delta'] = merged['cost_of_labor'] - merged['Cost_of_Labor_Avg']
    merged['labor_pct_change'] = (merged['labor_delta'] / merged['Cost_of_Labor_Avg']) * 100
    merged['living_delta'] = merged['cost_of_living'] - merged['Cost_of_Living_Avg']
    merged['living_pct_change'] = (merged['living_delta'] / merged['Cost_of_Living_Avg']) * 100
    
    # Classify direction
    def classify_direction(pct):
        if pct > 0.5:
            return "Rising"
        elif pct < -0.5:
            return "Declining"
        else:
            return "Stable"
    
    merged['labor_direction'] = merged['labor_pct_change'].apply(classify_direction)
    merged['living_direction'] = merged['living_pct_change'].apply(classify_direction)
    
    print(f"\n   Cost of Labor direction:")
    print(f"   {merged['labor_direction'].value_counts().to_string()}")
    print(f"\n   Cost of Living direction:")
    print(f"   {merged['living_direction'].value_counts().to_string()}")
    
    # Step 4: Aggregate to county level
    print("\n[4/4] Aggregating to county level...")
    
    county_agg = merged.groupby(['state', 'county']).agg(
        Labor_Aug2025=('Cost_of_Labor_Avg', 'mean'),
        Labor_Jan2026=('cost_of_labor', 'mean'),
        Labor_Delta=('labor_delta', 'mean'),
        Labor_Pct_Change=('labor_pct_change', 'mean'),
        Living_Aug2025=('Cost_of_Living_Avg', 'mean'),
        Living_Jan2026=('cost_of_living', 'mean'),
        Living_Delta=('living_delta', 'mean'),
        Living_Pct_Change=('living_pct_change', 'mean'),
        ZIP_Count=('zip', 'count')
    ).reset_index()
    
    # Round values
    for col in ['Labor_Aug2025', 'Labor_Jan2026', 'Labor_Delta', 'Labor_Pct_Change',
                'Living_Aug2025', 'Living_Jan2026', 'Living_Delta', 'Living_Pct_Change']:
        county_agg[col] = county_agg[col].round(2)
    
    # Add County_State_Key for Power BI join
    county_agg['County_State_Key'] = county_agg['county'] + '|' + county_agg['state']
    
    # Classify county-level direction
    county_agg['Labor_Direction'] = county_agg['Labor_Pct_Change'].apply(classify_direction)
    county_agg['Living_Direction'] = county_agg['Living_Pct_Change'].apply(classify_direction)
    
    print(f"   Counties: {len(county_agg):,}")
    
    # Export CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    export_cols = [
        'County_State_Key',
        'Labor_Aug2025', 'Labor_Jan2026', 'Labor_Delta', 'Labor_Pct_Change', 'Labor_Direction',
        'Living_Aug2025', 'Living_Jan2026', 'Living_Delta', 'Living_Pct_Change', 'Living_Direction'
    ]
    
    county_agg[export_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n   Saved to: {OUTPUT_FILE}")
    
    # Show sample
    print(f"\n   Sample data (first 5 counties):")
    print(f"   {'County':<30} {'Labor Aug':>10} {'Labor Jan':>10} {'Δ %':>8} {'Dir':>10} {'Living Aug':>11} {'Living Jan':>11} {'Δ %':>8} {'Dir':>10}")
    print(f"   {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*11} {'-'*8} {'-'*10}")
    for _, r in county_agg.head(5).iterrows():
        key = r['County_State_Key'][:30]
        print(f"   {key:<30} {r['Labor_Aug2025']:>10.1f} {r['Labor_Jan2026']:>10.1f} {r['Labor_Pct_Change']:>7.2f}% {r['Labor_Direction']:>10} {r['Living_Aug2025']:>10.1f} {r['Living_Jan2026']:>10.1f} {r['Living_Pct_Change']:>7.2f}% {r['Living_Direction']:>10}")
    
    # Verify a known county
    sf = county_agg[county_agg['County_State_Key'].str.contains('San Francisco')]
    if not sf.empty:
        r = sf.iloc[0]
        print(f"\n   VERIFICATION — San Francisco:")
        print(f"      Cost of Labor:  Aug={r['Labor_Aug2025']:.1f} → Jan={r['Labor_Jan2026']:.1f} ({r['Labor_Pct_Change']:+.2f}%) {r['Labor_Direction']}")
        print(f"      Cost of Living: Aug={r['Living_Aug2025']:.1f} → Jan={r['Living_Jan2026']:.1f} ({r['Living_Pct_Change']:+.2f}%) {r['Living_Direction']}")
    
    # Summary stats
    print(f"\n   SUMMARY:")
    print(f"   Cost of Labor direction (county level):")
    print(f"      Rising:    {(county_agg['Labor_Direction'] == 'Rising').sum()} counties")
    print(f"      Stable:    {(county_agg['Labor_Direction'] == 'Stable').sum()} counties")
    print(f"      Declining: {(county_agg['Labor_Direction'] == 'Declining').sum()} counties")
    print(f"   Cost of Living direction (county level):")
    print(f"      Rising:    {(county_agg['Living_Direction'] == 'Rising').sum()} counties")
    print(f"      Stable:    {(county_agg['Living_Direction'] == 'Stable').sum()} counties")
    print(f"      Declining: {(county_agg['Living_Direction'] == 'Declining').sum()} counties")
    
    print(f"\n{'=' * 60}")
    print("DONE!")
    print("=" * 60)
    print("\nNEXT STEPS IN POWER BI:")
    print("1. Get Data > CSV > ERI_Delta_Aug2025_Jan2026.csv")
    print("2. Create relationship: County_State_Key > County_Summary.County_State_Key")
    print("3. Add to Inform tab (at the end, after age columns):")
    print("   - Labor_Pct_Change (rename: 'Cost of Labor Δ %')")
    print("   - Labor_Direction (rename: 'CoL Trend')")
    print("   - Living_Pct_Change (rename: 'Cost of Living Δ %')")
    print("   - Living_Direction (rename: 'CoLiv Trend')")
    print("4. Publish to Power BI Service")


if __name__ == "__main__":
    calculate_deltas_and_export()
