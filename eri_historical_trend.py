"""
ERI Historical Trend Processor — 13 Files (May 2024 → January 2026)
Reads all ERI Cost of Labor & Cost of Living files, extracts averages per ZIP,
calculates period-over-period and YoY deltas, aggregates to county level.

Outputs:
1. CSV for Power BI Inform tab (county-level current values + YoY delta)
2. CSV for Neo4j time-series loading (ZIP-level, all 13 periods)

Author: Michel Pierre-Louis
Date: February 2026
"""

import os
import re
import pandas as pd
import warnings
from neo4j import GraphDatabase
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

ERI_FOLDER = r"C:\Users\mpierrelouis1\OneDrive - SODEXO\Desktop\Cost of Labor and Living Raw Data"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

OUTPUT_DIR = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data")

# Files in chronological order with their period labels
# Format: (filename, period_label, sort_date)
ERI_FILES = [
    ("May 2024 Cost of Labor and Cost of Living ERI.xlsx",       "2024-05", "2024-05-01"),
    ("July 2024 Cost of Labor and Cost of Living ERI.xlsx",      "2024-07", "2024-07-01"),
    ("October 2024 Cost of Labor and Cost of Living ERI.xlsx",   "2024-10", "2024-10-01"),
    ("November 2024 Cost of Labor and Cost of Living ERI.xlsx",  "2024-11", "2024-11-01"),
    ("January 2025 Cost of Labor and Cost of Living ERI.xlsx",   "2025-01", "2025-01-01"),
    ("Feb 2025 Cost of Labor and Cost of Living ERI.xlsx",       "2025-02", "2025-02-01"),
    ("April 2025 Cost of Labor and Cost of Living ERI.xlsx",     "2025-04", "2025-04-01"),
    ("May 2025 Cost of Labor and Cost of Living ERI.xlsx",       "2025-05", "2025-05-01"),
    ("July 2025 Cost of Labor and Cost of Living ERI.xlsx",      "2025-07", "2025-07-01"),
    ("August 2025 Cost of Labor and Cost of Living ERI.xlsx",    "2025-08", "2025-08-01"),
    ("October 2025 Cost of Labor and Cost of Living ERI.xlsx",   "2025-10", "2025-10-01"),
    ("November 2025 Cost of Labor and Cost of Living ERI.xlsx",  "2025-11", "2025-11-01"),
    ("January 2026 Cost of Labor and Cost of Living ERI.xlsx",   "2026-01", "2026-01-01"),
]


def find_labor_sheet(wb_sheets):
    """Find the Cost of Labor sheet regardless of naming variation"""
    for name in wb_sheets:
        if 'labor' in name.lower() or 'cost of labor' in name.lower():
            return name
    return None


def find_living_sheet(wb_sheets):
    """Find the Cost of Living sheet regardless of naming variation"""
    for name in wb_sheets:
        if 'cost of livin' in name.lower() or 'cost of living' in name.lower():
            return name
    return None


def parse_eri_file(filepath, period_label):
    """Parse a single ERI file and return labor + living DataFrames with ZIP and average.
    
    All 13 files have consistent structure:
    - Header row at index 6 (row 7 in Excel), data starts at index 7
    - Col 0: City_State, Col 1: State, Col 2: PostCode
    - Col 3: 30K, Col 4: 40K, Col 5: 50K, Col 6: Average
    """
    
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names
    
    labor_sheet = find_labor_sheet(sheets)
    living_sheet = find_living_sheet(sheets)
    
    if not labor_sheet or not living_sheet:
        print(f"   WARNING: Could not find sheets in {os.path.basename(filepath)}")
        print(f"   Available sheets: {sheets}")
        return None, None
    
    # All files: header at row 6, data starts row 7 (skiprows=7)
    # Average is always column index 6
    df_labor = pd.read_excel(filepath, sheet_name=labor_sheet, header=None, skiprows=7)
    df_living = pd.read_excel(filepath, sheet_name=living_sheet, header=None, skiprows=7)
    
    # Extract ZIP (col 2) and Average (col 6)
    labor_data = pd.DataFrame({
        'ZIP': pd.to_numeric(df_labor.iloc[:, 2], errors='coerce'),
        f'labor_{period_label}': pd.to_numeric(df_labor.iloc[:, 6], errors='coerce')
    })
    
    living_data = pd.DataFrame({
        'ZIP': pd.to_numeric(df_living.iloc[:, 2], errors='coerce'),
        f'living_{period_label}': pd.to_numeric(df_living.iloc[:, 6], errors='coerce')
    })
    
    # Clean
    labor_data = labor_data.dropna(subset=['ZIP']).copy()
    labor_data['ZIP'] = labor_data['ZIP'].astype(int).astype(str).str.zfill(5)
    
    living_data = living_data.dropna(subset=['ZIP']).copy()
    living_data['ZIP'] = living_data['ZIP'].astype(int).astype(str).str.zfill(5)
    
    return labor_data, living_data


def load_zip_geography_from_neo4j():
    """Get ZIP → State, County mapping from Neo4j"""
    print("   Querying Neo4j for ZIP geography...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run('''
                MATCH (z:ZipCode)
                WHERE z.state IS NOT NULL AND z.county IS NOT NULL
                RETURN z.zip AS zip, z.state AS state, z.county AS county
            ''')
            records = [dict(r) for r in result]
            df = pd.DataFrame(records)
            df['zip'] = df['zip'].astype(str).str.zfill(5)
            print(f"   Retrieved {len(df):,} ZIPs with geography")
            return df
    finally:
        driver.close()


def load_eri_time_series_to_neo4j(merged_labor, merged_living, periods):
    """Load time-series ERI data onto ZipCode nodes in Neo4j"""
    print("   Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    labor_cols = [f'labor_{p}' for p in periods]
    living_cols = [f'living_{p}' for p in periods]
    
    # Build a combined dataset
    combined = merged_labor[['ZIP'] + labor_cols].merge(
        merged_living[['ZIP'] + living_cols], on='ZIP', how='inner'
    )
    
    print(f"   Preparing {len(combined):,} ZIPs with {len(periods)} time periods...")
    
    batch_size = 500
    total_updated = 0
    
    try:
        with driver.session() as session:
            # Store periods list and time-series arrays on each ZipCode node
            for start in range(0, len(combined), batch_size):
                batch = combined.iloc[start:start + batch_size]
                
                for _, row in batch.iterrows():
                    zip_code = row['ZIP']
                    
                    # Build labor and living history as lists
                    labor_history = []
                    living_history = []
                    for p in periods:
                        lv = row.get(f'labor_{p}')
                        liv = row.get(f'living_{p}')
                        labor_history.append(round(float(lv), 2) if pd.notna(lv) else None)
                        living_history.append(round(float(liv), 2) if pd.notna(liv) else None)
                    
                    session.run('''
                        MATCH (z:ZipCode {zip: $zip})
                        SET z.eri_periods = $periods,
                            z.eri_labor_history = $labor_history,
                            z.eri_living_history = $living_history
                    ''', zip=zip_code, periods=periods,
                         labor_history=labor_history, living_history=living_history)
                
                total_updated += len(batch)
                if total_updated % 5000 == 0 or total_updated == len(combined):
                    print(f"   Updated {total_updated:,} / {len(combined):,} ZIPs...")
    finally:
        driver.close()
    
    print(f"   Neo4j: {total_updated:,} ZIPs updated with ERI time-series")


def main():
    print("=" * 65)
    print("ERI HISTORICAL TREND PROCESSOR")
    print("13 Files: May 2024 → January 2026 (20 months)")
    print("=" * 65)
    
    # ========================================================
    # STEP 1: Read all 13 files
    # ========================================================
    print("\n[1/5] Reading 13 ERI files...")
    
    all_labor = None
    all_living = None
    periods = []
    
    for filename, period, sort_date in ERI_FILES:
        filepath = os.path.join(ERI_FOLDER, filename)
        if not os.path.exists(filepath):
            print(f"   MISSING: {filename}")
            continue
        
        print(f"   Reading: {period} — {filename[:40]}...")
        labor_df, living_df = parse_eri_file(filepath, period)
        
        if labor_df is None:
            continue
        
        periods.append(period)
        print(f"      Labor: {len(labor_df):,} ZIPs, Living: {len(living_df):,} ZIPs")
        
        if all_labor is None:
            all_labor = labor_df
            all_living = living_df
        else:
            all_labor = all_labor.merge(labor_df, on='ZIP', how='outer')
            all_living = all_living.merge(living_df, on='ZIP', how='outer')
    
    print(f"\n   Total periods loaded: {len(periods)}")
    print(f"   Labor ZIPs (union): {len(all_labor):,}")
    print(f"   Living ZIPs (union): {len(all_living):,}")
    
    # ========================================================
    # STEP 2: Calculate YoY Delta (latest vs 12 months prior)
    # ========================================================
    print("\n[2/5] Calculating YoY deltas...")
    
    # Current period: 2026-01
    # Prior year closest: 2025-01
    current_period = "2026-01"
    prior_year_period = "2025-01"
    
    current_labor_col = f"labor_{current_period}"
    prior_labor_col = f"labor_{prior_year_period}"
    current_living_col = f"living_{current_period}"
    prior_living_col = f"living_{prior_year_period}"
    
    # Labor YoY
    all_labor['labor_yoy_delta'] = all_labor[current_labor_col] - all_labor[prior_labor_col]
    all_labor['labor_yoy_pct'] = (all_labor['labor_yoy_delta'] / all_labor[prior_labor_col]) * 100
    
    # Living YoY
    all_living['living_yoy_delta'] = all_living[current_living_col] - all_living[prior_living_col]
    all_living['living_yoy_pct'] = (all_living['living_yoy_delta'] / all_living[prior_living_col]) * 100
    
    def classify_direction(pct):
        if pd.isna(pct):
            return "N/A"
        if pct > 0.5:
            return "Rising"
        elif pct < -0.5:
            return "Declining"
        else:
            return "Stable"
    
    all_labor['labor_direction'] = all_labor['labor_yoy_pct'].apply(classify_direction)
    all_living['living_direction'] = all_living['living_yoy_pct'].apply(classify_direction)
    
    valid_labor = all_labor.dropna(subset=[current_labor_col, prior_labor_col])
    valid_living = all_living.dropna(subset=[current_living_col, prior_living_col])
    
    print(f"   YoY comparison: {prior_year_period} → {current_period}")
    print(f"   Labor ZIPs with both periods: {len(valid_labor):,}")
    print(f"   Living ZIPs with both periods: {len(valid_living):,}")
    print(f"\n   Cost of Labor YoY direction (ZIP level):")
    print(f"      Rising:    {(valid_labor['labor_direction'] == 'Rising').sum():,}")
    print(f"      Stable:    {(valid_labor['labor_direction'] == 'Stable').sum():,}")
    print(f"      Declining: {(valid_labor['labor_direction'] == 'Declining').sum():,}")
    print(f"\n   Cost of Living YoY direction (ZIP level):")
    print(f"      Rising:    {(valid_living['living_direction'] == 'Rising').sum():,}")
    print(f"      Stable:    {(valid_living['living_direction'] == 'Stable').sum():,}")
    print(f"      Declining: {(valid_living['living_direction'] == 'Declining').sum():,}")
    
    # ========================================================
    # STEP 3: Aggregate to County Level for Power BI
    # ========================================================
    print("\n[3/5] Aggregating to county level for Power BI...")
    
    geo = load_zip_geography_from_neo4j()
    
    # Merge geography
    labor_geo = valid_labor.merge(geo, left_on='ZIP', right_on='zip', how='inner')
    living_geo = valid_living.merge(geo, left_on='ZIP', right_on='zip', how='inner')
    
    # County aggregation — Dan suggested median, but we use average for consistency with existing data
    county_labor = labor_geo.groupby(['state', 'county']).agg(
        Labor_PY=( prior_labor_col, 'mean'),
        Labor_CY=(current_labor_col, 'mean'),
        Labor_YoY_Delta=('labor_yoy_delta', 'mean'),
        Labor_YoY_Pct=('labor_yoy_pct', 'mean'),
    ).reset_index()
    
    county_living = living_geo.groupby(['state', 'county']).agg(
        Living_PY=(prior_living_col, 'mean'),
        Living_CY=(current_living_col, 'mean'),
        Living_YoY_Delta=('living_yoy_delta', 'mean'),
        Living_YoY_Pct=('living_yoy_pct', 'mean'),
    ).reset_index()
    
    # Merge labor + living at county level
    county = county_labor.merge(county_living, on=['state', 'county'], how='outer')
    
    # Round
    for col in ['Labor_PY', 'Labor_CY', 'Labor_YoY_Delta', 'Labor_YoY_Pct',
                'Living_PY', 'Living_CY', 'Living_YoY_Delta', 'Living_YoY_Pct']:
        county[col] = county[col].round(2)
    
    # Add County_State_Key and direction
    county['County_State_Key'] = county['county'] + '|' + county['state']
    county['Labor_Direction'] = county['Labor_YoY_Pct'].apply(classify_direction)
    county['Living_Direction'] = county['Living_YoY_Pct'].apply(classify_direction)
    
    print(f"   Counties: {len(county):,}")
    print(f"\n   Cost of Labor YoY (county level):")
    print(f"      Rising:    {(county['Labor_Direction'] == 'Rising').sum()}")
    print(f"      Stable:    {(county['Labor_Direction'] == 'Stable').sum()}")
    print(f"      Declining: {(county['Labor_Direction'] == 'Declining').sum()}")
    print(f"\n   Cost of Living YoY (county level):")
    print(f"      Rising:    {(county['Living_Direction'] == 'Rising').sum()}")
    print(f"      Stable:    {(county['Living_Direction'] == 'Stable').sum()}")
    print(f"      Declining: {(county['Living_Direction'] == 'Declining').sum()}")
    
    # Export for Power BI
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pbi_file = os.path.join(OUTPUT_DIR, "ERI_YoY_Delta_Jan2025_Jan2026.csv")
    export_cols = [
        'County_State_Key',
        'Labor_PY', 'Labor_CY', 'Labor_YoY_Delta', 'Labor_YoY_Pct', 'Labor_Direction',
        'Living_PY', 'Living_CY', 'Living_YoY_Delta', 'Living_YoY_Pct', 'Living_Direction'
    ]
    county[export_cols].to_csv(pbi_file, index=False)
    print(f"\n   Power BI CSV saved: {pbi_file}")
    
    # Sample verification
    sf = county[county['County_State_Key'].str.contains('San Francisco')]
    if not sf.empty:
        r = sf.iloc[0]
        print(f"\n   VERIFICATION — San Francisco:")
        print(f"      Cost of Labor:  Jan 2025={r['Labor_PY']:.1f} → Jan 2026={r['Labor_CY']:.1f} ({r['Labor_YoY_Pct']:+.2f}%) {r['Labor_Direction']}")
        print(f"      Cost of Living: Jan 2025={r['Living_PY']:.1f} → Jan 2026={r['Living_CY']:.1f} ({r['Living_YoY_Pct']:+.2f}%) {r['Living_Direction']}")
    
    # Show top 5 rising counties
    print(f"\n   TOP 5 RISING Cost of Labor counties:")
    top_rising = county.nlargest(5, 'Labor_YoY_Pct')
    for _, r in top_rising.iterrows():
        print(f"      {r['County_State_Key']:<35} {r['Labor_YoY_Pct']:+.2f}%")
    
    print(f"\n   TOP 5 DECLINING Cost of Labor counties:")
    top_declining = county.nsmallest(5, 'Labor_YoY_Pct')
    for _, r in top_declining.iterrows():
        print(f"      {r['County_State_Key']:<35} {r['Labor_YoY_Pct']:+.2f}%")
    
    # ========================================================
    # STEP 4: Export full time-series CSV for Neo4j
    # ========================================================
    print("\n[4/5] Exporting full time-series for Neo4j...")
    
    neo4j_file = os.path.join(OUTPUT_DIR, "ERI_TimeSeries_AllPeriods.csv")
    
    # Combine labor + living into one wide DataFrame
    ts_combined = all_labor.merge(all_living, on='ZIP', how='outer')
    ts_combined.to_csv(neo4j_file, index=False)
    print(f"   Time-series CSV saved: {neo4j_file}")
    print(f"   {len(ts_combined):,} ZIPs × {len(periods)} periods")
    
    # ========================================================
    # STEP 5: Load time-series into Neo4j
    # ========================================================
    print("\n[5/5] Loading ERI time-series into Neo4j...")
    load_eri_time_series_to_neo4j(all_labor, all_living, periods)
    
    # ========================================================
    # DONE
    # ========================================================
    print(f"\n{'=' * 65}")
    print("DONE!")
    print("=" * 65)
    print(f"\nFILES CREATED:")
    print(f"   1. {pbi_file}")
    print(f"      → Power BI: YoY delta for Inform tab")
    print(f"   2. {neo4j_file}")
    print(f"      → Full time-series backup")
    print(f"\nNEO4J UPDATED:")
    print(f"   → eri_periods: list of {len(periods)} period labels")
    print(f"   → eri_labor_history: Cost of Labor values per period")
    print(f"   → eri_living_history: Cost of Living values per period")
    print(f"\nNEXT STEPS IN POWER BI:")
    print(f"   1. Get Data > CSV > ERI_YoY_Delta_Jan2025_Jan2026.csv")
    print(f"   2. Relationship: County_State_Key → County_Summary.County_State_Key")
    print(f"      (One-to-one, Single direction, Active)")
    print(f"   3. Add to Inform tab (end of table):")
    print(f"      - Labor_CY (rename: 'Cost of Labor CY')")
    print(f"      - Labor_YoY_Pct (rename: 'Cost of Labor YoY Δ %')")
    print(f"      - Labor_Direction (rename: 'CoL Trend')")
    print(f"      - Living_CY (rename: 'Cost of Living CY')")
    print(f"      - Living_YoY_Pct (rename: 'Cost of Living YoY Δ %')")
    print(f"      - Living_Direction (rename: 'CoLiv Trend')")
    print(f"   4. Publish to Power BI Service")


if __name__ == "__main__":
    main()
