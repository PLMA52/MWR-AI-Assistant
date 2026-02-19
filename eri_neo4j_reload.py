"""
ERI Neo4j Time-Series Loader (Re-run only)
Reads from ERI_TimeSeries_AllPeriods.csv and loads into Neo4j.
Use this if Step 5 failed but Steps 1-4 completed successfully.
"""

import os
import pandas as pd
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "RYA3u0QAGRSbpdjJpB4r5-yWvAVPW4rdioPy00lRx3g")

OUTPUT_DIR = os.path.expanduser("~/OneDrive - SODEXO/MWR_Automation_Data")
ts_file = os.path.join(OUTPUT_DIR, "ERI_TimeSeries_AllPeriods.csv")

print("Loading time-series CSV...")
df = pd.read_csv(ts_file)
df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
print(f"   Loaded {len(df):,} ZIPs")

# Extract period labels from column names — only actual date periods (YYYY-MM format)
import re
labor_cols = [c for c in df.columns if c.startswith('labor_') and re.match(r'labor_\d{4}-\d{2}$', c)]
living_cols = [c for c in df.columns if c.startswith('living_') and re.match(r'living_\d{4}-\d{2}$', c)]
periods = [c.replace('labor_', '') for c in labor_cols]
print(f"   Periods: {periods}")

print(f"\nConnecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

batch_size = 500
total_updated = 0

try:
    with driver.session() as session:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            
            for _, row in batch.iterrows():
                zip_code = row['ZIP']
                
                labor_history = []
                living_history = []
                for p in periods:
                    lv = row.get(f'labor_{p}')
                    liv = row.get(f'living_{p}')
                    labor_history.append(round(float(lv), 2) if pd.notna(lv) else 0.0)
                    living_history.append(round(float(liv), 2) if pd.notna(liv) else 0.0)
                
                session.run('''
                    MATCH (z:ZipCode {zip: $zip})
                    SET z.eri_periods = $periods,
                        z.eri_labor_history = $labor_history,
                        z.eri_living_history = $living_history
                ''', zip=zip_code, periods=periods,
                     labor_history=labor_history, living_history=living_history)
            
            total_updated += len(batch)
            if total_updated % 5000 == 0 or total_updated == len(df):
                print(f"   Updated {total_updated:,} / {len(df):,} ZIPs...")
finally:
    driver.close()

print(f"\nDone! {total_updated:,} ZIPs updated with ERI time-series in Neo4j")
