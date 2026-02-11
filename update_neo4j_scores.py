#!/usr/bin/env python3
"""
Update Neo4j AuraDB with fresh risk scores from the re-scoring pipeline.
Runs after mwr_rescore.py generates a new CSV.
"""

import os
import sys
import pandas as pd
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
CSV_FILE = "MWR_Combined_ZipCode_Risk_v2.csv"


def update_scores():
    if not all([NEO4J_URI, NEO4J_PASSWORD]):
        print("⚠️ Neo4j credentials not configured. Skipping update.")
        sys.exit(0)

    if not os.path.exists(CSV_FILE):
        print(f"❌ {CSV_FILE} not found!")
        sys.exit(1)

    print("📊 Loading fresh scores...")
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=['New_Combined_Risk_Score'])
    print(f"   {len(df):,} ZIPs with scores")

    print("🔌 Connecting to Neo4j AuraDB...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Update in batches of 1000
    batch_size = 1000
    total = len(df)
    updated = 0

    with driver.session() as session:
        for start in range(0, total, batch_size):
            batch = df.iloc[start:start + batch_size]
            records = batch[['zip', 'New_Combined_Risk_Score', 'New_Risk_Score_Pct',
                             'New_Risk_Tier', 'Sub_State_Complexity_Score',
                             'Forward_Looking_Score', 'Sustained_Pressure_Score']].to_dict('records')

            # Standardize zip to string
            for r in records:
                r['zip'] = str(r['zip']).zfill(5)

            session.run("""
                UNWIND $records AS r
                MATCH (z:ZipCode {zip: r.zip})
                SET z.newRiskScore = r.New_Combined_Risk_Score,
                    z.newRiskScorePct = r.New_Risk_Score_Pct,
                    z.newRiskTier = r.New_Risk_Tier,
                    z.subStateComplexity = r.Sub_State_Complexity_Score,
                    z.forwardLooking = r.Forward_Looking_Score,
                    z.sustainedPressure = r.Sustained_Pressure_Score,
                    z.lastRescore = datetime()
            """, records=records)

            updated += len(batch)
            print(f"   Updated {updated:,}/{total:,} ZIPs...")

    driver.close()
    print(f"✅ Neo4j updated with fresh risk scores!")


if __name__ == "__main__":
    update_scores()
