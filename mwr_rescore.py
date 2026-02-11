#!/usr/bin/env python3
"""
===============================================================================
MWR AUTOMATED RE-SCORING PIPELINE — Consolidated v1.0
===============================================================================
Combines three separate scoring systems into one automated pipeline:

1. v19c ML Model: ZIP-level risk predictions using ensemble ML
2. 3D State Scoring: Sub_State_Complexity + Forward_Looking + Sustained_Pressure
3. County Risk Adjustments: Within-state variation based on wage premiums

INPUT:  Neo4j AuraDB (ZIP nodes with wage history) + ERI/MW Tracker data files
OUTPUT: MWR_Combined_ZipCode_Risk_v2.csv (41,554 rows, 61 columns)
        → Pushed to OneDrive → Power BI auto-refreshes

SCHEDULE: Monthly (chained after BLS data pull on 5th of each month)
          Can also be triggered manually via workflow_dispatch

AUTHOR: Michel Pierre-Louis / Claude AI
DATE:   February 2026
===============================================================================
"""

from __future__ import annotations
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               HistGradientBoostingRegressor)

warnings.filterwarnings("ignore")

# Optional ML libraries
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# Neo4j (optional - for cloud runs)
try:
    from neo4j import GraphDatabase
    _HAS_NEO4J = True
except ImportError:
    _HAS_NEO4J = False

RANDOM_STATE = 42
TIMESTAMP = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration for local vs GitHub Actions runs."""
    
    # Neo4j AuraDB
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://551c1b37.databases.neo4j.io")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    # Base path on Michel's machine
    _BASE = os.getenv("MWR_DATA_BASE",
        r"C:\Users\mpierrelouis1\OneDrive - SODEXO\Desktop\Daniel Green"
        r"\Daniel Green_AsOf_09072021\Daniel Green\Risk Assessment MinWage_ZipCode")
    
    # File paths - set via environment or defaults for local
    MW_FILE = os.getenv("MW_FILE", os.path.join(
        _BASE,
        "Min Wage Details State City and County current and future with July 2025 as current.xlsb"))
    ERI_FILE = os.getenv("ERI_FILE", os.path.join(
        _BASE,
        "August 2025 Cost of Labor and Cost of Living ERI.xlsx"))
    CBSA_FILE = os.getenv("CBSA_FILE", os.path.join(
        _BASE,
        "April 2025 Zip Code to CBSA w Preferred City (Edit Version).xlsx"))
    FRONTLINE_FILE = os.getenv("FRONTLINE_FILE", os.path.join(
        _BASE, "Meeting Notes",
        "Frontline Job Desc Detail Comp Benchmark - Costing Model View by Zip Code (Winter 2026).xlsx"))
    
    # Output — goes to MWR_Automation_Data folder that Power BI reads from
    OUTPUT_CSV = os.getenv("OUTPUT_CSV", os.path.join(
        r"C:\Users\mpierrelouis1\OneDrive - SODEXO\MWR_Automation_Data",
        "MWR_Combined_ZipCode_Risk_v2.csv"))
    
    # Mode: 'full' = retrain ML + recalculate everything
    #        'rescore' = use saved model weights, recalculate scores only
    MODE = os.getenv("RESCORE_MODE", "full")


# ==============================================================================
# SUSTAINED PRESSURE DATA (MANUALLY MAINTAINED)
# ==============================================================================
# These represent consecutive years of minimum wage increases per state.
# Updated when new wage legislation is enacted. Source: ERI MW Tracker + research.

SUSTAINED_PRESSURE = {
    'California': 8, 'Washington': 8,
    'Colorado': 7, 'Arizona': 7, 'DC': 7,
    'New York': 6, 'New Jersey': 6, 'Florida': 6,
    'Oregon': 6, 'Massachusetts': 6, 'Connecticut': 6, 'Maryland': 6,
    'Missouri': 5, 'Minnesota': 5, 'Illinois': 5, 'New Mexico': 5,
    'Maine': 5, 'Vermont': 5, 'Rhode Island': 5,
    'Hawaii': 4, 'Nebraska': 4, 'Ohio': 4, 'Delaware': 4,
    'Nevada': 4, 'South Dakota': 4, 'Montana': 4, 'Arkansas': 4,
    'Virginia': 4,
    'Michigan': 3, 'Alaska': 3,
    'Puerto Rico': 2,
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def standardize_zip(val) -> str:
    """Standardize ZIP code to 5-digit string."""
    try:
        return str(int(float(val))).zfill(5)
    except (ValueError, TypeError):
        return str(val).strip().zfill(5)


def norm(col: pd.Series) -> pd.Series:
    """Normalize a column to 0-1 range."""
    c_min, c_max = col.min(), col.max()
    if c_max - c_min == 0:
        return col * 0
    return (col - c_min) / (c_max - c_min)


@dataclass
class ModelResult:
    """Container for model training results."""
    name: str
    model: Any
    r2: float
    r2_cv: float
    rmse: float
    mae: float
    feature_importance: Optional[pd.DataFrame] = None


# ==============================================================================
# SECTION 1: LOAD BASE ZIP DATA FROM NEO4J
# ==============================================================================

def load_base_zip_data_from_neo4j() -> pd.DataFrame:
    """Load ZIP-level data from Neo4j AuraDB."""
    print("\n📂 Loading base ZIP data from Neo4j AuraDB...")
    
    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    )
    
    query = """
    MATCH (z:ZipCode)
    OPTIONAL MATCH (z)-[:IN_STATE]->(s:State)
    RETURN 
        z.zip AS zip,
        z.preferredCity AS preferredCity,
        z.county AS county,
        z.state AS State_Abbr,
        s.name AS State_Name,
        z.fips AS fips,
        z.areaCode AS areaCode,
        z.cbsaCode AS cbsaCode,
        z.timeZone AS timeZone,
        s.abbr AS abbr,
        s.federalOrState AS federalOrState,
        s.hasLocalRates AS hasLocalRates,
        s.HasVolatility AS HasVolatility,
        z.volatilityScore AS Old_RiskScore,
        z.riskTier AS Old_RiskTier,
        z.riskTier5 AS Old_RiskTier5,
        z.HCF AS HCF,
        z.MHJ AS MHJ,
        z.UCP AS UCP,
        z.IDX AS IDX,
        z.LRF AS LRF,
        z.HCFxUCP AS HCFxUCP,
        z.MHJxIDX AS MHJxIDX,
        z.topDriver AS topDriver
    ORDER BY z.zip
    """
    
    with driver.session() as session:
        result = session.run(query)
        records = [dict(r) for r in result]
    
    driver.close()
    
    df = pd.DataFrame(records)
    df['zip'] = df['zip'].apply(standardize_zip)
    df['StateCounty'] = df['State_Name'].fillna('') + ' - ' + df['county'].fillna('')
    
    # Set _NodeId_ placeholder
    df['_NodeId_'] = range(len(df))
    
    print(f"   ✓ Loaded {len(df):,} ZIP codes from Neo4j")
    return df


def load_base_zip_data_from_csv(filepath: str) -> pd.DataFrame:
    """Load base ZIP data from existing CSV (fallback if Neo4j unavailable)."""
    print(f"\n📂 Loading base ZIP data from CSV: {filepath}")
    df = pd.read_csv(filepath)
    
    # Keep only source columns (not calculated ones)
    source_cols = ['zip', '_NodeId_', 'preferredCity', 'county', 'State_Abbr', 'State_Name',
                   'StateCounty', 'fips', 'areaCode', 'cbsaCode', 'timeZone',
                   'abbr', 'federalOrState', 'hasLocalRates', 'HasVolatility',
                   'Old_RiskScore', 'Old_RiskTier', 'Old_RiskTier5', 'Old_VolatilityScore',
                   'HCF', 'MHJ', 'UCP', 'IDX', 'LRF', 'HCFxUCP', 'MHJxIDX', 'topDriver',
                   'wage_jan2019', 'wage_july2019', 'wage_jan2020', 'wage_jan2021',
                   'wage_jan2022', 'wage_jan2023', 'wage_july2023', 'wage_jan2024']
    
    existing = [c for c in source_cols if c in df.columns]
    df = df[existing].copy()
    df['zip'] = df['zip'].apply(standardize_zip)
    
    print(f"   ✓ Loaded {len(df):,} ZIP codes from CSV")
    return df


# ==============================================================================
# SECTION 2: LOAD MW TRACKER & BUILD FEATURES FOR v19c ML MODEL
# ==============================================================================

def load_mw_tracker_features() -> Dict:
    """Load MW tracker and extract features for ML model."""
    print("\n📂 Loading MW Tracker features...")
    
    mw = pd.read_excel(Config.MW_FILE, engine='pyxlsb', sheet_name=None)
    full_tracker = mw['Full Tracker']
    local_tracker = mw['Local Tracker ']
    trends = mw['Minimum Wage Trends by Zip Code']
    
    # Column aliases
    current_col = 'Greater of State, City, or County Current Minimum Wage'
    future_col = ('Greater of State, City, or County Late New Minimum Wage '
                  '- If no Change same as current ')
    
    # Standardize ZIP
    full_tracker['ZIP'] = full_tracker['zip'].apply(standardize_zip)
    trends['ZIP'] = trends['zip'].apply(standardize_zip)
    
    # === ZIP-Level Features ===
    zip_features = full_tracker[['ZIP', 'state', current_col, future_col]].copy()
    zip_features.columns = ['ZIP', 'State', 'Current_Wage', 'Future_Wage']
    zip_features['Wage_Change_Pct'] = (
        (zip_features['Future_Wage'] - zip_features['Current_Wage']) /
        zip_features['Current_Wage'] * 100
    ).fillna(0)
    zip_features['Has_Scheduled_Increase'] = (
        zip_features['Future_Wage'] > zip_features['Current_Wage']
    ).astype(int)
    zip_features['Has_Local_Jurisdiction'] = (
        (full_tracker.get('Current City Minimum Wage: : (Some areas within a city might be '
                          'classified as incorporated - Please check with your local area to '
                          'verify the minimum wage is correct)', 0) > 0) |
        (full_tracker.get('Current County Minimum Wage: (Some areas within a county might be '
                          'classified as incorporated - Please check with your local area to '
                          'verify the minimum wage is correct)', 0) > 0)
    ).astype(int)
    
    # === State-Level Features ===
    print("   Computing state-level features...")
    
    city_count = full_tracker.groupby('state')['Unique City Naming'].nunique().reset_index()
    city_count.columns = ['State', 'City_Jurisdiction_Count']
    
    county_count = full_tracker.groupby('state')['Unique County Naming '].nunique().reset_index()
    county_count.columns = ['State', 'County_Jurisdiction_Count']
    
    wage_complexity = full_tracker.groupby('state')[current_col].nunique().reset_index()
    wage_complexity.columns = ['State', 'Wage_Complexity']
    
    wage_stats = full_tracker.groupby('state')[current_col].agg(
        ['max', 'min', 'std', 'mean']).reset_index()
    wage_stats.columns = ['State', 'State_Max_Wage', 'State_Min_Wage',
                          'State_Wage_Std', 'State_Avg_Wage']
    wage_stats['Wage_Range'] = wage_stats['State_Max_Wage'] - wage_stats['State_Min_Wage']
    
    full_tracker['Has_Future_Change'] = (
        full_tracker[future_col] > full_tracker[current_col]).astype(int)
    future_stats = full_tracker.groupby('state').agg({
        'Has_Future_Change': 'sum',
        future_col: 'max'
    }).reset_index()
    future_stats.columns = ['State', 'Future_Change_Count', 'State_Max_Future_Wage']
    
    # Industry carveouts from Local Tracker
    local_tracker['State_Ex'] = local_tracker[
        'DATABASE CITY OR COUNTY NAME - Unique Code'].str[-2:]
    carveouts = local_tracker['State_Ex'].value_counts().reset_index()
    carveouts.columns = ['State', 'Industry_Carveout_Count']
    
    # === Historical Volatility from Trends ===
    print("   Computing historical volatility...")
    wage_cols = [c for c in trends.columns
                 if 'Wage' in c and '% Change' not in c and c not in ['zip', 'ZIP']]
    pct_cols = [c for c in trends.columns if '% Change' in c]
    
    hist_features = []
    for _, row in trends.iterrows():
        z = row['ZIP']
        wages = [row[c] for c in wage_cols if pd.notna(row[c]) and row[c] > 0]
        pcts = [row[c] for c in pct_cols if pd.notna(row[c])]
        
        if len(wages) < 2:
            hist_features.append({
                'ZIP': z, 'HCF_calc': 0, 'MHJ_calc': 0,
                'Avg_Pct_Change': 0, 'Wage_Volatility': 0, 'Recent_Change_Rate': 0
            })
            continue
        
        changes = sum(1 for i in range(1, len(wages)) if wages[i] != wages[i - 1])
        years = len(wage_cols) / 2
        recent = pcts[-4:] if len(pcts) >= 4 else pcts
        
        hist_features.append({
            'ZIP': z,
            'HCF_calc': changes / years if years > 0 else 0,
            'MHJ_calc': max(pcts) if pcts else 0,
            'Avg_Pct_Change': float(np.mean(pcts)) if pcts else 0,
            'Wage_Volatility': float(np.std(pcts)) if len(pcts) > 1 else 0,
            'Recent_Change_Rate': float(np.mean(recent)) if recent else 0,
        })
    
    hist_df = pd.DataFrame(hist_features)
    
    # Merge all features to ZIP level
    df = zip_features.copy()
    for feat_df in [city_count, county_count, wage_complexity,
                    wage_stats, future_stats, carveouts]:
        df = df.merge(feat_df, on='State', how='left')
    df = df.merge(hist_df, on='ZIP', how='left')
    
    print(f"   ✓ Built features for {len(df):,} ZIPs")
    
    return {'ml_features': df, 'full_tracker': full_tracker, 'trends': trends}


# ==============================================================================
# SECTION 3: ADD ERI, CBSA, AND FRONTLINE FEATURES
# ==============================================================================

def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ERI Cost of Labor, CBSA metro, and Frontline differential features."""
    print("\n📂 Adding external data features...")
    
    # ERI Cost of Labor
    if os.path.exists(Config.ERI_FILE):
        eri = pd.read_excel(Config.ERI_FILE, sheet_name='Comparison List - Labor', skiprows=6)
        eri['ZIP'] = eri['PostCode'].apply(standardize_zip)
        eri_sub = eri[['ZIP', 'Cost of Labor Average', 'COL Market Measure']].copy()
        eri_sub.columns = ['ZIP', 'Cost_of_Labor_Index', 'COL_Market_Measure']
        df = df.merge(eri_sub, on='ZIP', how='left')
        df['COL_Above_Baseline'] = (df['COL_Market_Measure'] == 'Above').astype(int)
        print(f"   ✓ ERI: {len(eri):,} records merged")
    else:
        df['Cost_of_Labor_Index'] = 100.0
        df['COL_Above_Baseline'] = 0
        print("   ⚠ ERI file not found, using defaults")
    
    # CBSA Metropolitan Area
    if os.path.exists(Config.CBSA_FILE):
        cbsa = pd.read_excel(Config.CBSA_FILE, sheet_name='CBSA April 2025')
        cbsa['ZIP'] = cbsa['ZIP'].apply(standardize_zip)
        cbsa['Is_Metro'] = (cbsa['Type'] == 'M').astype(int)
        df = df.merge(cbsa[['ZIP', 'Is_Metro']], on='ZIP', how='left')
        df['Is_Metro'] = df['Is_Metro'].fillna(0).astype(int)
        print(f"   ✓ CBSA: {len(cbsa):,} records merged")
    else:
        df['Is_Metro'] = 0
        print("   ⚠ CBSA file not found, using defaults")
    
    # Frontline Differential Rate
    if os.path.exists(Config.FRONTLINE_FILE):
        fl = pd.read_excel(Config.FRONTLINE_FILE,
                           sheet_name='Current FL Diff Rate', skiprows=1)
        fl['ZIP'] = fl['PostCode'].apply(standardize_zip)
        diff_col = 'Fall 2025 Differential Rate - Salary (CBSA Max Method)'
        rank_col = 'Diff Rank Class'
        if diff_col in fl.columns:
            fl_sub = fl[['ZIP', diff_col, rank_col]].copy()
            fl_sub.columns = ['ZIP', 'Differential_Rate', 'Diff_Rank_Class']
            rank_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
            fl_sub['Diff_Rank_Num'] = fl_sub['Diff_Rank_Class'].map(rank_map).fillna(2)
            df = df.merge(fl_sub[['ZIP', 'Differential_Rate', 'Diff_Rank_Num']],
                          on='ZIP', how='left')
            print(f"   ✓ Frontline: {len(fl):,} records merged")
    else:
        df['Differential_Rate'] = 100.0
        df['Diff_Rank_Num'] = 2
        print("   ⚠ Frontline file not found, using defaults")
    
    # Fill missing numerics
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['Wage_Change_Pct', 'Wage_Change_Abs']:
            df[col] = df[col].fillna(0)
    
    return df


# ==============================================================================
# SECTION 4: v19c ML MODEL — COMPOSITE RISK TARGET + TRAINING
# ==============================================================================

def create_risk_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the composite risk target for the ML model to learn.
    
    4 COMPONENTS:
    1. FINANCIAL RISK (35 pts): Scheduled wage changes
    2. OPERATIONAL RISK (35 pts): Structural complexity
    3. ECONOMIC PRESSURE (20 pts): High wages + COL
    4. HISTORICAL VOLATILITY (10 pts): Past change patterns
    
    California scores high on OPERATIONAL (not financial) because of
    44 industry carveouts, 1241 city jurisdictions, 29 unique wage levels.
    """
    print("\n🎯 Creating composite risk target...")
    
    # Financial Risk (35 pts max)
    df['Risk_Financial'] = (
        df['Has_Scheduled_Increase'] * 15 +
        norm(df['Wage_Change_Pct']) * 15 +
        norm(df.get('HCF_calc', df.get('HCF', pd.Series(0, index=df.index)))) * 5
    )
    
    # Operational Risk (35 pts max) — THIS IS WHERE CALIFORNIA SCORES HIGH
    df['Risk_Operational'] = (
        norm(df['City_Jurisdiction_Count']) * 12 +
        norm(df['Industry_Carveout_Count']) * 12 +
        norm(df['Wage_Complexity']) * 8 +
        norm(df['Wage_Range']) * 3
    )
    
    # Economic Pressure (20 pts max)
    df['Risk_Economic'] = (
        norm(df['Current_Wage']) * 8 +
        norm(df['State_Max_Wage']) * 6 +
        norm(df['Cost_of_Labor_Index'].fillna(100)) * 6
    )
    
    # Historical Volatility (10 pts max)
    df['Risk_Historical'] = (
        norm(df.get('MHJ_calc', df.get('MHJ', pd.Series(0, index=df.index)))) * 5 +
        norm(df['Wage_Volatility']) * 5
    )
    
    # Total (0-100)
    df['Target_Risk_Score'] = (
        df['Risk_Financial'] + df['Risk_Operational'] +
        df['Risk_Economic'] + df['Risk_Historical']
    )
    t_min, t_max = df['Target_Risk_Score'].min(), df['Target_Risk_Score'].max()
    if t_max > t_min:
        df['Target_Risk_Score'] = (df['Target_Risk_Score'] - t_min) / (t_max - t_min) * 100
    
    print(f"   Score range: {df['Target_Risk_Score'].min():.1f} - {df['Target_Risk_Score'].max():.1f}")
    print(f"   Mean: {df['Target_Risk_Score'].mean():.1f}, Median: {df['Target_Risk_Score'].median():.1f}")
    
    return df


def train_ensemble(df: pd.DataFrame) -> tuple:
    """Train ML ensemble and return best model with predictions."""
    print("\n🤖 Training ML ensemble...")
    
    exclude = ['ZIP', 'State', 'Target_Risk_Score',
               'Risk_Financial', 'Risk_Operational', 'Risk_Economic', 'Risk_Historical',
               'Wage_Change_Pct', 'Wage_Change_Abs', 'Has_Scheduled_Increase',
               'City_Wage', 'County_Wage', 'Future_Wage',
               'COL_Market_Measure', 'Diff_Rank_Class', 'CBSA_Code']
    
    features = [c for c in df.columns
                if c not in exclude and df[c].dtype in ['int64', 'float64']]
    
    X = df[features].fillna(0)
    y = df['Target_Risk_Score']
    
    print(f"   Features: {len(features)}, Samples: {len(X):,}")
    
    models = {
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, learning_rate=0.1,
            min_samples_leaf=10, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=10, random_state=RANDOM_STATE),
    }
    
    if _HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_child_weight=10, random_state=RANDOM_STATE, verbosity=0)
    if _HAS_CAT:
        models['CatBoost'] = CatBoostRegressor(
            iterations=200, depth=5, learning_rate=0.1,
            min_data_in_leaf=10, random_seed=RANDOM_STATE, verbose=False)
    if _HAS_LGB:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_child_samples=10, random_state=RANDOM_STATE, verbose=-1)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        model.fit(X, y)
        y_pred = model.predict(X)
        
        importance_df = None
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        result = ModelResult(
            name, model,
            r2=r2_score(y, y_pred),
            r2_cv=cv_scores.mean(),
            rmse=np.sqrt(mean_squared_error(y, y_pred)),
            mae=mean_absolute_error(y, y_pred),
            feature_importance=importance_df
        )
        results.append(result)
        print(f"   {name}: R²(train)={result.r2:.4f}, R²(CV)={result.r2_cv:.4f}")
    
    best = max(results, key=lambda x: x.r2_cv)
    print(f"\n   ✅ Best: {best.name} (R²_CV={best.r2_cv:.4f})")
    
    df['Predicted_Risk'] = best.model.predict(X)
    
    return best, df, features


# ==============================================================================
# SECTION 5: 3D STATE-LEVEL SCORING
# ==============================================================================

def calculate_3d_state_scores(mw_data: Dict, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 3D state-level risk scores using the validated formulas.
    
    Sub_State_Complexity = (City_Jurisdictions × 3) + (County_Jurisdictions × 2)
                         + (Unique_Min_Wages × 1.5) + (Industry_Carveouts × 2)
    
    Forward_Looking = (Future_Change_Count × 2)
                    + (15 if Max_Future_Wage > $20)
                    + (15 if Max_Future_Wage > $25)
    
    Sustained_Pressure = (Consecutive_Years_Increasing × 8) + (Wage_Range × 2)
    """
    print("\n📐 Calculating 3D state-level risk scores...")
    
    ml_features = mw_data['ml_features']
    
    # Build state-level feature table
    state_features = ml_features.groupby('State').agg({
        'City_Jurisdiction_Count': 'first',
        'County_Jurisdiction_Count': 'first',
        'Wage_Complexity': 'first',
        'Industry_Carveout_Count': 'first',
        'Future_Change_Count': 'first',
        'State_Max_Future_Wage': 'first',
        'Wage_Range': 'first',
        'State_Max_Wage': 'first',
        'State_Min_Wage': 'first',
        'ZIP': 'count',
    }).reset_index()
    
    state_features.columns = ['State', 'City_Jurisdictions', 'County_Jurisdictions',
                              'Unique_Min_Wages', 'Industry_Carveouts',
                              'Future_Change_Count', 'Max_Future_Wage',
                              'Wage_Range', 'Current_Max_Wage', 'Current_Min_Wage',
                              'ZIP_Count']
    
    # Add consecutive years from manual table
    # Map state abbreviation to full name for lookup
    abbr_to_name = dict(zip(base_df['State_Abbr'].dropna(), base_df['State_Name'].dropna()))
    state_features['Consecutive_Years_Increasing'] = state_features['State'].map(
        lambda abbr: SUSTAINED_PRESSURE.get(abbr_to_name.get(abbr, ''), 0)
    )
    
    # Calculate 3D sub-scores
    state_features['Sub_State_Complexity_Score'] = (
        state_features['City_Jurisdictions'] * 3 +
        state_features['County_Jurisdictions'] * 2 +
        state_features['Unique_Min_Wages'] * 1.5 +
        state_features['Industry_Carveouts'] * 2
    )
    
    state_features['Forward_Looking_Score'] = (
        state_features['Future_Change_Count'] * 2 +
        np.where(state_features['Max_Future_Wage'] > 20, 15, 0) +
        np.where(state_features['Max_Future_Wage'] > 25, 15, 0)
    )
    
    state_features['Sustained_Pressure_Score'] = (
        state_features['Consecutive_Years_Increasing'] * 8 +
        state_features['Wage_Range'] * 2
    )
    
    print(f"   ✓ Calculated 3D scores for {len(state_features)} states")
    
    # Show top 10
    top10 = state_features.nlargest(10, 'Sub_State_Complexity_Score')
    for _, row in top10.iterrows():
        total = (row['Sub_State_Complexity_Score'] + row['Forward_Looking_Score'] +
                 row['Sustained_Pressure_Score'])
        print(f"   {row['State']}: SubState={row['Sub_State_Complexity_Score']:.1f}, "
              f"Forward={row['Forward_Looking_Score']:.1f}, "
              f"Sustained={row['Sustained_Pressure_Score']:.1f}, Total={total:.1f}")
    
    return state_features


# ==============================================================================
# SECTION 6: COUNTY RISK ADJUSTMENTS
# ==============================================================================

def calculate_county_adjustments(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate county-level risk adjustments based on wage variation.
    
    Counties with higher wages than state average get positive adjustment.
    Counties with faster 5-year growth get positive adjustment.
    """
    print("\n🏛️ Calculating county-level risk adjustments...")
    
    # Need wage columns for county calculation
    wage_cols_needed = ['wage_jan2019', 'wage_jan2024']
    has_wages = all(c in base_df.columns for c in wage_cols_needed)
    
    if not has_wages:
        print("   ⚠ Wage history columns not found, skipping county adjustments")
        return pd.DataFrame()
    
    county_metrics = base_df.groupby(['State_Name', 'county']).agg({
        'zip': 'count',
        'wage_jan2019': 'mean',
        'wage_jan2024': 'mean',
    }).reset_index()
    county_metrics.columns = ['State_Name', 'county', 'County_ZIP_Count',
                              'wage_2019', 'wage_2024']
    
    # 5-year wage change
    county_metrics['County_Wage_Change_5yr'] = (
        (county_metrics['wage_2024'] - county_metrics['wage_2019']) /
        county_metrics['wage_2019'] * 100
    ).fillna(0)
    
    # State average wage for comparison
    state_avg = base_df.groupby('State_Name')['wage_jan2024'].mean().reset_index()
    state_avg.columns = ['State_Name', 'State_Avg_Wage_2024']
    county_metrics = county_metrics.merge(state_avg, on='State_Name', how='left')
    
    # County wage premium
    county_metrics['County_Wage_Premium_Pct'] = (
        (county_metrics['wage_2024'] - county_metrics['State_Avg_Wage_2024']) /
        county_metrics['State_Avg_Wage_2024'] * 100
    ).fillna(0)
    
    # County risk adjustment
    max_premium = county_metrics['County_Wage_Premium_Pct'].abs().replace(0, 1).max()
    max_growth = county_metrics['County_Wage_Change_5yr'].replace(0, 1).max()
    
    county_metrics['County_Risk_Adjustment'] = (
        (county_metrics['County_Wage_Premium_Pct'] / max_premium * 10) +
        (county_metrics['County_Wage_Change_5yr'] / max_growth * 5)
    ).fillna(0)
    
    print(f"   ✓ Calculated adjustments for {len(county_metrics):,} counties")
    
    return county_metrics[['State_Name', 'county', 'County_ZIP_Count',
                           'County_Wage_Premium_Pct', 'County_Wage_Change_5yr',
                           'County_Risk_Adjustment']]


# ==============================================================================
# SECTION 7: ASSEMBLE FINAL CSV (61 COLUMNS)
# ==============================================================================

def assemble_final_csv(
    base_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    state_scores: pd.DataFrame,
    county_adj: pd.DataFrame
) -> pd.DataFrame:
    """
    Assemble the final MWR_Combined_ZipCode_Risk_v2.csv with all 61 columns.
    
    Combines:
    - Base ZIP data (from Neo4j/CSV)
    - ML model predictions (v19c)
    - 3D state scores
    - County adjustments
    """
    print("\n🔧 Assembling final CSV...")
    
    # Start with base data
    final = base_df.copy()
    
    # --- ML MODEL SCORES ---
    # Map v19c predictions to base ZIPs
    ml_scores = ml_df[['ZIP', 'Predicted_Risk']].copy()
    ml_scores.columns = ['zip', 'New_Risk_Score_Pct']
    final = final.merge(ml_scores, on='zip', how='left')
    
    # Scale to 0-500 for Power BI
    final['New_Combined_Risk_Score'] = final['New_Risk_Score_Pct'] * 5
    
    # Risk tiers (on 0-500 scale)
    def get_tier_500(score):
        if pd.isna(score):
            return np.nan
        if score >= 350:
            return 'Critical'
        elif score >= 275:
            return 'High'
        elif score >= 200:
            return 'Elevated'
        elif score >= 125:
            return 'Moderate'
        else:
            return 'Low'
    
    def get_color(tier):
        colors = {
            'Critical': 'RED', 'High': 'ORANGE', 'Elevated': 'YELLOW',
            'Moderate': 'LIGHT_GREEN', 'Low': 'GREEN'
        }
        return colors.get(tier, np.nan)
    
    def get_color_hex(tier):
        hexes = {
            'Critical': '#FF0000', 'High': '#FFA500', 'Elevated': '#FFFF00',
            'Moderate': '#90EE90', 'Low': '#008000'
        }
        return hexes.get(tier, np.nan)
    
    def get_level_num(tier):
        levels = {'Critical': 5, 'High': 4, 'Elevated': 3, 'Moderate': 2, 'Low': 1}
        return levels.get(tier, np.nan)
    
    final['New_Risk_Tier'] = final['New_Combined_Risk_Score'].apply(get_tier_500)
    final['New_Risk_Color'] = final['New_Risk_Tier'].apply(get_color)
    final['Risk_Level_Numeric'] = final['New_Risk_Tier'].apply(get_level_num)
    final['Risk_Color_Hex'] = final['New_Risk_Tier'].apply(get_color_hex)
    
    # --- 3D STATE SCORES ---
    state_merge_cols = [
        'State', 'Sub_State_Complexity_Score', 'Forward_Looking_Score',
        'Sustained_Pressure_Score', 'ZIP_Count', 'Unique_Min_Wages',
        'Current_Min_Wage', 'Current_Max_Wage', 'City_Jurisdictions',
        'County_Jurisdictions', 'Future_Change_Count', 'Max_Future_Wage',
        'Consecutive_Years_Increasing', 'Industry_Carveouts'
    ]
    
    # Rename State to State_Abbr for merging
    state_scores_merge = state_scores[
        [c for c in state_merge_cols if c in state_scores.columns]
    ].copy()
    state_scores_merge = state_scores_merge.rename(columns={'State': 'State_Abbr'})
    
    final = final.merge(state_scores_merge, on='State_Abbr', how='left')
    
    # --- COUNTY ADJUSTMENTS ---
    if len(county_adj) > 0:
        final = final.merge(county_adj, on=['State_Name', 'county'], how='left')
        
        # County risk score = state pct + adjustment, clipped to 0-100
        final['County_Risk_Score_Pct'] = (
            final['New_Risk_Score_Pct'] + final['County_Risk_Adjustment']
        ).clip(0, 100)
        
        # County tier
        def get_tier_100(score):
            if pd.isna(score):
                return np.nan
            if score >= 70:
                return 'Critical'
            elif score >= 55:
                return 'High'
            elif score >= 40:
                return 'Elevated'
            elif score >= 25:
                return 'Moderate'
            else:
                return 'Low'
        
        final['County_Risk_Tier'] = final['County_Risk_Score_Pct'].apply(get_tier_100)
        final['County_Risk_Color'] = final['County_Risk_Tier'].apply(get_color_hex)
    else:
        for col in ['County_Risk_Score_Pct', 'County_Risk_Tier', 'County_Risk_Color',
                     'County_Risk_Adjustment', 'County_Wage_Premium_Pct',
                     'County_Wage_Change_5yr', 'County_ZIP_Count']:
            final[col] = np.nan
    
    # --- ENSURE CORRECT COLUMN ORDER (61 columns) ---
    column_order = [
        'zip', '_NodeId_', 'preferredCity', 'county', 'State_Abbr', 'State_Name',
        'StateCounty', 'fips', 'areaCode', 'cbsaCode', 'timeZone',
        'abbr', 'federalOrState', 'hasLocalRates', 'HasVolatility',
        'Old_RiskScore', 'Old_RiskTier', 'Old_RiskTier5', 'Old_VolatilityScore',
        'HCF', 'MHJ', 'UCP', 'IDX', 'LRF', 'HCFxUCP', 'MHJxIDX', 'topDriver',
        'wage_jan2019', 'wage_july2019', 'wage_jan2020', 'wage_jan2021',
        'wage_jan2022', 'wage_jan2023', 'wage_july2023', 'wage_jan2024',
        'New_Combined_Risk_Score', 'New_Risk_Score_Pct', 'New_Risk_Tier',
        'New_Risk_Color', 'Risk_Level_Numeric', 'Risk_Color_Hex',
        'Sub_State_Complexity_Score', 'Forward_Looking_Score', 'Sustained_Pressure_Score',
        'ZIP_Count', 'Unique_Min_Wages', 'Current_Min_Wage', 'Current_Max_Wage',
        'City_Jurisdictions', 'County_Jurisdictions', 'Future_Change_Count',
        'Max_Future_Wage', 'Consecutive_Years_Increasing', 'Industry_Carveouts',
        'County_Risk_Score_Pct', 'County_Risk_Tier', 'County_Risk_Color',
        'County_Risk_Adjustment', 'County_Wage_Premium_Pct', 'County_Wage_Change_5yr',
        'County_ZIP_Count'
    ]
    
    # Only keep columns that exist
    existing_cols = [c for c in column_order if c in final.columns]
    missing_cols = [c for c in column_order if c not in final.columns]
    
    if missing_cols:
        print(f"   ⚠ Missing columns (will be NaN): {missing_cols}")
        for col in missing_cols:
            final[col] = np.nan
    
    final = final[column_order]
    
    print(f"\n   ✓ Final CSV: {len(final):,} rows, {len(final.columns)} columns")
    
    return final


# ==============================================================================
# SECTION 8: VALIDATION
# ==============================================================================

def validate_output(df: pd.DataFrame):
    """Validate the output matches expected patterns."""
    print("\n" + "=" * 70)
    print("🔍 VALIDATION")
    print("=" * 70)
    
    # Basic checks
    print(f"\n   Rows: {len(df):,} (expected: ~41,554)")
    print(f"   Columns: {len(df.columns)} (expected: 61)")
    
    # Tier distribution
    print(f"\n   Risk Tier Distribution:")
    tier_counts = df['New_Risk_Tier'].value_counts()
    for tier in ['Critical', 'High', 'Elevated', 'Moderate', 'Low']:
        count = tier_counts.get(tier, 0)
        print(f"      {tier}: {count:,} ({count / len(df) * 100:.1f}%)")
    
    # State validation
    valid = df.dropna(subset=['New_Combined_Risk_Score'])
    state_avg = valid.groupby('State_Name')['New_Combined_Risk_Score'].mean(
    ).sort_values(ascending=False)
    
    print(f"\n   Top 10 States by Average Risk Score:")
    for state, score in state_avg.head(10).items():
        print(f"      {state}: {score:.1f}")
    
    # California check
    ca_score = state_avg.get('California', 0)
    ca_rank = list(state_avg.index).index('California') + 1 if 'California' in state_avg.index else 'N/A'
    print(f"\n   California: Score={ca_score:.1f}, Rank=#{ca_rank}")
    if isinstance(ca_rank, int) and ca_rank <= 3:
        print("   ✅ California is in top 3 — CORRECT")
    else:
        print("   ⚠️ California ranking may need review")
    
    # NaN check
    nan_count = df['New_Combined_Risk_Score'].isna().sum()
    print(f"\n   ZIPs with missing scores: {nan_count} ({nan_count / len(df) * 100:.1f}%)")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    print("=" * 70)
    print("MWR AUTOMATED RE-SCORING PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Mode: {Config.MODE}")
    
    # STEP 1: Load base ZIP data
    if _HAS_NEO4J and Config.NEO4J_PASSWORD:
        base_df = load_base_zip_data_from_neo4j()
    elif os.path.exists(Config.OUTPUT_CSV):
        base_df = load_base_zip_data_from_csv(Config.OUTPUT_CSV)
    else:
        print("❌ No data source available (Neo4j or existing CSV)")
        sys.exit(1)
    
    # STEP 2: Load MW tracker features for ML model
    mw_data = load_mw_tracker_features()
    
    # STEP 3: Add external features (ERI, CBSA, Frontline)
    ml_df = add_external_features(mw_data['ml_features'])
    
    # STEP 4: Create risk target + train ML ensemble
    ml_df = create_risk_target(ml_df)
    ml_df = ml_df.dropna(subset=['Target_Risk_Score'])
    best, ml_df, features = train_ensemble(ml_df)
    
    # STEP 5: Calculate 3D state-level scores
    state_scores = calculate_3d_state_scores(mw_data, base_df)
    
    # STEP 6: Calculate county adjustments
    county_adj = calculate_county_adjustments(base_df)
    
    # STEP 7: Assemble final CSV
    final = assemble_final_csv(base_df, ml_df, state_scores, county_adj)
    
    # STEP 8: Save output
    output_path = Config.OUTPUT_CSV
    final.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
    
    # STEP 9: Validate
    validate_output(final)
    
    # STEP 10: Save model metrics
    metrics = {
        'timestamp': TIMESTAMP,
        'best_model': best.name,
        'r2_train': round(best.r2, 4),
        'r2_cv': round(best.r2_cv, 4),
        'rmse': round(best.rmse, 4),
        'total_zips': len(final),
        'features_used': len(features),
    }
    with open('mwr_rescore_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n📊 Metrics saved to mwr_rescore_metrics.json")
    
    print("\n" + "=" * 70)
    print("✅ MWR RE-SCORING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
