import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="Characterize baseline clinical subtypes using longitudinal imaging data")
    # By default, use the strictly validated merge file
    parser.add_argument("--enriched-csv", default=ROOT.parent / "Ressources" / "PPMI_Global_Enriched_Visits.csv")
    # By default, use the canonical longitudinal assignments from the subscores pipeline (02) or extended (05)
    # We fallback to extended_pipeline if present, otherwise specify path via CLI
    parser.add_argument("--assignments-csv", default=ROOT / "results" / "sustain" / "longitudinal" / "extended_subscores_longitudinal_assignments.csv")
    parser.add_argument("--out-dir", default=ROOT / "results" / "imaging_validation")
    return parser.parse_args()

def main():
    args = parse_args()
    enriched_path = Path(args.enriched_csv)
    assign_path = Path(args.assignments_csv)
    out_dir = Path(args.out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not assign_path.exists():
        print(f"[!] Warning: Assignments file not found at {assign_path}.")
        print("Please point --assignments-csv to the correct SuStaIn output CSV (e.g., from phase 02 or 04).")
        return
        
    print(f"Loading subtype assignments from: {assign_path.name}")
    df_assign = pd.read_csv(assign_path)
    
    # A patient's clinical subtype is defined by their baseline/overall trajectory in SuStaIn
    # We take the mode (most frequent assignment across visits) to define their "True" Clinical Subtype
    mode_subtype = df_assign.groupby('PATNO')['Subtype'].agg(lambda x: x.mode()[0]).reset_index()
    mode_subtype.rename(columns={'Subtype': 'Clinical_Subtype'}, inplace=True)
    
    print(f"Loading enriched visits from: {enriched_path.name}")
    df_visits = pd.read_csv(enriched_path, low_memory=False)
    
    # Filter target visits (Month 12, 24, 36, 48)
    target_visits = ['V04', 'V06', 'V08', 'V10']
    df_target = df_visits[df_visits['EVENT_ID'].isin(target_visits)].copy()
    
    # Merge imaging data with the patient's Clinical Subtype assigned at baseline
    df_merged = df_target.merge(mode_subtype, on='PATNO', how='inner')
    
    print(f"Total matching patients with imaging visits: {df_merged['PATNO'].nunique()}")
    
    # Define potential imaging features from the dataset
    raw_img_features = [
        'DATSCAN_CAUDATE_R', 'DATSCAN_CAUDATE_L', 
        'DATSCAN_PUTAMEN_R', 'DATSCAN_PUTAMEN_L', 
        'Moyenne_SN_R', 'Moyenne_SN_L', 
        'Moyenne_Caudate_R', 'Moyenne_Caudate_L',
        'Moyenne_Putamen_R', 'Moyenne_Putamen_L'
    ]
    
    img_features = [f for f in raw_img_features if f in df_merged.columns]
    
    # Create average (Mean of Left & Right) for simplified visualization
    if 'DATSCAN_CAUDATE_R' in img_features and 'DATSCAN_CAUDATE_L' in img_features:
        df_merged['DAT_CAUDATE_MEAN'] = df_merged[['DATSCAN_CAUDATE_R', 'DATSCAN_CAUDATE_L']].mean(axis=1)
        img_features.append('DAT_CAUDATE_MEAN')
        
    if 'DATSCAN_PUTAMEN_R' in img_features and 'DATSCAN_PUTAMEN_L' in img_features:
        df_merged['DAT_PUTAMEN_MEAN'] = df_merged[['DATSCAN_PUTAMEN_R', 'DATSCAN_PUTAMEN_L']].mean(axis=1)
        img_features.append('DAT_PUTAMEN_MEAN')
        
    if 'Moyenne_SN_R' in img_features and 'Moyenne_SN_L' in img_features:
        df_merged['R2_SN_MEAN'] = df_merged[['Moyenne_SN_R', 'Moyenne_SN_L']].mean(axis=1)
        img_features.append('R2_SN_MEAN')

    sns.set_theme(style="whitegrid", palette="muted")
    
    results = []
    
    for feat in img_features:
        print(f" -> Mapping and testing feature: {feat}")
        
        # 1. Boxplots over time
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_merged, 
            x='EVENT_ID', 
            y=feat, 
            hue='Clinical_Subtype', 
            order=target_visits, 
            showfliers=False
        )
        sns.stripplot(
            data=df_merged, 
            x='EVENT_ID', 
            y=feat, 
            hue='Clinical_Subtype', 
            order=target_visits, 
            dodge=True, 
            alpha=0.4, 
            color='k', 
            legend=False
        )
        
        plt.title(f"External Validation: {feat} across future visits\n(Grouped by Baseline Clinical Subtype)")
        plt.xlabel("Visit (V04=M12, V06=M24, V08=M36, V10=M48)")
        plt.ylabel(feat)
        plt.legend(title="Subtype")
        plt.tight_layout()
        plt.savefig(out_dir / f"{feat}_longitudinal_boxplot.png", dpi=150)
        plt.close()
        
        # 2. Statistical testing per visit
        for visit in target_visits:
            v_data = df_merged[df_merged['EVENT_ID'] == visit].dropna(subset=[feat, 'Clinical_Subtype'])
            if v_data.empty:
                continue
            
            subtypes = sorted(v_data['Clinical_Subtype'].unique())
            if len(subtypes) > 1:
                # Kruskal-Wallis text across subtypes
                groups = [v_data[v_data['Clinical_Subtype'] == s][feat].values for s in subtypes]
                stat, p_val = kruskal(*groups)
                
                results.append({
                    "Feature": feat,
                    "Visit": visit,
                    "N_Patients": len(v_data),
                    "Kruskal_Stat": stat,
                    "p_value": p_val,
                    "Significant_05": p_val < 0.05,
                    "Significant_001": p_val < 0.01
                })
                
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(out_dir / "imaging_statistical_tests.csv", index=False)
        sig_count = df_res['Significant_05'].sum()
        print(f"\nStats saved to 'imaging_statistical_tests.csv'.")
        print(f"Found {sig_count} statistically significant differences (p < 0.05) out of {len(df_res)} tests.")
    else:
        print("\nNo statistics computed (missing data?).")
        
    print(f"\nDone. All plots and reports exported in:\n {out_dir}")

if __name__ == '__main__':
    main()