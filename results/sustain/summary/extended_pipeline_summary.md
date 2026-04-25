# Extended PPMI Pipeline Summary

## Fusion
- Full outer merge of `result_4_wide_format.csv` with `Fichier_PrincipalV61.xlsx` across all visits available in both sources.
- Patient-level demographics (EDUCYRS, PDDXDT, BIRTHDT) added from the General sheet.
- CSV values take priority for overlapping columns; Excel fills remaining NaN (covers ~60 Excel-only patients and BL visit).
- Computed imaging eligibility: patients with imaging available in at least 2 visits among V04/V06/V08.

## Integrated Sources
- Added RBDSQ, SCOPA-AUT, Epworth, medication, vital signs, and features of parkinsonism by PATNO+VISIT.
- Added QUIP-CS from `QUIP-Current-Short_13Mar2026.csv` (QUIP_ANY_ICD, QUIP_ICD_COUNT, per-domain flags).

## Outputs
- `extended_consolidated_long.csv`
- `extended_consolidated_wide.csv`
- `extended_ppmi_visit_workbook.xlsx` (1 sheet per visit + General)
- `extended_baseline_clustering_comparison.csv`
- `extended_sustain_methodology_results.csv`
- `extended_longitudinal_stability.csv`