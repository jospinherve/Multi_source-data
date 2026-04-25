"""
Extended PPMI fusion + clustering  analysis pipeline.

This script implements the requested workflow:
1) Merge Fichier_PrincipalV61.xlsx and result_4_wide_format.csv, including imaging.
2) Integrate additional CSV sources (RBDSQ, SCOPA-AUT, medication, vital signs,
   Epworth, features of parkinsonism).
3) Build consolidated long and wide datasets.
4) Compare methodological variants (totals, subscores, augmented, missforest, PCA)
   on baseline clustering.

"""
#%%
from __future__ import annotations

import json
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
#%%
# Make project root importable when script is launched from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VISITS_LONGITUDINAL = ["V04", "V06", "V08"]

#%%
def _visit_sort_key(visit: str) -> Tuple[int, object]:
    """Sort visits as BL, then Vxx, then Rxx, then others."""
    if visit == "SC":
        return (-1, 0)
    if visit == "BL":
        return (0, 0)
    m_v = re.fullmatch(r"V(\d{2})", visit) # V la lettre , \d chiffre et {2} exactement deux chiffres
    if m_v:
        return (1, int(m_v.group(1)))
    m_r = re.fullmatch(r"R(\d{2})", visit)
    if m_r:
        return (2, int(m_r.group(1)))
    return (3, visit)


def discover_visits_from_wide_csv(wide_path: Path) -> List[str]:
    """Extract available visit suffixes from result_4_wide_format.csv columns."""
    header = pd.read_csv(wide_path, sep=";", nrows=0)
    visit_re = re.compile(r"_(SC|BL|V\d{2}|R\d{2})$")
    visits = set()
    for col in header.columns:
        m = visit_re.search(str(col).upper())
        if m:
            visits.add(m.group(1))
    return sorted(visits, key=_visit_sort_key)


def discover_visits_from_excel(excel_path: Path) -> List[str]:
    """Extract available visit sheet names from Fichier_PrincipalV61.xlsx."""
    xl = pd.ExcelFile(excel_path)
    visit_re = re.compile(r"^(SC|BL|V\d{2}|R\d{2})$")
    visits = {str(sheet).strip().upper() for sheet in xl.sheet_names if visit_re.fullmatch(str(sheet).strip().upper())}
    return sorted(visits, key=_visit_sort_key)


def discover_fusion_visits(wide_path: Path, excel_path: Path) -> List[str]:
    """Union of all available visits in both primary PPMI sources."""
    csv_visits = set(discover_visits_from_wide_csv(wide_path))
    excel_visits = set(discover_visits_from_excel(excel_path))
    all_visits = csv_visits | excel_visits
    return sorted(all_visits, key=_visit_sort_key)


@dataclass
# Cette classe permet de stocker les résultats de l'analyse de clustering pour chaque ensemble de caractéristiques testé, y compris le nombre de caractéristiques utilisées, le nombre de sujets inclus, le meilleur nombre de clusters (k), et les scores d'évaluation (silhouette, calinski-harabasz, davies-bouldin).
class AnalysisResult:
    name: str
    clustering_method: str
    n_features: int
    n_subjects: int
    best_k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float

# Cette fonction permet de normaliser les identifiants de visite (EVENT_ID) en les convertissant en un format standardisé. Elle gère différents formats d'identifiants, tels que "BL" pour la visite de base, "Vxx" pour les visites régulières, et "Rxx" pour les visites à distance, en les mappant respectivement à "BL", "Vxx" et "Vxx". Si l'identifiant ne correspond à aucun de ces formats, la fonction retourne None.
def normalize_visit(event_id: object) -> Optional[str]:
    if pd.isna(event_id):
        return None
    value = str(event_id).strip().upper()
    if value in {"SC", "SCR", "SCREEN", "SCREENING"}:
        return "SC"
    if value == "BL":
        return "BL"
    if value.startswith("V") and len(value) >= 2:
        suffix = value[1:]
        if suffix.isdigit():
            return f"V{int(suffix):02d}"
    remote_map = {"R04": "V04", "R06": "V06", "R08": "V08"}
    if value in remote_map:
        return remote_map[value]
    return None

# Cette fonction permet de charger le fichier CSV "result_4_wide_format.csv" et de le transformer en format long, en extrayant les visites disponibles à partir des suffixes de colonnes. Pour chaque visite détectée, elle sélectionne les colonnes correspondantes, les renomme pour supprimer le suffixe de visite, ajoute une colonne "VISIT" indiquant la visite correspondante, et concatène tous les sous-ensembles pour obtenir un DataFrame long avec les colonnes "PATNO", "VISIT" et les autres variables associées à chaque visite.
def load_wide_long(wide_path: Path, visits: Sequence[str]) -> pd.DataFrame:
    wide = pd.read_csv(wide_path, sep=";", low_memory=False)
    pieces: List[pd.DataFrame] = []
    for visit in visits:
        suffix = f"_{visit}"
        cols = [c for c in wide.columns if c.endswith(suffix)]
        if not cols:
            continue
        subset = wide[["PATNO", *cols]].copy() # le *cols permet de sélectionner toutes les colonnes dont les noms sont dans la liste cols, en plus de la colonne "PATNO" qui est toujours incluse.
        rename = {c: c[: -len(suffix)] for c in cols}
        subset = subset.rename(columns=rename)
        subset["VISIT"] = visit
        pieces.append(subset)
    if not pieces:
        return pd.DataFrame(columns=["PATNO", "VISIT"])
    long_df = pd.concat(pieces, ignore_index=True)
    long_df["PATNO"] = pd.to_numeric(long_df["PATNO"], errors="coerce")
    long_df = long_df.dropna(subset=["PATNO"])
    long_df["PATNO"] = long_df["PATNO"].astype(int)
    return long_df


# Tokens that identify imaging/morphometry columns within Excel visit sheets.
_IMAGING_TOKENS = frozenset(
    ["DATSCAN", "CAUDATE", "PUTAMEN", "GM_VOLUME", "IMAGEID", "MOYENNE", "MEDIANE", "WGP", "WRN", "WSN"]
)


def load_excel_all_long(
    excel_path: Path, visits: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Load ALL clinical and imaging columns from Excel visit sheets into long format.

    Unlike the former ``load_excel_imaging_long``, this function retains every
    column present in a visit sheet (DISEASE_DURATION, NP3TOT_ON/OFF, DOPA,
    NHY_ON/OFF, STAI_TOTAL, DATScan, Moyenne/Mediane morphometry, …) so that
    the downstream merge is truly exhaustive.

    Returns
    -------
    long_df : pd.DataFrame
        All patients × all visits in long format (PATNO, VISIT, …).
    imaging_by_visit : dict
        Mapping visit → list of imaging column names (for eligibility logic).
    """
    rows: List[pd.DataFrame] = []
    imaging_by_visit: Dict[str, List[str]] = {}

    for visit in visits:
        try:
            df = pd.read_excel(excel_path, sheet_name=visit)
        except ValueError:
            continue
        if "PATNO" not in df.columns:
            continue
        df["PATNO"] = pd.to_numeric(df["PATNO"], errors="coerce")
        df = df.dropna(subset=["PATNO"]).copy()
        df["PATNO"] = df["PATNO"].astype(int)
        df["VISIT"] = visit
        # Drop internal Excel navigation column not needed downstream
        df = df.drop(columns=[c for c in ("EVENT_ID",) if c in df.columns])
        img_cols = [
            c
            for c in df.columns
            if c not in ("PATNO", "VISIT")
            and any(tok in str(c).upper() for tok in _IMAGING_TOKENS)
        ]
        imaging_by_visit[visit] = img_cols
        rows.append(df)

    if rows:
        long_df = pd.concat(rows, ignore_index=True)
    else:
        long_df = pd.DataFrame(columns=["PATNO", "VISIT"])
    return long_df, imaging_by_visit


def load_excel_general(excel_path: Path) -> pd.DataFrame:
    """Load patient-level demographics from the General sheet.

    Returns PATNO + EDUCYRS + PDDXDT + BIRTHDT — fields that are absent from
    the per-visit CSV data (``result_4_wide_format.csv``).
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="General")
    except ValueError:
        return pd.DataFrame(columns=["PATNO"])
    df["PATNO"] = pd.to_numeric(df["PATNO"], errors="coerce")
    df = df.dropna(subset=["PATNO"]).copy()
    df["PATNO"] = df["PATNO"].astype(int)
    keep = [c for c in df.columns if c in {"PATNO", "EDUCYRS", "PDDXDT", "BIRTHDT"}]
    return df[keep]

# cette focntion permet de calculer l'éligibilité à l'imagerie pour chaque patient en fonction des colonnes d'imagerie disponibles dans le DataFrame fusionné. Elle sélectionne les colonnes d'imagerie pertinentes, vérifie si chaque visite contient des données d'imagerie non nulles, puis agrège ces informations au niveau du patient pour déterminer combien de visites contiennent des données d'imagerie et si le patient est éligible (au moins 2 visites avec imagerie).
def compute_imaging_eligibility(merged_long: pd.DataFrame, imaging_cols: Sequence[str]) -> pd.DataFrame:
    use_cols = [c for c in imaging_cols if c in merged_long.columns]
    if not use_cols:
        return pd.DataFrame(columns=["PATNO", "n_visits_with_imaging", "eligible_2plus"])
    subset = merged_long[merged_long["VISIT"].isin(VISITS_LONGITUDINAL)].copy()
    subset["has_imaging"] = subset[use_cols].notna().any(axis=1)
    out = (
        subset.groupby("PATNO", as_index=False)["has_imaging"]
        .sum()
        .rename(columns={"has_imaging": "n_visits_with_imaging"})
    )
    out["eligible_2plus"] = out["n_visits_with_imaging"] >= 2 # out contient une colonne "n_visits_with_imaging" qui indique le nombre de visites avec données d'imagerie pour chaque patient, et une colonne "eligible_2plus" qui est un booléen indiquant si le patient est éligible (True si au moins 2 visites avec imagerie, False sinon).
    return out

# Cette fonction permet de  calculer la première valeur non nulle d'une série pandas, en supprimant d'abord les valeurs nulles (NaN) de la série, puis en retournant la première valeur restante. Si la série est vide après avoir supprimé les valeurs nulles, la fonction retourne NaN. 
def _first_non_null(series: pd.Series):
    series = series.dropna()
    return series.iloc[0] if not series.empty else np.nan

# cette fonction permet de charger une feuille d'un classeur Excel en format long, en fusionnant les différentes feuilles correspondant à différentes visites. Elle lit chaque feuille du classeur, vérifie si elle correspond à une visite valide (BL, Vxx, Rxx), extrait les données pertinentes, ajoute une colonne "VISIT" indiquant la visite correspondante, puis concatène tous les sous-ensembles pour obtenir un DataFrame long avec les colonnes "PATNO", "VISIT" et les autres variables associées à chaque visite. Si aucune feuille valide n'est trouvée, la fonction tente de trouver une feuille avec une colonne de visite explicite et retourne cette feuille.
def aggregate_prefixed_source(
    path: Path,
    prefix: str,
    numeric_cols: Sequence[str],
    text_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate an attached source file at PATNO+VISIT level with prefixed columns."""
    df = pd.read_csv(path, low_memory=False)
    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    df = df.dropna(subset=["VISIT"]).copy()

    agg_spec: Dict[str, object] = {}
    rename: Dict[str, str] = {}

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg_spec[col] = "mean"
            rename[col] = f"{prefix}_{col}"

    for col in text_cols:
        if col in df.columns:
            agg_spec[col] = _first_non_null # pour les colonnes textuelles, on prend la première valeur non nulle (si plusieurs lignes par PATNO+VISIT) — on suppose que ces champs sont constants ou quasi-constants à travers les lignes d'une même visite.
            rename[col] = f"{prefix}_{col}"

    if not agg_spec:
        return pd.DataFrame(columns=["PATNO", "VISIT"])

    out = df.groupby(["PATNO", "VISIT"], as_index=False).agg(agg_spec)
    return out.rename(columns=rename)

# Cette fonction permet d'agréger les données du questionnaire RBDSQ en calculant un score total à partir de plusieurs items individuels. Elle lit le fichier CSV, identifie les colonnes correspondant aux items du RBDSQ, convertit ces colonnes en valeurs numériques, calcule un score total en sommant les items, normalise les visites à l'aide de la fonction `normalize_visit`, puis agrège les scores totaux au niveau de chaque patient et visite en prenant la moyenne (au cas où il y aurait plusieurs lignes par PATNO+VISIT). Le résultat est un DataFrame avec les colonnes "PATNO", "VISIT" et "RBDSQ_TOTAL".
def aggregate_rbdsq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith(("DRM", "SLP", "MVA", "RBD", "STROKE", "NARCOLEP"))]
    item_cols = [c for c in item_cols if c not in {"DRMREMEM"}]
    for col in item_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["RBDSQ_TOTAL"] = df[item_cols].sum(axis=1, min_count=1)
    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    out = (
        df.dropna(subset=["VISIT"])
        .groupby(["PATNO", "VISIT"], as_index=False)["RBDSQ_TOTAL"]
        .mean()
    )
    return out


def aggregate_scopa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    sc_cols = [c for c in df.columns if c.startswith("SCAU")]
    for col in sc_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SCOPA_AUT_TOTAL"] = df[sc_cols].sum(axis=1, min_count=1)
    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    out = (
        df.dropna(subset=["VISIT"])
        .groupby(["PATNO", "VISIT"], as_index=False)["SCOPA_AUT_TOTAL"]
        .mean()
    )
    return out


def aggregate_quip(path: Path) -> pd.DataFrame:
    """Aggregate QUIP-CS into visit-level screening indicators."""
    df = pd.read_csv(path, low_memory=False)
    icd_q1 = ["TMGAMBLE", "TMSEX", "TMBUY", "TMEAT"]
    icd_q2 = ["CNTRLGMB", "CNTRLSEX", "CNTRLBUY", "CNTRLEAT"]
    other_items = ["TMTORACT", "TMTMTACT", "TMTRWD"]
    med_items = ["TMDISMED", "CNTRLDSM"]

    for col in [*icd_q1, *icd_q2, *other_items, *med_items]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # In QUIP-CS coding, value 2 often means "Not Applicable" for DDS items.
    for col in med_items:
        if col in df.columns:
            df[col] = df[col].replace(2, np.nan)

    # Cette fonction permet de  calculer des indicateurs de screenings basés sur les données de QUIP-CS. Elle prend en entrée un DataFrame `df` et calcule les indicateurs suivants:
    def domain_flag(q1: str, q2: str) -> pd.Series:
        s1 = df[q1].fillna(0) if q1 in df.columns else 0
        s2 = df[q2].fillna(0) if q2 in df.columns else 0
        return ((s1 == 1) | (s2 == 1)).astype(float)  # retourne 

    df["QUIP_GAMBLING_POS"] = domain_flag("TMGAMBLE", "CNTRLGMB")
    df["QUIP_SEX_POS"] = domain_flag("TMSEX", "CNTRLSEX")
    df["QUIP_BUYING_POS"] = domain_flag("TMBUY", "CNTRLBUY")
    df["QUIP_EATING_POS"] = domain_flag("TMEAT", "CNTRLEAT")
    df["QUIP_ANY_ICD"] = (
        (df["QUIP_GAMBLING_POS"] == 1)
        | (df["QUIP_SEX_POS"] == 1)
        | (df["QUIP_BUYING_POS"] == 1)
        | (df["QUIP_EATING_POS"] == 1)
    ).astype(float)
    df["QUIP_ICD_COUNT"] = (
        df["QUIP_GAMBLING_POS"].fillna(0)
        + df["QUIP_SEX_POS"].fillna(0)
        + df["QUIP_BUYING_POS"].fillna(0)
        + df["QUIP_EATING_POS"].fillna(0)
    )

    hobby_cols = [c for c in other_items if c in df.columns]
    if hobby_cols:
        df["QUIP_HOBBYISM_PUNDING"] = (df[hobby_cols].fillna(0).max(axis=1) >= 1).astype(float)

    med_cols = [c for c in med_items if c in df.columns]
    if med_cols:
        df["QUIP_DDS"] = (df[med_cols].fillna(0).max(axis=1) >= 1).astype(float)

    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    keep_cols = [
        "QUIP_ANY_ICD",
        "QUIP_ICD_COUNT",
        "QUIP_GAMBLING_POS",
        "QUIP_SEX_POS",
        "QUIP_BUYING_POS",
        "QUIP_EATING_POS",
        "QUIP_HOBBYISM_PUNDING",
        "QUIP_DDS",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = (
        df.dropna(subset=["VISIT"])
        .groupby(["PATNO", "VISIT"], as_index=False)[keep_cols]
        .max()
    )
    return out


def resolve_source_path(
    ressources: Path, sources_dir: Path, filename: str, required: bool = True
) -> Optional[Path]:
    """Resolve a source CSV from Fichiers sources/, then Ressources/ as fallback."""
    candidates = [sources_dir / filename, ressources / filename]
    for path in candidates:
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"Missing source file: {filename}")
    return None


def aggregate_epworth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    ess_cols = [f"ESS{i}" for i in range(1, 9) if f"ESS{i}" in df.columns]
    for col in ess_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ESS_TOTAL"] = df[ess_cols].sum(axis=1, min_count=1)
    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    out = df.dropna(subset=["VISIT"]).groupby(["PATNO", "VISIT"], as_index=False)["ESS_TOTAL"].mean()
    return out


def aggregate_medication(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    numeric_cols = [c for c in ["TOTDDA", "LEDD", "LEDDOSE", "LEDDOSFRQ"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    df = df.dropna(subset=["VISIT"])
    agg = {c: "mean" for c in numeric_cols}
    out = df.groupby(["PATNO", "VISIT"], as_index=False).agg(agg)
    rename = {c: f"MED_{c}" for c in numeric_cols}
    out = out.rename(columns=rename)
    return out


def aggregate_vitals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    numeric_cols = [
        c
        for c in ["WGTKG", "HTCM", "TEMPC", "SYSSUP", "DIASUP", "HRSUP", "SYSSTND", "DIASTND", "HRSTND"]
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    out = (
        df.dropna(subset=["VISIT"])
        .groupby(["PATNO", "VISIT"], as_index=False)[numeric_cols]
        .mean()
        .rename(columns={c: f"VITAL_{c}" for c in numeric_cols})
    )
    return out


def aggregate_features_pd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    numeric_cols = [c for c in ["FEATBRADY", "FEATPOSINS", "FEATRIGID", "FEATTREMOR", "PSGLVL"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["VISIT"] = df["EVENT_ID"].map(normalize_visit)
    out = (
        df.dropna(subset=["VISIT"])
        .groupby(["PATNO", "VISIT"], as_index=False)[numeric_cols]
        .mean()
        .rename(columns={c: f"PARK_{c}" for c in numeric_cols})
    )
    return out


def merge_sources(base_long: pd.DataFrame, sources: Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged = base_long.copy()
    for src in sources:
        merged = merged.merge(src, on=["PATNO", "VISIT"], how="left")
    return merged


def to_numeric_frame(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def missforest_impute(df: pd.DataFrame, max_iter: int = 6, n_estimators: int = 120, random_state: int = 42) -> pd.DataFrame:
    import warnings
    from sklearn.exceptions import DataConversionWarning
    
    x = df.copy()
    for col in x.columns:
        med = x[col].median()
        if pd.isna(med):
            med = 0.0
        x[col] = x[col].fillna(med)

    original_nan = df.isna()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
        for _ in range(max_iter):
            for col in sorted(df.columns, key=lambda c: original_nan[c].sum()):
                miss_mask = original_nan[col]
                if miss_mask.sum() == 0:
                    continue
                obs_mask = ~miss_mask
                if obs_mask.sum() < 20:
                    continue
                predictors = [c for c in x.columns if c != col]
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    max_depth=10,
                )
                model.fit(x.loc[obs_mask, predictors], x.loc[obs_mask, col])
                x.loc[miss_mask, col] = model.predict(x.loc[miss_mask, predictors])
    return x


def select_existing(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def evaluate_clustering(
    x: np.ndarray,
    method: str = "sustain",
    z_thresholds: Sequence[float] = (1.0, 2.0, 3.0),
) -> Tuple[int, float, float, float, np.ndarray]:
    """Evaluate clustering quality and select the best k.

    method="kmeans" uses KMeans on x.
    method="sustain" uses a SuStaIn-like event representation + GMM.
    """
    best = None
    best_labels = None
    n = x.shape[0]
    k_max = min(6, n - 1)

    if method == "sustain":
        means = x.mean(axis=0)
        stds = x.std(axis=0)
        stds[stds == 0] = 1.0
        x_z = (x - means) / stds
        eval_space = build_event_matrix(x_z, z_thresholds)
    else:
        eval_space = x

    for k in range(2, k_max + 1):
        try:
            if method == "sustain":
                model = GaussianMixture(
                    n_components=k,
                    random_state=42,
                    covariance_type="full",
                    n_init=10,
                )
                labels = model.fit_predict(eval_space)
            else:
                model = KMeans(n_clusters=k, random_state=42, n_init=20)
                labels = model.fit_predict(eval_space)

            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(eval_space, labels)
            cal = calinski_harabasz_score(eval_space, labels)
            dav = davies_bouldin_score(eval_space, labels)
        except Exception:
            continue

        if best is None or sil > best[1]:
            best = (k, sil, cal, dav)
            best_labels = labels

    if best is None:
        raise ValueError(f"Not enough samples for clustering with method={method!r}")
    return best[0], best[1], best[2], best[3], best_labels


def run_baseline_analysis(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    cluster_method: str = "sustain",
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    results: List[AnalysisResult] = []
    labels_by_set: Dict[str, pd.Series] = {}

    for name, features in feature_sets.items():
        features = [f for f in features if f in df.columns]
        if len(features) < 4:
            continue
        numeric_df = to_numeric_frame(df[["PATNO", *features]].copy(), features)
        keep = [
            c
            for c in features
            if numeric_df[c].notna().mean() >= 0.4 and numeric_df[c].std(skipna=True) > 0
        ]
        if len(keep) < 4:
            continue

        x_raw = numeric_df[keep]
        x_imp = missforest_impute(x_raw)
        scaler = StandardScaler() # permet de standardiser les données en les centrant sur la moyenne et en les réduisant à l'aide de l'écart type, ce qui est souvent recommandé avant d'appliquer des algorithmes de clustering pour éviter que les variables avec des échelles différentes ne dominent la formation des clusters.
        x_std = scaler.fit_transform(x_imp)

        if name.endswith("_pca"):
            pca = PCA(n_components=0.90, random_state=42)
            x_std = pca.fit_transform(x_std)

        used_method = cluster_method
        try:
            best_k, sil, cal, dav, labels = evaluate_clustering(x_std, method=cluster_method)
        except Exception:
            if cluster_method == "sustain":
                used_method = "kmeans"
                best_k, sil, cal, dav, labels = evaluate_clustering(x_std, method="kmeans")
            else:
                continue

        results.append(
            AnalysisResult(
                name=name,
                clustering_method=used_method,
                n_features=len(keep),
                n_subjects=len(numeric_df),
                best_k=best_k,
                silhouette=float(sil),
                calinski_harabasz=float(cal),
                davies_bouldin=float(dav),
            )
        )
        labels_by_set[name] = pd.Series(labels, index=numeric_df["PATNO"].values)

    if not results:
        out = pd.DataFrame(
            columns=[
                "name",
                "clustering_method",
                "n_features",
                "n_subjects",
                "best_k",
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
            ]
        )
        return out, labels_by_set

    out = pd.DataFrame([r.__dict__ for r in results]).sort_values("silhouette", ascending=False)
    return out, labels_by_set


def infer_biomarker_directions(df: pd.DataFrame, features: Sequence[str]) -> Dict[str, str]:
    """Infer increasing/decreasing direction using correlation with NP3TOT when possible."""
    directions: Dict[str, str] = {}
    ref = "NP3TOT" if "NP3TOT" in df.columns else features[0]
    ref_vals = pd.to_numeric(df[ref], errors="coerce")
    for feat in features:
        vals = pd.to_numeric(df[feat], errors="coerce")
        corr = vals.corr(ref_vals)
        directions[feat] = "increasing" if pd.isna(corr) or corr >= 0 else "decreasing"
    return directions


def build_event_matrix(X_z: np.ndarray, z_thresholds: Sequence[float]) -> np.ndarray:
    """Build binary event matrix from z-scores using SuStaIn-like thresholds."""
    n_samples, n_biomarkers = X_z.shape
    n_events = len(z_thresholds) * n_biomarkers
    events = np.zeros((n_samples, n_events), dtype=float)
    for j, zt in enumerate(z_thresholds):
        for i in range(n_biomarkers):
            idx = j * n_biomarkers + i
            events[:, idx] = (X_z[:, i] >= zt).astype(float)
    return events


def fit_zscore_sustain(
    X_train: np.ndarray,
    features: Sequence[str],
    directions: Dict[str, str],
    n_subtypes: int,
    z_thresholds: Sequence[float],
    random_state: int = 42,
    means: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Fit a z-score SuStaIn-like model (GMM on thresholded event matrix)."""
    if means is None:
        means = X_train.mean(axis=0)
    else:
        means = np.asarray(means, dtype=float)
        
    if stds is None:
        stds = X_train.std(axis=0)
    else:
        stds = np.asarray(stds, dtype=float)
        
    stds = stds.copy()
    stds[stds == 0] = 1.0

    X_z = np.zeros_like(X_train, dtype=float)
    for i, feat in enumerate(features):
        if directions.get(feat, "increasing") == "decreasing":
            X_z[:, i] = (means[i] - X_train[:, i]) / stds[i]
        else:
            X_z[:, i] = (X_train[:, i] - means[i]) / stds[i]

    events = build_event_matrix(X_z, z_thresholds)
    gmm = GaussianMixture(
        n_components=n_subtypes,
        random_state=random_state,
        covariance_type="full",
        n_init=10,
    )
    gmm.fit(events)
    log_likelihood = float(gmm.score(events) * len(events))
    return {
        "means": means,
        "stds": stds,
        "directions": directions,
        "z_thresholds": list(z_thresholds),
        "gmm": gmm,
        "bic": float(gmm.bic(events)),
        "aic": float(gmm.aic(events)),
        "log_likelihood": log_likelihood,
    }


def apply_zscore_sustain(model: Dict[str, object], X: np.ndarray, features: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply fitted z-score SuStaIn-like model to new data."""
    means = np.asarray(model["means"], dtype=float)
    stds = np.asarray(model["stds"], dtype=float)
    directions = model["directions"]
    z_thresholds = model["z_thresholds"]
    gmm: GaussianMixture = model["gmm"]

    X_z = np.zeros_like(X, dtype=float)
    for i, feat in enumerate(features):
        if directions.get(feat, "increasing") == "decreasing":
            X_z[:, i] = (means[i] - X[:, i]) / stds[i]
        else:
            X_z[:, i] = (X[:, i] - means[i]) / stds[i]

    events = build_event_matrix(X_z, z_thresholds)
    subtypes = gmm.predict(events)
    probs = gmm.predict_proba(events)
    stages = events.sum(axis=1).astype(int)
    return subtypes, stages, probs

# Permet de calculer des métriques de stabilité des sous-types selon les paires de visites V04, V06 et V08. Pour chaque paire de visites, la fonction extrait les sous-types assignés à chaque patient pour ces visites, fusionne les données pour ne conserver que les patients présents dans les deux visites, puis calcule l'Adjusted Rand Index (ARI) et le taux de stabilité (proportion de patients ayant le même sous-type dans les deux visites). Les résultats sont retournés dans un DataFrame avec une ligne par paire de visites, indiquant le nombre de patients communs, l'ARI et le taux de stabilité.
def stability_metrics(assignments: pd.DataFrame) -> pd.DataFrame:
    pairs = [("V04", "V06"), ("V06", "V08"), ("V04", "V08")]
    rows = []
    for a, b in pairs:
        da = assignments[assignments["VISIT"] == a][["PATNO", "Subtype"]].rename(columns={"Subtype": f"S_{a}"})
        db = assignments[assignments["VISIT"] == b][["PATNO", "Subtype"]].rename(columns={"Subtype": f"S_{b}"})
        common = da.merge(db, on="PATNO", how="inner")
        if len(common) < 5:
            continue
        ari = adjusted_rand_score(common[f"S_{a}"], common[f"S_{b}"]) # l'Adjusted Rand Index (ARI) est une mesure de similarité entre deux partitions (ou clusterings) qui prend en compte le hasard. Il varie entre -1 et 1, où 1 indique une correspondance parfaite entre les deux clusterings, 0 indique une correspondance due au hasard, et des valeurs négatives indiquent une correspondance pire que le hasard.
        stable_rate = float((common[f"S_{a}"] == common[f"S_{b}"]).mean())
        rows.append({"pair": f"{a}-{b}", "n_common": len(common), "ari": ari, "stable_rate": stable_rate})
    return pd.DataFrame(rows)


def longitudinal_imaging_characterization(assignments: pd.DataFrame, merged_long: pd.DataFrame, imaging_cols: Sequence[str]) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    use_img = [c for c in imaging_cols if c in merged_long.columns]
    if not use_img:
        return pd.DataFrame()
    joined = assignments.merge(merged_long[["PATNO", "VISIT", *use_img]], on=["PATNO", "VISIT"], how="left")
    out = joined.groupby(["VISIT", "Subtype"], as_index=False)[use_img].mean(numeric_only=True)
    return out

# Cette fonction permet de pivoter un DataFrame de format long (avec des colonnes "PATNO", "VISIT" et d'autres variables) en format large, où chaque variable est suffixée par la visite correspondante. Pour chaque visite unique dans la colonne "VISIT", la fonction sélectionne les colonnes associées à cette visite, les renomme pour inclure le suffixe de visite, puis concatène tous les sous-ensembles pour obtenir un DataFrame large avec une ligne par patient et des colonnes pour chaque variable-visite.
def pivot_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df_long.columns if c not in {"PATNO", "VISIT"}]
    parts: List[pd.DataFrame] = []
    for visit, block in df_long.groupby("VISIT"):
        tmp = block[["PATNO", *value_cols]].copy()
        tmp = tmp.rename(columns={c: f"{c}_{visit}" for c in value_cols})
        parts.append(tmp.set_index("PATNO"))
    wide = pd.concat(parts, axis=1)
    wide = wide.reset_index().sort_values("PATNO")
    return wide


def export_visit_workbook(df_long: pd.DataFrame, general: pd.DataFrame, out_path: Path) -> None:
    """Export an extended workbook with one sheet per visit, similar to Fichier_PrincipalV61.xlsx."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if general is not None and not general.empty:
            general_cols = [c for c in ["PATNO", "EDUCYRS", "PDDXDT", "BIRTHDT"] if c in general.columns]
            general_export = general[general_cols].drop_duplicates(subset=["PATNO"]).sort_values("PATNO")
            general_export.to_excel(writer, sheet_name="General", index=False)

        visit_order = sorted(df_long["VISIT"].dropna().unique().tolist(), key=_visit_sort_key)
        for visit in visit_order:
            sheet = str(visit)[:31]
            block = df_long[df_long["VISIT"] == visit].copy()
            if block.empty:
                continue
            block = block.drop(columns=["VISIT"], errors="ignore")
            priority_cols = [c for c in ["PATNO", "COHORT", "AGE_AT_VISIT", "EDUCYRS", "SEX", "PDDXDT", "BIRTHDT"] if c in block.columns]
            other_cols = [c for c in block.columns if c not in priority_cols]
            block = block[priority_cols + other_cols].sort_values("PATNO")
            block.to_excel(writer, sheet_name=sheet, index=False)


def main() -> None:
    root = Path(__file__).resolve().parent
    ressources = root.parent / "Ressources"
    sources_dir = ressources / "Fichiers sources"
    
    # 0) Prepare subdirectories for organized outputs
    sustain_out = root / "results" / "sustain"
    sustain_out.mkdir(parents=True, exist_ok=True)
    
    consolidated_dir = sustain_out / "consolidated"
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_dir = sustain_out / "baseline_clustering"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    longitudinal_dir = sustain_out / "longitudinal"
    longitudinal_dir.mkdir(parents=True, exist_ok=True)
    
    imaging_dir = sustain_out / "imaging"
    imaging_dir.mkdir(parents=True, exist_ok=True)

    excel_path = ressources / "Fichier_PrincipalV61.xlsx"
    wide_path = ressources / "result_4_wide_format.csv"

    # Fusion must be as complete as possible: use all visits available in both files.
    visits_for_fusion = discover_fusion_visits(wide_path, excel_path)
    if not visits_for_fusion:
        raise RuntimeError("No visit detected in source files for fusion.")
    print(f"Visits used for fusion ({len(visits_for_fusion)}): {visits_for_fusion}")

    # 1) Fusion: CSV wide → long  +  Excel all columns → long
    wide_long = load_wide_long(wide_path, visits_for_fusion)
    excel_all_long, per_visit_img_cols = load_excel_all_long(excel_path, visits_for_fusion)
    print(f"Imaging columns by visit from Excel: {per_visit_img_cols}")
    # Outer join: keeps ALL patients from both sources, including the ~60 patients
    # present only in the Excel file and BL rows available only in the Excel file.
    # Columns present in both DataFrames get an "_XL" suffix on the Excel copy.
    merged_long = wide_long.merge(
        excel_all_long,
        on=["PATNO", "VISIT"],
        how="outer",
        suffixes=("", "_XL"),
    )

    # For columns duplicated in both sources, prefer the CSV value and fill any
    # remaining NaN from the Excel counterpart (e.g. covers Excel-only patients).
    xl_overlap_cols = [c for c in merged_long.columns if c.endswith("_XL")]
    for xl_col in xl_overlap_cols:
        base_col = xl_col[:-3]  # strip "_XL"
        if base_col in merged_long.columns:
            merged_long[base_col] = merged_long[base_col].combine_first(merged_long[xl_col])
    merged_long = merged_long.drop(columns=xl_overlap_cols)

    # Merge patient-level demographics only available in the General sheet
    # (EDUCYRS, PDDXDT, BIRTHDT — absent from the per-visit CSV).
    general = load_excel_general(excel_path)
    if not general.empty:
        merged_long = merged_long.merge(general, on="PATNO", how="left")

    imaging_cols = sorted({c for cols in per_visit_img_cols.values() for c in cols if c in merged_long.columns})

    # 2) Integrate attached/new sources
    rbdsq_path = resolve_source_path(ressources, sources_dir, "REM_Sleep_Behavior_Disorder_Questionnaire_13Mar2026.csv")
    scopa_path = resolve_source_path(ressources, sources_dir, "SCOPA-AUT_13Mar2026.csv")
    epworth_path = resolve_source_path(ressources, sources_dir, "Epworth_Sleepiness_Scale_13Mar2026.csv")
    quip_path = resolve_source_path(
        ressources, sources_dir, "QUIP-Current-Short_13Mar2026.csv", required=False
    )
    medication_path = resolve_source_path(ressources, sources_dir, "LEDD_Concomitant_Medication_Log_13Mar2026.csv")
    vitals_path = resolve_source_path(ressources, sources_dir, "Vital_Signs_13Mar2026.csv")
    park_path = resolve_source_path(ressources, sources_dir, "Features_of_Parkinsonism_13Mar2026.csv")
    pd_dx_path = resolve_source_path(ressources, sources_dir, "PD_Diagnosis_History_13Mar2026.csv")

    rbdsq = aggregate_rbdsq(rbdsq_path)
    rbdsq_detail = aggregate_prefixed_source(
        rbdsq_path,
        prefix="RBDSQ",
        numeric_cols=[
            "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL", "DRMFIGHT",
            "DRMUMV", "DRMOBJFL", "MVAWAKEN", "SLPDSTRB", "STROKE", "HETRA", "PARKISM", "RLS",
            "NARCLPSY", "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH",
        ],
    )
    scopa = aggregate_scopa(scopa_path)
    scopa_detail = aggregate_prefixed_source(
        scopa_path,
        prefix="SCOPA",
        numeric_cols=[
            "SCAU1", "SCAU2", "SCAU3", "SCAU4", "SCAU5", "SCAU6", "SCAU7", "SCAU8", "SCAU9",
            "SCAU10", "SCAU11", "SCAU12", "SCAU13", "SCAU14", "SCAU15", "SCAU16", "SCAU17",
            "SCAU18", "SCAU19", "SCAU20", "SCAU21", "SCAU22", "SCAU23", "SCAU23A", "SCAU23AT",
            "SCAU24", "SCAU25", "SCAU26A", "SCAU26AT", "SCAU26B", "SCAU26BT", "SCAU26C",
            "SCAU26CT", "SCAU26D", "SCAU26DT",
        ],
    )
    epworth = aggregate_epworth(epworth_path)
    epworth_detail = aggregate_prefixed_source(
        epworth_path,
        prefix="ESS",
        numeric_cols=["ESS1", "ESS2", "ESS3", "ESS4", "ESS5", "ESS6", "ESS7", "ESS8"],
    )
    quip_available = quip_path is not None
    if quip_available:
        quip = aggregate_quip(quip_path)
        quip_detail = aggregate_prefixed_source(
            quip_path,
            prefix="QUIP",
            numeric_cols=[
                "TMGAMBLE", "TMSEX", "TMBUY", "TMEAT", "CNTRLGMB", "CNTRLSEX", "CNTRLBUY", "CNTRLEAT",
                "TMTORACT", "TMTMTACT", "TMTRWD", "TMDISMED", "CNTRLDSM",
            ],
        )
    else:
        print("WARNING: QUIP-Current-Short_13Mar2026.csv not found; fallback to NUPSOURC_score2 proxy.")
        quip = pd.DataFrame(columns=["PATNO", "VISIT"])
        quip_detail = pd.DataFrame(columns=["PATNO", "VISIT"])
    medication = aggregate_medication(medication_path)
    medication_detail = aggregate_prefixed_source(
        medication_path,
        prefix="MED",
        numeric_cols=["TOTDDA", "LEDDSTRMG", "LEDDOSE", "LEDDOSFRQ", "LEDD"],
        text_cols=["LEDTRT", "LEDDOSSTR", "STARTDT", "STOPDT"],
    )
    vitals = aggregate_vitals(vitals_path)
    vitals_detail = aggregate_prefixed_source(
        vitals_path,
        prefix="VITAL",
        numeric_cols=["WGTKG", "HTCM", "TEMPC", "SYSSUP", "DIASUP", "HRSUP", "SYSSTND", "DIASTND", "HRSTND"],
        text_cols=["BPARM"],
    )
    park_feats = aggregate_features_pd(park_path)
    park_detail = aggregate_prefixed_source(
        park_path,
        prefix="PARK",
        numeric_cols=["FEATBRADY", "FEATPOSINS", "FEATRIGID", "FEATTREMOR", "PSGLVL"],
    )
    diagnosis_detail = aggregate_prefixed_source(
        pd_dx_path,
        prefix="DX",
        numeric_cols=["DXTREMOR", "DXRIGID", "DXBRADY", "DXPOSINS", "DXOTHSX"],
        text_cols=["SXDT", "PDDXDT", "DOMSIDE"],
    )

    merged_long = merge_sources(
        merged_long,
        [
            rbdsq,
            rbdsq_detail,
            scopa,
            scopa_detail,
            quip,
            quip_detail,
            epworth,
            epworth_detail,
            medication,
            medication_detail,
            vitals,
            vitals_detail,
            park_feats,
            park_detail,
            diagnosis_detail,
        ],
    )

    # Some source merges can introduce duplicate column names; keep first instance.
    dup_cols = merged_long.columns[merged_long.columns.duplicated()].tolist()
    if dup_cols:
        print(f"WARNING: dropping {len(dup_cols)} duplicate columns after merge (e.g. {dup_cols[:5]}).")
        merged_long = merged_long.loc[:, ~merged_long.columns.duplicated()].copy()

    print("\nFinal merged_long shape:", merged_long.shape)
    # Prefer explicit QUIP-CS variables from source CSVs.
    if "QUIP_ANY_ICD" in merged_long.columns:
        merged_long["QUIP_SCORE"] = pd.to_numeric(merged_long["QUIP_ANY_ICD"], errors="coerce")
    elif "NUPSOURC_score2" in merged_long.columns:
        merged_long["QUIP_SCORE"] = pd.to_numeric(merged_long["NUPSOURC_score2"], errors="coerce")

    # Imaging eligibility requirement for V04/V06/V08
    imaging_elig = compute_imaging_eligibility(merged_long, imaging_cols)
    imaging_elig.to_csv(consolidated_dir / "extended_imaging_eligible_patients.csv", index=False)

    # 3) Consolidated datasets
    merged_long.to_csv(consolidated_dir / "extended_consolidated_long.csv", index=False)
    consolidated_wide = pivot_long_to_wide(merged_long)
    consolidated_wide.to_csv(consolidated_dir / "extended_consolidated_wide.csv", index=False)
    del consolidated_wide
    gc.collect()
    print("INFO: skipping extended_ppmi_visit_workbook.xlsx export to keep memory usage stable.")

    # Prepare baseline PD cohort (BL) with SC fallback for MOCA family.
    cohort_col = "COHORT"
    if cohort_col not in merged_long.columns:
        raise RuntimeError("COHORT variable not found after fusion.")
    visit_col = merged_long["VISIT"]
    if isinstance(visit_col, pd.DataFrame):
        visit_col = visit_col.iloc[:, 0]
    v04 = merged_long.loc[visit_col == "BL"].copy()
    sc = merged_long.loc[visit_col == "SC"].copy()

    moca_cols = [c for c in merged_long.columns if c == "MCATOT" or c.startswith("MCA")]
    if not sc.empty and moca_cols:
        sc_fallback = (
            sc[["PATNO", *moca_cols]]
            .sort_values("PATNO")
            .drop_duplicates(subset=["PATNO"], keep="first")
            .set_index("PATNO")
        )
        for col in moca_cols:
            if col in v04.columns and col in sc_fallback.columns:
                v04[col] = v04[col].combine_first(v04["PATNO"].map(sc_fallback[col]))

    v04_pd = v04[v04[cohort_col] == "Parkinson's Disease"].copy()
     # La v04 est la baseline , je n'ai pas changé le nom de la variable pour ne pas commettre d'erreur en oubliant un terme 
    # 4) Methodological analysis: feature-set variants on baseline clustering
    totals_set = [
        "NP1RTOT",
        "NP2PTOT",
        "NP3TOT",
        "MCATOT",
        "MSEADLG",
        "RBDSQ_TOTAL",
        "SCOPA_AUT_TOTAL",
        "QUIP_ANY_ICD",
    ]

    subscores_set = [
        c
        for c in v04_pd.columns
        if (
            (c.startswith("NP1") and c != "NP1RTOT")
            or (c.startswith("NP2") and c != "NP2PTOT")
            or (c.startswith("NP3") and c != "NP3TOT")
            or (c.startswith("MCA") and c != "MCATOT")
        )
    ]
    subscores_set += ["MSEADLG", "RBDSQ_TOTAL", "SCOPA_AUT_TOTAL", "QUIP_ANY_ICD", "QUIP_ICD_COUNT"]

    extra_set = [
        *imaging_cols,
        *[c for c in v04_pd.columns if c.startswith("MED_")],
        *[c for c in v04_pd.columns if c.startswith("VITAL_")],
        *[c for c in v04_pd.columns if c.startswith("PARK_")],
        "ESS_TOTAL",
    ]

    feature_sets = {
        "global_scores": select_existing(v04_pd, totals_set),
        "subscores": select_existing(v04_pd, subscores_set),
        "augmented_composites_imaging": select_existing(v04_pd, totals_set + extra_set),
        "augmented_composites_imaging_pca": select_existing(v04_pd, totals_set + extra_set),
    }
    print("\nFeature sets for baseline clustering:")
    for set_name, features in feature_sets.items():
        print(f"  {set_name}: {len(features)} features")
    clustering_results, labels_by_set = run_baseline_analysis(v04_pd, feature_sets)
    clustering_results.to_csv(baseline_dir / "extended_baseline_clustering_comparison.csv", index=False)

    # Baseline cluster label exports for quick downstream interpretation
    for scenario, labels in labels_by_set.items():
        pd.DataFrame({"PATNO": labels.index, "cluster": labels.values + 1}).to_csv(
            baseline_dir / f"extended_{scenario}_baseline_clusters.csv", index=False
        )

    # 5) Longitudinal z-score SuStaIn runs for methodological scenarios
    # Keep only PD for SuStaIn, and only imaging-eligible for imaging-focused run
    long_pd = merged_long[merged_long[cohort_col] == "Parkinson's Disease"].copy()
    eligible_patnos = set(imaging_elig.loc[imaging_elig["eligible_2plus"], "PATNO"])
    long_pd_img = long_pd[long_pd["PATNO"].isin(eligible_patnos)].copy()

    sustain_scenarios = {
        "global_scores": select_existing(long_pd, totals_set),
        "subscores": select_existing(long_pd, subscores_set),
        "augmented_with_imaging": select_existing(long_pd_img, totals_set + extra_set),
    }

    sustain_rows = []
    all_stability = []

    for scenario, feats in sustain_scenarios.items():
        work_df = long_pd_img if scenario == "augmented_with_imaging" else long_pd
        if len(feats) < 4:
            continue

        cols = ["PATNO", "VISIT", *feats]
        cols = [c for c in cols if c in work_df.columns]
        stacked = work_df[work_df["VISIT"].isin(VISITS_LONGITUDINAL)][cols].copy()
        if stacked.empty:
            continue

        for col in feats:
            if col in stacked.columns:
                stacked[col] = pd.to_numeric(stacked[col], errors="coerce")

        existing = [f for f in feats if f in stacked.columns]
        if len(existing) < 4:
            continue

        # Keep patients with V04 and at least one extra non-empty visit in V06/V08.
        v04_patno = set(stacked.loc[stacked["VISIT"] == "V04", "PATNO"].unique())
        availability = (
            stacked.groupby("PATNO")[existing]
            .apply(lambda x: x.notna().any(axis=1).sum())
            .rename("n_nonempty_visits")
            .reset_index()
        )
        keep_patno = set(availability.loc[availability["n_nonempty_visits"] >= 2, "PATNO"]) & v04_patno
        stacked = stacked[stacked["PATNO"].isin(keep_patno)].copy()

        numeric = to_numeric_frame(stacked[existing].copy(), existing)
        used_features = [
            f for f in existing if numeric[f].notna().mean() >= 0.35 and numeric[f].std(skipna=True) > 0
        ]
        if len(used_features) > 32: # limit to 32 features for SuStaIn to avoid overfitting and computational issues; keep those with highest variance
            variances = numeric[used_features].var(skipna=True).sort_values(ascending=False)
            used_features = variances.head(32).index.tolist()

        if len(used_features) < 4:
            continue

        x_imp = missforest_impute(to_numeric_frame(stacked[used_features].copy(), used_features))
        stacked.loc[:, used_features] = x_imp.values

        train = stacked[stacked["VISIT"] == "V04"].copy()
        n_pat = int(train["PATNO"].nunique())
        if n_pat < 30 or train.empty:
            continue

        X_train = train[used_features].to_numpy(dtype=float)
        directions = infer_biomarker_directions(train, used_features)
        
        # Compute BL means and stds for proper longitudinal z-score standardization
        bl_df = work_df[(work_df["VISIT"] == "BL") & (work_df["PATNO"].isin(keep_patno))].copy()
        bl_numeric = to_numeric_frame(bl_df[used_features].copy(), used_features)
        bl_means = bl_numeric.mean().values
        bl_stds = bl_numeric.std(ddof=0).values
        # Fallback onto V04 train set if perfectly missing / zero variance
        fallback_means = X_train.mean(axis=0)
        fallback_stds = X_train.std(axis=0)
        for i in range(len(used_features)):
            if pd.isna(bl_means[i]):
                bl_means[i] = fallback_means[i]
            if pd.isna(bl_stds[i]) or bl_stds[i] == 0:
                bl_stds[i] = fallback_stds[i]

        k_max = min(8, max(3, n_pat - 1))
        candidates: List[Tuple[int, Dict[str, object]]] = []
        for k in range(3, k_max + 1):
            model = fit_zscore_sustain(
                X_train=X_train,
                features=used_features,
                directions=directions,
                n_subtypes=k,
                z_thresholds=(1, 2, 3),
                random_state=42,
                means=bl_means,
                stds=bl_stds,
            )
            candidates.append((k, model))
        if not candidates:
            continue

        best_k, best_model = min(candidates, key=lambda t: t[1]["bic"])

        scenario_prefix = f"extended_{scenario}"

        X_all = stacked[used_features].to_numpy(dtype=float)
        subtype_idx, stage_idx, _ = apply_zscore_sustain(best_model, X_all, used_features)
        assignments = pd.DataFrame(
            {
                "PATNO": stacked["PATNO"].values,
                "VISIT": stacked["VISIT"].values,
                "Subtype": subtype_idx + 1,
                "Stage": stage_idx,
            }
        )
        assignments.to_csv(longitudinal_dir / f"extended_{scenario}_longitudinal_assignments.csv", index=False)

        stability = stability_metrics(assignments)
        if not stability.empty:
            stability.insert(0, "scenario", scenario)
            all_stability.append(stability)

        img_char = longitudinal_imaging_characterization(assignments, merged_long, imaging_cols)
        if not img_char.empty:
            img_char.insert(0, "scenario", scenario)
            img_char.to_csv(imaging_dir / f"extended_{scenario}_imaging_characterization.csv", index=False)

        sustain_rows.append(
            {
                "scenario": scenario,
                "n_patients": int(n_pat),
                "n_features_used": int(len(used_features)),
                "best_n_subtypes": int(best_k),
                "best_loglikelihood": float(best_model["log_likelihood"]),
                "best_bic": float(best_model["bic"]),
            }
        )
    print("\nSuStaIn z-score modeling results:")
    
    sustain_df = pd.DataFrame(sustain_rows)
    sustain_df.to_csv(longitudinal_dir / "extended_sustain_methodology_results.csv", index=False)

    if all_stability:
        pd.concat(all_stability, ignore_index=True).to_csv(
            longitudinal_dir / "extended_longitudinal_stability.csv", index=False
        )

    summary = {
        "QUIP variables are integrated from QUIP-Current-Short_13Mar2026.csv (with QUIP_ANY_ICD used in core feature sets)."
        if quip_available
        else "QUIP-Current-Short_13Mar2026.csv not found; QUIP_SCORE fallback uses NUPSOURC_score2 from result_4_wide_format.csv."
    }

    summary = {
        "files_generated": [
            "extended_consolidated_long.csv",
            "extended_consolidated_wide.csv",
            "extended_ppmi_visit_workbook.xlsx",
            "extended_imaging_eligible_patients.csv",
            "extended_baseline_clustering_comparison.csv",
            "extended_sustain_methodology_results.csv",
            "extended_longitudinal_stability.csv",
        ],
        
    }

    with (sustain_out / "extended_pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    quip_note_md = (
        "- Added QUIP-CS from `QUIP-Current-Short_13Mar2026.csv` (QUIP_ANY_ICD, QUIP_ICD_COUNT, per-domain flags)."
        if quip_available
        else "- QUIP CSV not found in sources; fallback proxy used: `NUPSOURC_score2` from `result_4_wide_format.csv`."
    )

    md_lines = [
        "# Extended PPMI Pipeline Summary",
        "",
        "## Outputs",
        "- `consolidated/extended_consolidated_long.csv`",
        "- `consolidated/extended_consolidated_wide.csv`",
        "- `baseline_clustering/extended_baseline_clustering_comparison.csv`",
        "- `longitudinal/extended_sustain_methodology_results.csv`",
        "- `longitudinal/extended_longitudinal_stability.csv`",
    ]
    (sustain_out / "extended_pipeline_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

# %%
