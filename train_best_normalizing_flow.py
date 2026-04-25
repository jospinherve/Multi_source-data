"""Train the best conditional normalizing flow for the PPMI subscores scenario.

This script consumes the already computed SuStaIn outputs, selects the best
feature scenario (subscores), and trains a GPU-friendly conditional Real NVP
with ActNorm, invertible linear mixing, and FiLM-conditioned coupling layers.

.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler


LOGGER = logging.getLogger("train_best_normalizing_flow")

SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent

METADATA_COLUMNS = {
    "PATNO",
    "VISIT",
    "Subtype",
    "Stage",
    "AGE_AT_VISIT",
}

GLOBAL_SCORE_CANDIDATES = [
    "NP2PTOT",
    "NP3TOT",
    "NP1RTOT",
    "MCATOT",
    "MSEADLG",
    "SCOPA_AUT_TOTAL",
    "ESS_TOTAL",
    "RBDSQ_TOTAL",
]

SUBSCORE_PREFIXES = (
    "NP",
    "MCA",
    "MSE",
    "GDS",
    "SCOPA",
    "RBDSQ",
    "ESS",
    "QUIP",
    "MED",
    "NHY",
    "COG",
    "FEAT",
    "VITAL",
    "DVT",
    "DVSD",
    "DYS",
    "HRPOSTMED",
    "FINAL_SEX_ENCODED",
)


@dataclass(frozen=True)
class BestFlowConfig:
    scenario: str = "subscores"
    architecture_label: str = "v4_ref_L8_H256"
    n_layers: int = 8
    hidden_units: int = 256
    use_inv1x1: bool = False
    use_film: bool = True
    conditioning: str = "subtype_stage_age"
    target_features: int = 32
    target_subtypes: int = 2
    cond_embedding_dim: int = 16
    cond_hidden_dim: int = 64
    preprocessing: str = "standard"
    seed: int = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConditionEncoder(nn.Module):
    def __init__(self, n_subtypes: int, embedding_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.subtype_embedding = nn.Embedding(n_subtypes, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond: torch.Tensor | None) -> torch.Tensor | None:
        if cond is None:
            return None
        subtype_idx = cond[:, 0].long() # permet de 
        numeric = cond[:, 1:3]
        subtype_emb = self.subtype_embedding(subtype_idx)
        return self.mlp(torch.cat([subtype_emb, numeric], dim=-1))


class ScaleTranslationNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.LeakyReLU(0.2)
        self.out = nn.Linear(hidden_dim, output_dim * 2)
        self.out.weight.data.zero_()
        self.out.bias.data.zero_()
        if cond_dim > 0:
            self.film1 = nn.Linear(cond_dim, hidden_dim * 2)
            self.film2 = nn.Linear(cond_dim, hidden_dim * 2)
            self.film3 = nn.Linear(cond_dim, hidden_dim * 2)

    def _film(self, h: torch.Tensor, layer: nn.Linear, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None or self.cond_dim == 0:
            return h
        gamma, beta = layer(cond).chunk(2, dim=-1)
        return gamma * h + beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.layer1(x))
        h = self._film(h, self.film1, cond) if self.cond_dim > 0 else h
        h = self.act(self.layer2(h))
        h = self._film(h, self.film2, cond) if self.cond_dim > 0 else h
        h = self.act(self.layer3(h))
        h = self._film(h, self.film3, cond) if self.cond_dim > 0 else h
        st = self.out(h)
        s, t = st.chunk(2, dim=-1)
        
        # Stabilité Numérique Critique : Clamper ou utiliser Tanh pour éviter la projection infinie lors de la génération
        # Un s grand donnerait np.exp(s) immense (crash de la mémoire).
        # Tanh projette entre -1 et 1, que l'on multiplie par 2 pour permettre exp(-2) à expr(2).
        s = torch.tanh(s) * 2.0
        
        return s, t


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, cond_dim: int, mask_type: str = "even") -> None:
        super().__init__()
        if mask_type == "even":
            mask = torch.tensor([i % 2 == 0 for i in range(dim)], dtype=torch.bool)
        else:
            mask = torch.tensor([i % 2 == 1 for i in range(dim)], dtype=torch.bool)
        self.register_buffer("mask", mask)
        dim_x1 = int(mask.sum().item())
        dim_x2 = dim - dim_x1
        self.st_net = ScaleTranslationNet(dim_x1, dim_x2, hidden_dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(x.device)
        x1 = x[:, mask]
        x2 = x[:, ~mask]
        s, t = self.st_net(x1, cond)
        y2 = x2 * torch.exp(s) + t
        y = torch.zeros_like(x)
        y[:, mask] = x1
        y[:, ~mask] = y2
        return y, s.sum(dim=-1)

    def inverse(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        mask = self.mask.to(z.device)
        z1 = z[:, mask]
        z2 = z[:, ~mask]
        s, t = self.st_net(z1, cond)
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.zeros_like(z)
        x[:, mask] = z1
        x[:, ~mask] = x2
        return x


class ActNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    def _initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0) + 1e-6
            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)
            self.initialized.fill_(True)

    def _clamped_log_scale(self) -> torch.Tensor:
        # UN SEUL endroit pour le clamping — forward ET inverse l'utilisent
        return torch.clamp(self.log_scale, min=-3.0, max=3.0)

    def forward(self, x: torch.Tensor, cond=None):
        if not bool(self.initialized):
            self._initialize(x)
        ls = self._clamped_log_scale()
        y = (x + self.bias) * torch.exp(ls)
        return y, ls.sum().expand(x.shape[0])

    def inverse(self, z: torch.Tensor, cond=None) -> torch.Tensor:
        ls = self._clamped_log_scale()  # même clamping !
        return z * torch.exp(-ls) - self.bias
# Cette classe implémente une transformation linéaire inversible avec une factorisation LU pour assurer l'inversibilité et la stabilité numérique. Elle est utilisée comme couche de mélange dans le flux normalisant conditionnel.
class InvertibleLinearLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        w_init = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(w_init)
        p, l, u = torch.linalg.lu(q)
        s = torch.diag(u)
        sign_s = torch.sign(s)
        sign_s[sign_s == 0] = 1.0
        self.register_buffer("p", p)
        self.register_buffer("sign_s", sign_s)
        self.l = nn.Parameter(torch.tril(l, diagonal=-1))
        self.u = nn.Parameter(torch.triu(u, diagonal=1))
        self.log_s = nn.Parameter(torch.log(torch.abs(s) + 1e-6))
        self.register_buffer("eye", torch.eye(dim))

    
    def _weight(self) -> torch.Tensor:
        safe_log_s = torch.clamp(self.log_s, min=-2.0, max=2.0)  # exp(2)≈7 max
        lower = torch.tril(self.l, diagonal=-1)
        lower = lower / (torch.norm(lower, dim=1, keepdim=True).clamp(min=1.0))  # normaliser lignes
        L = lower + self.eye
        U = torch.triu(self.u, diagonal=1)
        U = U / (torch.norm(U, dim=0, keepdim=True).clamp(min=1.0))  # normaliser colonnes
        S = torch.diag(self.sign_s * torch.exp(safe_log_s))
        W = self.p @ L @ (U + S)
        
        # Contrôle de norme spectrale : forcer sigma_max <= 1 + epsilon
        sigma_max = torch.linalg.matrix_norm(W, ord=2).detach()
        if sigma_max > 2.0:
            W = W / (sigma_max / 2.0)
        return W

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self._weight()
        y = x @ weight
        safe_log_s = torch.clamp(self.log_s, min=-2.0, max=2.0)
        log_det = safe_log_s.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        weight = self._weight()
        eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        for eps in (0.0, 1e-6, 1e-5, 1e-4):
            try:
                stabilized = weight + eps * eye
                result = torch.linalg.solve(stabilized.T, z.T).T
                if torch.isfinite(result).all():
                    return result
            except RuntimeError:
                continue
        raise RuntimeError("InvertibleLinearLU.inverse failed to produce finite values")


class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        dim: int,
        n_subtypes: int,
        n_layers: int,
        hidden_dim: int,
        use_inv1x1: bool,
        use_film: bool,
        cond_embedding_dim: int,
        cond_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_subtypes = n_subtypes
        self.cond_dim = cond_hidden_dim
        self.condition_encoder = ConditionEncoder(
            n_subtypes=n_subtypes,
            embedding_dim=cond_embedding_dim,
            hidden_dim=cond_hidden_dim,
            out_dim=cond_hidden_dim,
        )
        self.layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            self.layers.append(ActNorm(dim))
            if use_inv1x1:
                self.layers.append(InvertibleLinearLU(dim))
            mask_type = "even" if layer_idx % 2 == 0 else "odd"
            self.layers.append(AffineCouplingLayer(dim, hidden_dim, self.cond_dim if use_film else 0, mask_type=mask_type))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        cond_encoded = self.condition_encoder(cond) if cond is not None else None
        for layer in self.layers:
            if isinstance(layer, (ActNorm, InvertibleLinearLU)):
                z, layer_log_det = layer(z)
            else:
                z, layer_log_det = layer(z, cond_encoded)
            log_det = log_det + layer_log_det
        return z, log_det

    def inverse(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = z
        cond_encoded = self.condition_encoder(cond) if cond is not None else None
        for layer in reversed(self.layers):
            if isinstance(layer, (ActNorm, InvertibleLinearLU)):
                x = layer.inverse(x)
            else:
                x = layer.inverse(x, cond_encoded)
        return x

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        z, log_det = self.forward(x, cond)
        base = -0.5 * (z**2 + math.log(2 * math.pi))
        return base.sum(dim=-1) + log_det

    def sample(self, n_samples: int, cond: torch.Tensor | None = None, device: torch.device | None = None) -> torch.Tensor:

        if device is None:
            device = next(self.parameters()).device
        attempts = ((1.0, 4.0), (0.5, 3.0), (0.25, 2.0))
        for z_scale, max_norm in attempts:
            z = torch.randn(n_samples, self.dim, device=device) * z_scale
            z_norm = torch.norm(z, dim=1, keepdim=True)
            z = torch.where(z_norm > max_norm, z * (max_norm / (z_norm + 1e-7)), z)
            try:
                x = self.inverse(z, cond)
            except RuntimeError:
                continue

            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = torch.clamp(x, -1e6, 1e6)
            if torch.isfinite(x).all():
                return x

        import warnings
        warnings.warn("Sampling failed to produce finite values after retries. Returning zero samples.")
        return torch.zeros(n_samples, self.dim, device=device)


def load_config(path: Path) -> BestFlowConfig:
    if not path.exists():
        LOGGER.warning("Config file not found at %s. Falling back to built-in defaults.", path)
        return BestFlowConfig()

    try:
        import yaml  # pyright: ignore[reportMissingModuleSource]

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        best_flow = data.get("best_flow", {}) if isinstance(data, dict) else {}
        return BestFlowConfig(
            scenario=best_flow.get("scenario", "subscores"),
            architecture_label=best_flow.get("architecture_label", "v3_ref_L16_H1300"),
            n_layers=int(best_flow.get("n_layers", 12)),
            hidden_units=int(best_flow.get("hidden_units", 768)),
            use_inv1x1=bool(best_flow.get("use_inv1x1", False)),
            use_film=bool(best_flow.get("use_film", True)),
            conditioning=best_flow.get("conditioning", "subtype_stage_age"),
            target_features=int(best_flow.get("target_features", 32)),
            target_subtypes=int(best_flow.get("target_subtypes", 2)),
            cond_embedding_dim=int(best_flow.get("cond_embedding_dim", 16)),
            cond_hidden_dim=int(best_flow.get("cond_hidden_dim", 64)),
            preprocessing=str(best_flow.get("preprocessing", "standard")),
            seed=int(best_flow.get("seed", 42)),
        )
    except Exception as exc:
        LOGGER.warning(
            "Failed to parse config file at %s (%s). Falling back to built-in defaults.",
            path,
            exc,
        )
        return BestFlowConfig()


def build_candidate_columns(columns: Iterable[str]) -> list[str]:
    candidates: list[str] = []
    for col in columns:
        if col in METADATA_COLUMNS:
            continue
        if any(col.startswith(prefix) for prefix in SUBSCORE_PREFIXES):
            candidates.append(col)
    for col in GLOBAL_SCORE_CANDIDATES:
        if col in columns and col not in candidates:
            candidates.append(col)
    return candidates


def score_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    completeness_map = (numeric.notna().sum(axis=0) / max(1, numeric.shape[0])).to_dict()
    medians = numeric.median(axis=0)
    x_filled = numeric.fillna(medians)
    non_constant = [col for col in feature_cols if x_filled[col].nunique() > 1]
    if not non_constant:
        return pd.DataFrame(columns=["feature", "completeness", "mi_subtype", "mi_stage", "score"])

    x = x_filled[non_constant].to_numpy(dtype=np.float32)
    y_subtype = pd.to_numeric(df["Subtype"], errors="coerce").fillna(1).astype(int).to_numpy()
    y_subtype = np.clip(y_subtype - 1, 0, max(0, int(np.nanmax(y_subtype) - 1)))
    y_stage = pd.to_numeric(df["Stage"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    mi_subtype = mutual_info_classif(x, y_subtype, random_state=42)
    mi_stage = mutual_info_regression(x, y_stage, random_state=42)

    rows = []
    for idx, col in enumerate(non_constant):
        completeness = float(completeness_map.get(col, 0.0))
        score = 0.60 * float(mi_subtype[idx]) + 0.30 * float(mi_stage[idx]) + 0.10 * completeness
        rows.append(
            {
                "feature": col,
                "completeness": completeness,
                "mi_subtype": float(mi_subtype[idx]),
                "mi_stage": float(mi_stage[idx]),
                "score": float(score),
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "completeness", "mi_subtype"], ascending=[False, False, False])


def prepare_dataset(data_root: Path, scenario: str, target_features: int) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    long_path = data_root / "results" / "sustain" / "consolidated" / "extended_consolidated_long.csv"
    assign_path = data_root / "results" / "sustain" / "longitudinal" / f"extended_{scenario}_longitudinal_assignments.csv"

    header_cols = pd.read_csv(long_path, nrows=0).columns.tolist()
    candidate_columns = build_candidate_columns(header_cols)
    usecols = [col for col in (list(METADATA_COLUMNS) + candidate_columns) if col in header_cols]

    base_df = pd.read_csv(long_path, usecols=usecols)
    assign_df = pd.read_csv(assign_path)
    merged = base_df.merge(assign_df, on=["PATNO", "VISIT"], how="inner", validate="many_to_one")

    candidate_columns = [col for col in candidate_columns if col in merged.columns]
    if not candidate_columns:
        raise ValueError("No candidate feature columns found in merged dataset")

    scored = score_features(merged, candidate_columns)
    if scored.empty:
        raise ValueError("Feature scoring failed: no usable columns")

    selected = scored.head(target_features)["feature"].tolist()
    selected_df = merged[["PATNO", "VISIT", "Subtype", "Stage", "AGE_AT_VISIT"] + selected].copy()
    return selected_df, selected, scored

def remap_sentinel_values(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Remplace les sentinelles codage (101, 999, etc.) par NaN pour éviter la gonflement du StandardScaler.
    
    Les sentinelles sont généralement des codes manquants dans les échelles PPMI.
    Les remapper à NaN permet à fillna/median de les gérer correctement.
    """
    SENTINELS = (101, 999, 888, 777)
    output = frame.copy()
    for col in feature_cols:
        series = pd.to_numeric(output[col], errors="coerce")
        if series.dropna().empty:
            continue
        # Remplacer les sentinelles par NaN
        for sentinel in SENTINELS:
            series = series.mask(np.isclose(series, sentinel, atol=0.5), np.nan)
        output[col] = series
    return output


# cette fonction ajoute un petit bruit gaussien aux caractéristiques sélectionnées pour éviter les problèmes de quantification et améliorer la stabilité du flux normalisant lors de l'entraînement. Elle vérifie d'abord si les valeurs sont essentiellement entières, et si c'est le cas, elle ajoute un bruit normal de petite amplitude. Sinon, elle laisse les valeurs telles quelles.
def add_dequantization_jitter(frame: pd.DataFrame, feature_cols: list[str], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = remap_sentinel_values(frame, feature_cols)  # 🔧 Remapper sentinelles AVANT jitter
    for col in feature_cols:
        series = pd.to_numeric(output[col], errors="coerce")
        if series.dropna().empty:
            continue
        filled = series.fillna(series.median())
        if np.allclose(filled.dropna().values, np.round(filled.dropna().values)):
            output[col] = filled + rng.normal(loc=0.0, scale=0.01, size=len(filled))
        else:
            output[col] = filled
    return output


def build_clinical_bounds(df_real: pd.DataFrame, selected_features: list[str]) -> dict[str, tuple[float, float]]:
    """Construit des bounds cliniques réalistes basés sur les données réelles.
    
    Stratégie: 
    - Pour chaque feature, calculer (min, max) valides (excluant NaN et extrema proches des sentinelles)
    - Élargir les bounds de ±2.5% pour laisser un peu de marge aux samples générés
    """
    bounds = {}
    for col in selected_features:
        series = pd.to_numeric(df_real[col], errors="coerce")
        valid = series.dropna()
        if len(valid) == 0:
            bounds[col] = (-np.inf, np.inf)
            continue
        # Exclure les valeurs qui pourraient être des sentinelles mal-remappées
        q1, q99 = valid.quantile([0.01, 0.99])
        min_val = float(valid[valid >= q1].min())
        max_val = float(valid[valid <= q99].max())
        # Élargir de ±2.5% pour robustesse
        margin = (max_val - min_val) * 0.025
        bounds[col] = (min_val - margin, max_val + margin)
    return bounds


def apply_clinical_bounds(generated: np.ndarray, selected_features: list[str], bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Projette les samples générés dans une zone clinique plausible.

    On évite un clipping dur systématique, car il peut écraser toute la variance
    quand le modèle sort trop loin de la distribution réelle. La projection
    souple garde l'ordre relatif des samples tout en restant bornée.
    """
    projected = generated.copy()
    for idx, col in enumerate(selected_features):
        if col not in bounds:
            continue
        min_val, max_val = bounds[col]
        if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
            continue

        center = 0.5 * (min_val + max_val)
        half_range = max(1e-6, 0.5 * (max_val - min_val))
        normalized = (projected[:, idx] - center) / half_range
        projected[:, idx] = center + half_range * np.tanh(normalized)
        projected[:, idx] = np.clip(projected[:, idx], min_val, max_val)

    return projected



@dataclass(frozen=True)
class ConditionStats:
    """Global normalization stats computed once on the full dataset."""
    stage_mean: float
    stage_std: float
    age_mean: float
    age_std: float


def compute_condition_stats(df: pd.DataFrame) -> ConditionStats:
    stage = pd.to_numeric(df["Stage"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    age = pd.to_numeric(df["AGE_AT_VISIT"], errors="coerce")
    age = age.fillna(float(age.median())).to_numpy(dtype=np.float32)
    return ConditionStats(
        stage_mean=float(stage.mean()),
        stage_std=float(stage.std() + 1e-6),
        age_mean=float(age.mean()),
        age_std=float(age.std() + 1e-6),
    )


def build_condition_matrix(
    df: pd.DataFrame,
    n_subtypes: int,
    stats: ConditionStats | None = None,  # None = compute locally (legacy, à éviter)
) -> np.ndarray:
    subtype = df["Subtype"].astype(int).to_numpy() - 1
    subtype = np.clip(subtype, 0, n_subtypes - 1)
    stage = pd.to_numeric(df["Stage"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    age = pd.to_numeric(df["AGE_AT_VISIT"], errors="coerce")
    age = age.fillna(float(df["AGE_AT_VISIT"].median())).to_numpy(dtype=np.float32)

    if stats is not None:
        stage_norm = (stage - stats.stage_mean) / stats.stage_std
        age_norm = (age - stats.age_mean) / stats.age_std
    else:
        # Legacy path — local normalization, à ne plus utiliser en génération
        stage_norm = (stage - stage.mean()) / (stage.std() + 1e-6)
        age_norm = (age - age.mean()) / (age.std() + 1e-6)

    return np.column_stack(
        [subtype.astype(np.float32), stage_norm, age_norm]
    ).astype(np.float32)
def train_flow(
    features: np.ndarray,
    conditions: np.ndarray,
    config: BestFlowConfig,
    device: torch.device,
    output_dir: Path,
) -> tuple[ConditionalRealNVP, dict[str, list[float]], object]:
    x_train, x_val, c_train, c_val = train_test_split(
        features,
        conditions,
        test_size=0.2,
        random_state=42,
        stratify=conditions[:, 0].astype(int),
    )

    if config.preprocessing == "quantile":
        n_quantiles = min(1000, max(10, x_train.shape[0]))
        preprocessor = QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles, random_state=config.seed)
    else:
        preprocessor = StandardScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_val = preprocessor.transform(x_val)

    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError("Preprocessed features contain non-finite values")

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    c_train_t = torch.tensor(c_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    c_val_t = torch.tensor(c_val, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(x_train_t, c_train_t)
    val_ds = torch.utils.data.TensorDataset(x_val_t, c_val_t)
    batch_size = 256 if len(train_ds) >= 512 else 64
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = ConditionalRealNVP(
        dim=features.shape[1],
        n_subtypes=config.target_subtypes,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_units,
        use_inv1x1=config.use_inv1x1,
        use_film=config.use_film,
        cond_embedding_dim=config.cond_embedding_dim,
        cond_hidden_dim=config.cond_hidden_dim,
    ).to(device)

    base_lr = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    max_epochs = 200
    warmup_epochs = 10
    min_lr_ratio = 1e-6 / base_lr

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            warmup_progress = (current_epoch + 1) / max(1, warmup_epochs)
            return 0.1 + 0.9 * warmup_progress
        cosine_total = max(1, max_epochs - warmup_epochs)
        cosine_epoch = min(current_epoch - warmup_epochs, cosine_total)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_epoch / cosine_total))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val = float("inf")
    best_state = None
    patience = 55
    epochs_no_improve = 0
    history = {"epoch": [], "train_nll": [], "val_nll": [], "lr": [], "grad_norm": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        grad_norms = []
        for batch_idx, (x_batch, c_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_prob = model.log_prob(x_batch, c_batch)
            if not torch.isfinite(log_prob).all():
                raise RuntimeError(f"Non-finite log_prob detected at epoch={epoch}, batch={batch_idx}")
            loss = -log_prob.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch={epoch}, batch={batch_idx}")
            loss.backward()
            total_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu().item())
            if not np.isfinite(total_grad_norm):
                raise RuntimeError(f"Non-finite gradient norm detected at epoch={epoch}, batch={batch_idx}")
            grad_norms.append(total_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, c_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                log_prob = model.log_prob(x_batch, c_batch)
                if not torch.isfinite(log_prob).all():
                    raise RuntimeError(f"Non-finite validation log_prob detected at epoch={epoch}")
                val_loss = -log_prob.mean()
                if not torch.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite validation loss detected at epoch={epoch}")
                val_losses.append(float(val_loss.cpu().item()))

        train_nll = float(np.mean(train_losses)) if train_losses else float("inf")
        val_nll = float(np.mean(val_losses)) if val_losses else float("inf")
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["epoch"].append(epoch)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["lr"].append(current_lr)
        history["grad_norm"].append(float(np.mean(grad_norms)) if grad_norms else float("nan"))

        LOGGER.info("Epoch %03d | train_nll=%.4f | val_nll=%.4f | grad_norm=%.4f | lr=%.2e", epoch, train_nll, val_nll, history["grad_norm"][-1], current_lr)

        scheduler.step()

        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, preprocessor

def generate_samples(
    model: ConditionalRealNVP,
    selected_df: pd.DataFrame,
    selected_features: list[str],
    preprocessor: object,
    config: BestFlowConfig,
    cond_stats: ConditionStats,
    device: torch.device,
    n_samples_per_subtype: int = 500,
) -> pd.DataFrame | None:
    clinical_bounds = build_clinical_bounds(selected_df, selected_features)
    samples = []
    model.eval()
    rng = np.random.default_rng(config.seed)

    with torch.no_grad():
        for subtype_idx in range(config.target_subtypes):
            subtype_label = subtype_idx + 1
            subtype_df = selected_df[
                selected_df["Subtype"].astype(int) == subtype_label
            ]
            if subtype_df.empty:
                LOGGER.warning("No real data for subtype %d, skipping.", subtype_label)
                continue

            # Utiliser les stats GLOBALES — cohérent avec l'entraînement
            stage_sub = pd.to_numeric(
                subtype_df["Stage"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32)
            age_sub = pd.to_numeric(subtype_df["AGE_AT_VISIT"], errors="coerce")
            age_sub = age_sub.fillna(float(age_sub.median())).to_numpy(dtype=np.float32)

            idx = rng.choice(len(subtype_df), size=n_samples_per_subtype, replace=True)

            stage_norm = (stage_sub[idx] - cond_stats.stage_mean) / cond_stats.stage_std
            age_norm = (age_sub[idx] - cond_stats.age_mean) / cond_stats.age_std

            cond = np.column_stack([
                np.full(n_samples_per_subtype, float(subtype_idx)),
                stage_norm,
                age_norm,
            ]).astype(np.float32)
            cond_t = torch.tensor(cond, dtype=torch.float32, device=device)

            generated = model.sample(
                n_samples_per_subtype, cond=cond_t, device=device
            ).cpu().numpy()

            # Vérifier qualité avant inverse_transform
            finite_ratio = float(np.isfinite(generated).all(axis=1).mean())
            if finite_ratio < 0.9:
                LOGGER.error(
                    "Subtype %d: only %.1f%% fully-finite samples — model unstable.",
                    subtype_label, finite_ratio * 100,
                )
                continue

            generated = np.nan_to_num(generated, nan=0.0, posinf=10.0, neginf=-10.0)

            try:
                generated_orig = preprocessor.inverse_transform(generated)
            except Exception as e:
                LOGGER.error(
                    "inverse_transform failed for subtype %d: %s", subtype_label, e
                )
                continue

            violation_rate = _compute_violation_rate(
                generated_orig, selected_features, clinical_bounds
            )
            if violation_rate > 0.05:
                LOGGER.warning(
                    "Subtype %d: %.1f%% of values violate clinical bounds before projection.",
                    subtype_label, violation_rate * 100,
                )

            generated_orig = apply_clinical_bounds(
                generated_orig, selected_features, clinical_bounds
            )

            frame = pd.DataFrame(generated_orig, columns=selected_features)
            frame.insert(0, "Subtype", subtype_label)
            samples.append(frame)
            LOGGER.info(
                "Subtype %d: generated %d samples OK (violation_rate=%.1f%%).",
                subtype_label, n_samples_per_subtype, violation_rate * 100,
            )

    return pd.concat(samples, ignore_index=True) if samples else None


def _compute_violation_rate(
    generated: np.ndarray,
    selected_features: list[str],
    bounds: dict[str, tuple[float, float]],
) -> float:
    violations, total = 0, 0
    for idx, col in enumerate(selected_features):
        if col not in bounds:
            continue
        lo, hi = bounds[col]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        col_vals = generated[:, idx]
        violations += int(np.sum((col_vals < lo) | (col_vals > hi)))
        total += len(col_vals)
    return violations / max(1, total)


def check_roundtrip(
    model: ConditionalRealNVP,
    features: np.ndarray,
    conditions: np.ndarray,
    preprocessor: object,
    device: torch.device,
    n: int = 64,
) -> float:
    """Vérifie que inverse(forward(x)) ≈ x. Retourne l'erreur max."""
    model.eval()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(features), size=min(n, len(features)), replace=False)
    x_sample = preprocessor.transform(features[idx])
    x_t = torch.tensor(x_sample, dtype=torch.float32, device=device)
    c_t = torch.tensor(conditions[idx], dtype=torch.float32, device=device)

    with torch.no_grad():
        z, _ = model.forward(x_t, c_t)
        x_recon = model.inverse(z, c_t)

    err = (x_t - x_recon).abs().max().item()
    LOGGER.info("Roundtrip reconstruction error (max abs): %.6f", err)
    if err > 0.01:
        LOGGER.warning(
            "Roundtrip error=%.4f > 0.01 — inversion is numerically unstable. "
            "Generated samples will be unreliable.",
            err,
        )
    return err
def save_diagnostics(
    output_dir: Path,
    history: dict[str, list[float]],
    selected_features: list[str],
    scored_features: pd.DataFrame,
    selected_df: pd.DataFrame,
    model: ConditionalRealNVP,
    preprocessor: object,
    config: BestFlowConfig,
    device: torch.device,
    cond_stats: ConditionStats,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    scored_features.to_csv(output_dir / "feature_scores.csv", index=False)

    feature_payload = {
        "scenario": config.scenario,
        "architecture_label": config.architecture_label,
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
        "n_layers": config.n_layers,
        "hidden_units": config.hidden_units,
        "use_inv1x1": config.use_inv1x1,
        "use_film": config.use_film,
        "conditioning": config.conditioning,
        "device": str(device),
    }
    (output_dir / "selected_flow_config.json").write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")

    with open(output_dir / "preprocessing_transformer.pkl", "wb") as handle:
        pickle.dump(preprocessor, handle)

    torch.save(model.state_dict(), output_dir / "best_flow_model.pt")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_df["epoch"], history_df["train_nll"], label="Train NLL", linewidth=2)
    ax.plot(history_df["epoch"], history_df["val_nll"], label="Val NLL", linewidth=2)
    ax.set_title("Conditional Real NVP Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close(fig)

    if "grad_norm" in history_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history_df["epoch"], history_df["grad_norm"], color="#b83280", linewidth=2)
        ax.set_title("Gradient norm per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient norm")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "gradient_norm_curve.png", dpi=160)
        plt.close(fig)

    top_features = scored_features.head(20).sort_values("score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.28)))
    ax.barh(top_features["feature"], top_features["score"], color="#2b6cb0")
    ax.set_title("Top candidate features selected for the subscores flow")
    ax.set_xlabel("Selection score")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_selection_top20.png", dpi=160)
    plt.close(fig)

    real_x = selected_df[selected_features].to_numpy(dtype=np.float32)
    real_x_proc = preprocessor.transform(real_x)
    cond_matrix = build_condition_matrix(selected_df, config.target_subtypes, stats=cond_stats)
    with torch.no_grad():
        z_real, _ = model.forward(
            torch.tensor(real_x_proc, dtype=torch.float32, device=device),
            torch.tensor(cond_matrix, dtype=torch.float32, device=device),
        )
    z_real = z_real.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        if i >= z_real.shape[1]:
            ax.axis("off")
            continue
        ax.hist(z_real[:, i], bins=40, color="#2f855a", alpha=0.85)
        ax.set_title(f"Latent z[{i}] distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_distribution_overview.png", dpi=160)
    plt.close(fig)

    z_2d = PCA(n_components=2, random_state=config.seed).fit_transform(z_real)
    subtype_labels = cond_matrix[:, 0].astype(int) + 1
    fig, ax = plt.subplots(figsize=(8, 6))
    for subtype in sorted(np.unique(subtype_labels)):
        mask = subtype_labels == subtype
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=14, alpha=0.7, label=f"Subtype {subtype}")
    ax.set_title("Latent space separation by subtype (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_subtype_separation_pca.png", dpi=160)
    plt.close(fig)

    # Compare real latent codes against latent codes obtained from generated samples.
    n_compare = min(len(cond_matrix), 2000)
    rng = np.random.default_rng(config.seed)
    compare_idx = rng.choice(len(cond_matrix), size=n_compare, replace=False)
    cond_compare = cond_matrix[compare_idx]
    z_real_compare = z_real[compare_idx]

    with torch.no_grad():
        cond_compare_t = torch.tensor(cond_compare, dtype=torch.float32, device=device)
        x_generated_proc = model.sample(n_compare, cond=cond_compare_t, device=device)
        x_generated_proc = torch.nan_to_num(x_generated_proc, nan=0.0, posinf=1e6, neginf=-1e6)
        x_generated_proc = torch.clamp(x_generated_proc, -1e6, 1e6)
        z_generated_compare, _ = model.forward(x_generated_proc, cond_compare_t)

    z_generated_compare = z_generated_compare.cpu().numpy()
    z_combined = np.vstack([z_real_compare, z_generated_compare])
    z_combined_2d = PCA(n_components=2, random_state=config.seed).fit_transform(z_combined)
    z_real_2d = z_combined_2d[:n_compare]
    z_gen_2d = z_combined_2d[n_compare:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(z_real_2d[:, 0], z_real_2d[:, 1], s=14, alpha=0.50, label="z reel", color="#2b6cb0")
    ax.scatter(z_gen_2d[:, 0], z_gen_2d[:, 1], s=14, alpha=0.50, label="z genere", color="#c53030")
    ax.set_title("Comparaison des latents: z reel vs z genere (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_real_vs_generated_pca.png", dpi=160)
    plt.close(fig)

    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the best conditional normalizing flow for PPMI subscores")
    parser.add_argument("--config", default="/home_nfs/jospin/TFE/Article_Code/config.yaml", help="Project config file")
    parser.add_argument("--data-root", default=SCRIPT_ROOT, help="Article_Code root directory")
    parser.add_argument("--scenario", default=None, choices=["global_scores", "subscores"], help="Override the flow scenario (if None, uses config.yaml value)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--preprocessing", choices=["standard", "quantile"], default="standard", help="Feature preprocessing mode before flow training")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    config_path = Path(args.config).expanduser().resolve()
    LOGGER.info("Loading config from: %s", config_path)
    config = load_config(config_path)

    scenario = args.scenario if args.scenario is not None else config.scenario
    config = BestFlowConfig(
        scenario=scenario,
        architecture_label=config.architecture_label,
        n_layers=config.n_layers,
        hidden_units=config.hidden_units,
        use_inv1x1=config.use_inv1x1,
        use_film=config.use_film,
        conditioning=config.conditioning,
        target_features=config.target_features,
        target_subtypes=config.target_subtypes,
        cond_embedding_dim=config.cond_embedding_dim,
        cond_hidden_dim=config.cond_hidden_dim,
        preprocessing=args.preprocessing,
        seed=args.seed,
    )

    set_global_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_df, selected_features, scored_features = prepare_dataset(
        data_root, config.scenario, config.target_features
    )
    actual_subtypes = int(selected_df["Subtype"].max())
    config = replace(config, target_subtypes=actual_subtypes)

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Scenario: %s | Subtypes: %d", config.scenario, config.target_subtypes)
    LOGGER.info("Architecture: %s | layers=%d | hidden=%d", config.architecture_label, config.n_layers, config.hidden_units)

    # Calculer les stats globales UNE FOIS — utilisées partout
    cond_stats = compute_condition_stats(selected_df)
    LOGGER.info(
        "Condition stats — stage: mean=%.3f std=%.3f | age: mean=%.3f std=%.3f",
        cond_stats.stage_mean, cond_stats.stage_std,
        cond_stats.age_mean, cond_stats.age_std,
    )

    selected_df = add_dequantization_jitter(selected_df, selected_features)

    # Toutes les conditions utilisent les stats globales dès maintenant
    conditions = build_condition_matrix(selected_df, config.target_subtypes, stats=cond_stats)
    feature_values = selected_df[selected_features].to_numpy(dtype=np.float32)

    output_dir = data_root / "results" / "flow_diagnostics" / f"best_{config.scenario}_{config.architecture_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_dir / "selected_training_table.csv", index=False)

    
    (output_dir / "condition_stats.json").write_text(
        json.dumps({
            "stage_mean": cond_stats.stage_mean,
            "stage_std": cond_stats.stage_std,
            "age_mean": cond_stats.age_mean,
            "age_std": cond_stats.age_std,
        }, indent=2),
        encoding="utf-8",
    )

    model, history, preprocessor = train_flow(
        feature_values, conditions, config, device, output_dir
    )

    # Vérification de stabilité AVANT toute génération
    roundtrip_err = check_roundtrip(
        model, feature_values, conditions, preprocessor, device
    )
    if roundtrip_err > 0.1:
        LOGGER.error(
            "Roundtrip error=%.4f is too high. Generated samples will be unreliable. "
            "Consider reducing n_layers or disabling use_inv1x1.",
            roundtrip_err,
        )

    # Diagnostics visuels (courbes, PCA latente, feature scores)
    save_diagnostics(
        output_dir, history, selected_features, scored_features,
        selected_df, model, preprocessor, config, device, cond_stats,
    )

    # Génération finale — nouvelle fonction, stats globales, 500 samples
    generated_df = generate_samples(
        model, selected_df, selected_features, preprocessor,
        config, cond_stats, device, n_samples_per_subtype=500,
    )
    if generated_df is not None:
        generated_df.to_csv(output_dir / "generated_samples_by_subtype.csv", index=False)
        LOGGER.info("Generated %d samples saved.", len(generated_df))

        # Statistiques comparatives réel vs généré
        real_x = feature_values
        real_stats_df = pd.DataFrame({
            "feature": selected_features,
            "real_mean": real_x.mean(axis=0),
            "real_std": real_x.std(axis=0),
            "gen_mean": generated_df[selected_features].to_numpy(dtype=np.float32).mean(axis=0),
            "gen_std": generated_df[selected_features].to_numpy(dtype=np.float32).std(axis=0),
        })
        real_stats_df["abs_mean_gap"] = (real_stats_df["real_mean"] - real_stats_df["gen_mean"]).abs()
        real_stats_df["abs_std_gap"] = (real_stats_df["real_std"] - real_stats_df["gen_std"]).abs()
        real_stats_df.to_csv(output_dir / "sample_quality_stats.csv", index=False)
    else:
        LOGGER.error("Generation failed entirely. Check roundtrip_err and model stability.")

    LOGGER.info("Training completed. Outputs in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())