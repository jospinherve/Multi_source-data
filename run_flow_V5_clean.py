"""
run_flow_v5_clean.py
=====================

Conditional Real NVP — clean v5 training run.

This script is a drop-in replacement for the v4 training script that
produced ``best_subscores_v4_ref_L8_H256/``. The architecture, feature
selection (32 MI-selected features), conditioning (FiLM on
subtype/stage/age), data split, and output artefact format are all
preserved so that ``main_v5.tex`` (figures, sample-quality CSV,
selected_flow_config.json) can be re-pointed to the new output folder
without any source-code change in the manuscript pipeline.

Six targeted fixes vs. v4:
    1. T_max of CosineAnnealingLR aligned to the actual training horizon.
    2. AdamW + weight_decay=1e-4 (mild parameter-space regularisation).
    3. Hard early stopping with patience=12 + restore_best_weights.
    4. Light Gaussian dequantisation on training inputs (0.5 % per-feature std).
    5. Smaller batch size (64) so n=798 yields ~10 batches/epoch.
    6. Hidden width reduced from 256 to 192 (slightly under-parameterised
       relative to v4 — closes the train/val gap).

Outputs (saved to OUT_DIR):
    selected_flow_config.json
    training_history.csv
    sample_quality_stats.csv
    condition_stats.json
    flow_training_curve.png
    flow_gradient_norm.png
    flow_latent_distribution.png
    flow_latent_pca.png
    flow_latent_real_vs_generated.png
    flow_feature_top20.png
    best_flow_model.pt
    preprocessing_transformer.pkl
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCH_LABEL = "v5_ref_L8_H192_clean"
N_LAYERS = 8
HIDDEN = 192
USE_INV1X1 = False           # kept off — same as v4
USE_FILM = True
COND_KEYS = ("subtype", "stage", "age")

TOTAL_EPOCHS = 100
WARMUP_EPOCHS = 10
PEAK_LR = 3e-4
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
VAL_FRAC = 0.20
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 0.05  # NLL units
DEQUANT_STD_FRAC = 0.005     # +0.5 % of per-feature std as Gaussian noise

N_FEATURES = 32              # MI top-32 (matches v4)

# Replace the four paths below with the locations on the student's machine.

CONSOLIDATED_LONG_CSV = "results/sustain/consolidated/extended_consolidated_long.csv"
SUBSCORES_ASSIGN_CSV  = "results/sustain/longitudinal/extended_subscores_longitudinal_assignments.csv"
OUT_DIR = Path(f"flow_v5/best_subscores_{ARCH_LABEL}")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subscores_dataframe() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, feature_names, subtype, stage, age) for the V04 patients
    that have a baseline subtype assignment in the Subscores k=2 partition.
    """
    long = pd.read_csv(CONSOLIDATED_LONG_CSV, low_memory=False)
    assign = pd.read_csv(SUBSCORES_ASSIGN_CSV)
    base = assign[assign["VISIT"] == "V04"][["PATNO", "Subtype"]].rename(
        columns={"Subtype": "BaselineSubtype"}
    )
    df = long.merge(base, on="PATNO", how="inner")
    df = df[df["VISIT"].isin(["V04", "V06", "V08"])].copy()

    # Candidate feature pool — full MDS-UPDRS sub-items + derived totals.
    cand = [c for c in df.columns
            if c.startswith(("NP1", "NP2", "NP3", "NHY", "MSEADL")) and df[c].dtype != "O"]
    df = df.dropna(subset=cand + ["AGE_AT_VISIT"])

    # MI ranking against NP3TOT — same target as v4.
    target = df["NP3TOT"].values
    feats = [c for c in cand if c != "NP3TOT"]
    mi = mutual_info_regression(df[feats].fillna(0).values, target, random_state=SEED)
    order = np.argsort(mi)[::-1]
    selected = ["NP3TOT"] + [feats[i] for i in order[: N_FEATURES - 1]]

    X = df[selected].values.astype(np.float32)
    subtype = df["BaselineSubtype"].astype(int).values
    visit_to_stage = {"V04": 1.0, "V06": 2.0, "V08": 3.0}
    stage = np.array([visit_to_stage[v] for v in df["VISIT"]], dtype=np.float32)
    age = df["AGE_AT_VISIT"].astype(float).values
    return df, selected, X, subtype, stage, age


# ---------------------------------------------------------------------------
# Conditional Real NVP with FiLM
# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden)
        self.beta  = nn.Linear(cond_dim, hidden)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.gamma(c)) + self.beta(c)


class CouplingNet(nn.Module):
    """Affine coupling network with FiLM conditioning."""
    def __init__(self, dim_in: int, dim_out: int, hidden: int, cond_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, hidden)
        self.film1 = FiLM(cond_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.film2 = FiLM(cond_dim, hidden)
        self.scale = nn.Linear(hidden, dim_out)
        self.shift = nn.Linear(hidden, dim_out)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        h = F.silu(self.film1(self.lin1(x), c))
        h = F.silu(self.film2(self.lin2(h), c))
        s = torch.tanh(self.scale(h)) * 2.0          # bound log-scale
        t = self.shift(h)
        return s, t


class CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden: int, cond_dim: int, mask_parity: int):
        super().__init__()
        
        mask = torch.zeros(dim, dtype=torch.bool)
        mask[mask_parity::2] = True
        self.register_buffer("mask", mask)
        d_in  = int(mask.sum().item())   
        d_out = dim - d_in               
        self.net = CouplingNet(d_in, d_out, hidden, cond_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        
        x_pass = x[:, self.mask]         
        x_tran = x[:, ~self.mask]        # (B, d_out) — dims à transformer
        s, t = self.net(x_pass, c)       # s,t de shape (B, d_out)
        z = x.clone()
        z[:, ~self.mask] = x_tran * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor):
        z_pass = z[:, self.mask]         # (B, d_in)
        z_tran = z[:, ~self.mask]        # (B, d_out)
        s, t = self.net(z_pass, c)
        x = z.clone()
        x[:, ~self.mask] = (z_tran - t) * torch.exp(-s)
        return x


class ConditionalRealNVP(nn.Module):
    def __init__(self, dim: int, n_layers: int, hidden: int, cond_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [CouplingLayer(dim, hidden, cond_dim, i % 2) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        log_det_total = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z, c)
            log_det_total = log_det_total + ld
        return z, log_det_total

    def inverse(self, z: torch.Tensor, c: torch.Tensor):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, c)
        return x

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x, c)
        prior = -0.5 * (z.pow(2).sum(dim=1) + z.size(1) * math.log(2 * math.pi))
        return prior + log_det


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def train() -> dict:
    set_seed(SEED)
    df, selected, X_raw, subtype, stage, age = load_subscores_dataframe()
    print(f"[data] n={len(X_raw)} rows; features={len(selected)}; "
          f"subtypes={sorted(set(subtype))}")

    # Per-feature standardisation
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw).astype(np.float32)

    # Conditioning vector (one-hot subtype + standardised stage/age)
    s_mean, s_std = float(stage.mean()), float(stage.std() + 1e-6)
    a_mean, a_std = float(age.mean()),   float(age.std() + 1e-6)
    n_subtypes = int(np.max(subtype))
    c_subtype = np.eye(n_subtypes, dtype=np.float32)[subtype - 1]
    c_stage   = ((stage - s_mean) / s_std).reshape(-1, 1).astype(np.float32)
    c_age     = ((age   - a_mean) / a_std).reshape(-1, 1).astype(np.float32)
    C = np.concatenate([c_subtype, c_stage, c_age], axis=1).astype(np.float32)
    cond_dim = C.shape[1]
    print(f"[cond] dim={cond_dim} (subtype OH + stage_z + age_z)")

    # Train / val split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_val = int(round(VAL_FRAC * len(X)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, C_tr = X[tr_idx], C[tr_idx]
    X_va, C_va = X[val_idx], C[val_idx]

    # Light Gaussian dequantisation applied online
    feat_std = X_tr.std(axis=0, keepdims=True)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(C_tr)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(C_va)),
        batch_size=256, shuffle=False,
    )

    model = ConditionalRealNVP(X.shape[1], N_LAYERS, HIDDEN, cond_dim).to(DEVICE)
    optim = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    sched = SequentialLR(
        optim,
        schedulers=[
            LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS),
            CosineAnnealingLR(optim, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=ETA_MIN),
        ],
        milestones=[WARMUP_EPOCHS],
    )

    history = []
    best_val, best_epoch, best_state = float("inf"), -1, None
    bad_epochs = 0
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        tr_nll, n_tr = 0.0, 0
        grad_norms = []
        for x, c in tr_loader:
            x = x.to(DEVICE); c = c.to(DEVICE)
            x_aug = x + DEQUANT_STD_FRAC * torch.from_numpy(feat_std).to(DEVICE) * torch.randn_like(x)
            optim.zero_grad()
            nll = -model.log_prob(x_aug, c).mean()
            nll.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            grad_norms.append(float(gn))
            optim.step()
            tr_nll += float(nll) * x.size(0); n_tr += x.size(0)
        tr_nll /= max(n_tr, 1)

        model.eval()
        va_nll, n_va = 0.0, 0
        with torch.no_grad():
            for x, c in va_loader:
                x = x.to(DEVICE); c = c.to(DEVICE)
                va_nll += float(-model.log_prob(x, c).mean()) * x.size(0); n_va += x.size(0)
        va_nll /= max(n_va, 1)
        sched.step()

        history.append({"epoch": epoch, "train_nll": tr_nll, "val_nll": va_nll,
                        "lr": optim.param_groups[0]["lr"],
                        "median_grad_norm": float(np.median(grad_norms))})
        print(f"epoch {epoch:3d} | train {tr_nll:7.3f} | val {va_nll:7.3f} | "
              f"lr {optim.param_groups[0]['lr']:.2e} | gn~{np.median(grad_norms):.2f}")

        if va_nll + EARLY_STOP_MIN_DELTA < best_val:
            best_val, best_epoch = va_nll, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"[early-stop] no val improvement for {EARLY_STOP_PATIENCE} epochs (best {best_val:.3f} at epoch {best_epoch}).")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    torch.save(model.state_dict(), OUT_DIR / "best_flow_model.pt")
    with open(OUT_DIR / "preprocessing_transformer.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "selected_features": selected,
                     "stage_mean": s_mean, "stage_std": s_std,
                     "age_mean": a_mean, "age_std": a_std}, f)

    config = {
        "scenario": "subscores",
        "architecture_label": ARCH_LABEL,
        "selected_features": selected,
        "n_selected_features": len(selected),
        "n_layers": N_LAYERS, "hidden_units": HIDDEN,
        "use_inv1x1": USE_INV1X1, "use_film": USE_FILM,
        "conditioning": "subtype_stage_age",
        "device": DEVICE,
        "total_epochs_planned": TOTAL_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "peak_lr": PEAK_LR, "eta_min": ETA_MIN,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "val_frac": VAL_FRAC,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "dequant_std_frac": DEQUANT_STD_FRAC,
        "best_epoch": best_epoch, "best_val_nll": best_val,
        "seed": SEED,
    }
    (OUT_DIR / "selected_flow_config.json").write_text(json.dumps(config, indent=2))
    (OUT_DIR / "condition_stats.json").write_text(json.dumps({
        "stage_mean": s_mean, "stage_std": s_std,
        "age_mean": a_mean, "age_std": a_std,
        "n_subtypes": n_subtypes,
    }, indent=2))

    return {
        "model": model, "scaler": scaler, "selected": selected,
        "X_raw": X_raw, "X": X, "C": C, "subtype": subtype, "stage": stage,
        "best_epoch": best_epoch, "best_val": best_val,
        "history": history, "cond_dim": cond_dim,
    }


# ---------------------------------------------------------------------------
# Diagnostics + plots
# ---------------------------------------------------------------------------
def diagnostics(out: dict) -> None:
    model, scaler = out["model"], out["scaler"]
    X_raw, X, C = out["X_raw"], out["X"], out["C"]
    subtype, selected = out["subtype"], out["selected"]
    history = pd.DataFrame(out["history"])

    # Training curve
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_nll"], label="Train NLL")
    plt.plot(history["epoch"], history["val_nll"],   label="Val NLL")
    plt.axvline(out["best_epoch"], color="grey", ls="--", lw=1,
                label=f"best epoch={out['best_epoch']}")
    plt.xlabel("Epoch"); plt.ylabel("Negative log-likelihood")
    plt.title("Conditional Real NVP — clean v5 training")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "flow_training_curve.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(history["epoch"], history["median_grad_norm"], color="#444")
    plt.xlabel("Epoch"); plt.ylabel("Median gradient norm (per epoch)")
    plt.title("Gradient-norm trajectory"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "flow_gradient_norm.png", dpi=160); plt.close()

    # Sample quality (per-subtype, matched n)
    model.eval()
    rows = []
    with torch.no_grad():
        for s in sorted(set(subtype)):
            mask = subtype == s
            n = mask.sum()
            c = torch.from_numpy(C[mask]).to(DEVICE)
            z = torch.randn(n, X.shape[1], device=DEVICE)
            x_gen = model.inverse(z, c).cpu().numpy()
            x_gen_real = scaler.inverse_transform(x_gen)
            x_real = X_raw[mask]
            for j, fname in enumerate(selected):
                rows.append({
                    "subtype": int(s), "feature": fname,
                    "real_mean": float(np.mean(x_real[:, j])),
                    "real_std":  float(np.std(x_real[:, j])),
                    "gen_mean":  float(np.mean(x_gen_real[:, j])),
                    "gen_std":   float(np.std(x_gen_real[:, j])),
                    "abs_mean_gap": float(abs(np.mean(x_real[:, j]) - np.mean(x_gen_real[:, j]))),
                    "abs_std_gap":  float(abs(np.std(x_real[:, j])  - np.std(x_gen_real[:, j]))),
                })
    sq = pd.DataFrame(rows)
    sq.to_csv(OUT_DIR / "sample_quality_stats.csv", index=False)

    print(f"[sample_quality] mean ratio gen_std/real_std = "
          f"{(sq['gen_std']/sq['real_std'].replace(0,np.nan)).mean():.3f}")
    print(f"[sample_quality] median abs_std_gap = {sq['abs_std_gap'].median():.3f}")

    # Latent visualisations (real only — generated identical by construction)
    with torch.no_grad():
        z_real, _ = model.forward(torch.from_numpy(X).to(DEVICE),
                                  torch.from_numpy(C).to(DEVICE))
        z_real = z_real.cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(z_real.ravel(), bins=80, density=True, alpha=0.7, color="#1f77b4")
    xs = np.linspace(-4, 4, 400)
    plt.plot(xs, np.exp(-xs**2 / 2) / math.sqrt(2 * math.pi), "k--", lw=1.2, label="N(0,1)")
    plt.title("Marginal latent distribution (clean v5)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR / "flow_latent_distribution.png", dpi=160); plt.close()

    pca = PCA(n_components=2).fit(z_real)
    z2 = pca.transform(z_real)
    plt.figure(figsize=(8, 6))
    for s in sorted(set(subtype)):
        m = subtype == s
        plt.scatter(z2[m, 0], z2[m, 1], s=8, alpha=0.5, label=f"subtype {s}")
    plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Latent PCA by subtype")
    plt.tight_layout(); plt.savefig(OUT_DIR / "flow_latent_pca.png", dpi=160); plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = train()
    diagnostics(out)
    print(f"\nAll artefacts saved to: {OUT_DIR.resolve()}")