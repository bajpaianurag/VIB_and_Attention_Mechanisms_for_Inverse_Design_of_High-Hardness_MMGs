# Import Libraries
import os, json, csv, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import scipy.stats as st
from scipy.stats import norm
from collections import Counter
import optuna
from optuna.samplers import TPESampler

# Fixed seed (run for 10 seeds and average)
seed = 1
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

# Data Ingestion
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "H_v_dataset.csv"
data = pd.read_csv(data_path)
required_cols = {"Load", "HV"}
missing = required_cols - set(data.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Define column names (load + composition)
load_col = "Load" 
EXCLUDE = {"HV", load_col}
numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
composition_cols = [c for c in numeric_cols if c not in EXCLUDE]

if len(composition_cols) == 0:
    raise ValueError(
        "Could not infer composition_cols. Please set composition_cols explicitly "
        "to the list of elemental composition columns in H_v_dataset.csv."
    )

if load_col not in data.columns:
    raise ValueError(f"'{load_col}' not found in CSV columns. Available: {list(data.columns)}")

X_comp_tmp = data[composition_cols].to_numpy(dtype=np.float64)
if np.all(np.sum(X_comp_tmp, axis=1) <= 0):
    raise ValueError("All composition rows sum to <= 0. Check composition_cols selection.")

min_val = float(np.min(X_comp_tmp))
if min_val < -1e-8:
    raise ValueError(f"Composition matrix has negative entries (min={min_val}). Check input data.")
elif min_val < 0:
    print(f"[Warn] Small negative values in compositions (min={min_val}); likely rounding. Consider clipping to 0.")

print(f"[Columns] load_col='{load_col}' | num composition cols={len(composition_cols)}")
print(f"[Columns] composition_cols={composition_cols}")


# ==========================
# Data preparation
# ==========================
X_comp_all = data[composition_cols].to_numpy(dtype=np.float64)
row_sum = X_comp_all.sum(axis=1, keepdims=True)
row_sum[row_sum <= 0] = 1.0
X_comp_all = X_comp_all / row_sum

X_load_all_raw = data[[load_col]].to_numpy(dtype=np.float64)
y_all = data["HV"].to_numpy(dtype=np.float64)

EPS_NONZERO = 1e-12
nonzero_mask = X_comp_all > EPS_NONZERO

elem_names = np.array(composition_cols, dtype=object)
family_keys = []
for i in range(nonzero_mask.shape[0]):
    family_keys.append(tuple(elem_names[nonzero_mask[i]].tolist()))
family_keys = np.array(family_keys, dtype=object)

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
cluster_labels = kmeans.fit_predict(X_comp_all)

combined_labels = np.array(
    [f"{cluster_labels[i]}|{family_keys[i]}" for i in range(len(cluster_labels))],
    dtype=object
)

min_count = 5
combo_counts = Counter(combined_labels.tolist())
combined_labels = np.array(
    [lab if combo_counts[lab] >= min_count else "RARE" for lab in combined_labels],
    dtype=object
)
_, combined_codes = np.unique(combined_labels, return_inverse=True)

# Train/Test split (held out test)
TEST_FRAC = 0.20
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
(train_idx, test_idx), = strat_split.split(X_comp_all, combined_codes)
train_idx = np.asarray(train_idx)
test_idx  = np.asarray(test_idx)

# Calibration split carved from TRAIN
CAL_FRAC = 0.20
sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=seed)
train_fit_rel, cal_rel = next(sss_cal.split(X_comp_all[train_idx], combined_codes[train_idx]))

train_fit_idx = train_idx[np.asarray(train_fit_rel)]
cal_idx       = train_idx[np.asarray(cal_rel)]

X_comp_train_fit = X_comp_all[train_fit_idx]
X_load_train_fit_raw = X_load_all_raw[train_fit_idx]
y_train_fit = y_all[train_fit_idx]

X_comp_cal = X_comp_all[cal_idx]
X_load_cal_raw = X_load_all_raw[cal_idx]
y_cal = y_all[cal_idx]

X_comp_test = X_comp_all[test_idx]
X_load_test_raw = X_load_all_raw[test_idx]
y_test = y_all[test_idx]

scaler_load = StandardScaler().fit(X_load_train_fit_raw)
X_load_train_fit = scaler_load.transform(X_load_train_fit_raw)
X_load_cal       = scaler_load.transform(X_load_cal_raw)
X_load_test      = scaler_load.transform(X_load_test_raw)
X_train_fit_gp = np.hstack([X_comp_train_fit, X_load_train_fit]).astype(np.float64)
X_cal_gp       = np.hstack([X_comp_cal,       X_load_cal]).astype(np.float64)
X_test_gp      = np.hstack([X_comp_test,      X_load_test]).astype(np.float64)

print(f"[Split] Train_fit: {len(train_fit_idx)} | Cal: {len(cal_idx)} | Test: {len(test_idx)} | d_comp={X_comp_all.shape[1]}")


# =========================================================
# GP surrogate + Bayesian hyperparameter optimization
# =========================================================
def fit_gp_fixed(kernel_amp, length_scale, noise_level, X_train, y_train,
                 alpha_jitter=1e-10, nu=2.5):
    kernel = (
        C(float(kernel_amp), constant_value_bounds="fixed")
        * Matern(length_scale=float(length_scale), length_scale_bounds="fixed", nu=nu)
        + WhiteKernel(noise_level=float(noise_level), noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(alpha_jitter),
        normalize_y=True,
        optimizer=None,
        random_state=seed,
    )
    gp.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64).reshape(-1))
    return gp

def nlpd(y_true, mu, std):
    std = np.maximum(std, 1e-9)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * std**2) + ((y_true - mu)**2) / (std**2)))

def objective(trial: optuna.Trial) -> float:
    kernel_amp   = trial.suggest_float("kernel_amp", 1e-2, 1e3, log=True)
    length_scale = trial.suggest_float("length_scale", 1e-2, 1e2, log=True)
    noise_level  = trial.suggest_float("noise_level", 1e-8, 1e1, log=True)
    alpha_jitter = trial.suggest_float("alpha_jitter", 1e-12, 1e-6, log=True)

    gp = fit_gp_fixed(
        kernel_amp, length_scale, noise_level,
        X_train=X_train_fit_gp, y_train=y_train_fit,
        alpha_jitter=alpha_jitter, nu=2.5
    )
    mu_cal, std_cal = gp.predict(X_cal_gp, return_std=True)
    return nlpd(y_cal, mu_cal, std_cal)
    
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
study.optimize(objective, n_trials=5000, show_progress_bar=True)

best = study.best_params
best_gp_params = dict(best)
print("[GP baseline] Best calibration NLPD:", study.best_value)
print("[GP baseline] Best params:", best)

# Final GP refit on train_fit with best hyperparameters
gp_model = fit_gp_fixed(
    kernel_amp=best["kernel_amp"],
    length_scale=best["length_scale"],
    noise_level=best["noise_level"],
    X_train=X_train_fit_gp,
    y_train=y_train_fit,
    alpha_jitter=best["alpha_jitter"],
    nu=2.5
)

print("[GP baseline] Final fitted kernel:", gp_model.kernel_)


# Fit the final GP 
gp_model = fit_gp_fixed(
    kernel_amp=best["kernel_amp"],
    length_scale=best["length_scale"],
    noise_level=best["noise_level"],
    X_train=X_train_fit_gp,
    y_train=y_train_fit,
    alpha_jitter=best["alpha_jitter"],
    nu=2.5,
)

gp_report = {
    "kernel_form": "C * Matern(nu=2.5) + WhiteKernel",
    "kernel_amp": float(best["kernel_amp"]),
    "length_scale": float(best["length_scale"]),
    "noise_level": float(best["noise_level"]),
    "alpha_jitter": float(best["alpha_jitter"]),
    "normalize_y": True,
    "optimizer": None,
    "calibration_objective": "NLPD on calibration split",
    "best_calibration_nlpd": float(study.best_value),
    "fitted_kernel_string": str(gp_model.kernel_),
    "n_train_fit": int(X_train_fit_gp.shape[0]),
    "d_in": int(X_train_fit_gp.shape[1]),
    "seed": int(seed),
}

with open("gpbo_gp_hyperparams.json", "w") as f:
    json.dump(gp_report, f, indent=2)

print("[GP baseline] Saved GP hyperparameters to gpbo_gp_hyperparams.json")
print("[GP baseline] Fitted kernel:", gp_model.kernel_)


# =================================================================================
# Evaluate GP predictive performance + uncertainty diagnostics + visualizations
# =================================================================================
def compute_metrics(y_true, mu):
    mae = mean_absolute_error(y_true, mu)
    rmse = np.sqrt(mean_squared_error(y_true, mu))
    r2 = r2_score(y_true, mu)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def interval_coverage(y_true, mu, std, z):
    std = np.maximum(std, 1e-12)
    lo = mu - z * std
    hi = mu + z * std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def nlpd_score(y_true, mu, std):
    std = np.maximum(std, 1e-9)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * std**2) + ((y_true - mu)**2) / (std**2)))

# Predictions (CAL + TEST)
mu_cal, std_cal = gp_model.predict(X_cal_gp, return_std=True)
mu_test, std_test = gp_model.predict(X_test_gp, return_std=True)

std_cal = np.maximum(std_cal, 1e-12)
std_test = np.maximum(std_test, 1e-12)

# Point metrics
metrics_cal = compute_metrics(y_cal, mu_cal)
metrics_test = compute_metrics(y_test, mu_test)

# Uncertainty diagnostics
z_list = [0.5,0.75,1.0,1.25,1.5,1.645,1.8,1.96,2.2,2.5]
coverage_cal = {f"z={z:g}": interval_coverage(y_cal, mu_cal, std_cal, z) for z in z_list}
coverage_test = {f"z={z:g}": interval_coverage(y_test, mu_test, std_test, z) for z in z_list}

# Proper scoring rule (NLPD)
nlpd_cal = nlpd_score(y_cal, mu_cal, std_cal)
nlpd_test = nlpd_score(y_test, mu_test, std_test)

gp_metrics = {
    "cal": {**metrics_cal, "NLPD": float(nlpd_cal), "coverage": coverage_cal, "n": int(len(y_cal))},
    "test": {**metrics_test, "NLPD": float(nlpd_test), "coverage": coverage_test, "n": int(len(y_test))},
    "z_values": z_list,
    "note": "Coverage computed for Gaussian predictive intervals mu ± z*std using GP posterior std."
}

with open("gp_metrics.json", "w") as f:
    json.dump(gp_metrics, f, indent=2)

print("[GP baseline] Metrics saved to gp_metrics.json")
print("[GP baseline] TEST metrics:", gp_metrics["test"])

cal_pred_df = pd.DataFrame({
    "split": "cal",
    "y_true_HV": y_cal,
    "mu_HV": mu_cal,
    "std_HV": std_cal,
    "residual_HV": (y_cal - mu_cal),
    "z_residual": (y_cal - mu_cal) / std_cal
})
test_pred_df = pd.DataFrame({
    "split": "test",
    "y_true_HV": y_test,
    "mu_HV": mu_test,
    "std_HV": std_test,
    "residual_HV": (y_test - mu_test),
    "z_residual": (y_test - mu_test) / std_test
})

cal_pred_df.to_csv("gp_pointwise_predictions_cal.csv", index=False)
test_pred_df.to_csv("gp_pointwise_predictions_test.csv", index=False)

# ===================
# Visualizations
# ===================
# Parity plot (TEST) with uncertainty
plt.figure(figsize=(6.8, 6.5))
sc = plt.scatter(y_test, mu_test, c=std_test, s=22)
minv = float(min(y_test.min(), mu_test.min()))
maxv = float(max(y_test.max(), mu_test.max()))
plt.plot([minv, maxv], [minv, maxv], linewidth=1)
plt.xlabel("True hardness (HV)")
plt.ylabel("GP predicted mean (HV)")
plt.title("GP parity plot with uncertainty (color = predictive std, test set)")
cbar = plt.colorbar(sc)
cbar.set_label("Predictive std (HV)")
plt.tight_layout()
plt.savefig("gp_parity_test_color_std.png", dpi=300)
plt.close()

z_parity = 1.645
N = len(y_test)

max_points = 500
if N > max_points:
    idx = np.random.RandomState(seed).choice(np.arange(N), size=max_points, replace=False)
else:
    idx = np.arange(N)

y_true_sub = y_test[idx]
mu_sub = mu_test[idx]
std_sub = std_test[idx]

plt.figure(figsize=(6.8, 6.5))
plt.errorbar(
    y_true_sub, mu_sub,
    yerr=z_parity * std_sub,
    fmt="o", markersize=3.5, linewidth=0.7, capsize=0, alpha=0.9
)
minv = float(min(y_true_sub.min(), mu_sub.min()))
maxv = float(max(y_true_sub.max(), mu_sub.max()))
plt.plot([minv, maxv], [minv, maxv], linewidth=1)
plt.xlabel("True hardness (HV)")
plt.ylabel("GP predicted mean (HV)")
plt.title(f"GP parity plot with ±{z_parity:g}σ intervals (test set)")
plt.tight_layout()
plt.savefig("gp_parity_test_errorbars.png", dpi=300)
plt.close()


# ==========================
# GP-BO inverse design
# ==========================
SEQUENTIAL_UPDATE = True
FANTASY_MODE = "posterior_sample"
REFIT_EVERY = 1
FANTASY_RNG = np.random.RandomState(seed + 123)

L_design_raw = 0.5
L_design_scaled = scaler_load.transform(np.array([[L_design_raw]], dtype=np.float64)).reshape(1, 1)

def build_gp_input_from_comp(X_comp: np.ndarray, L_scaled_1x1: np.ndarray) -> np.ndarray:
    X_comp = np.asarray(X_comp, dtype=np.float64)
    if X_comp.ndim == 1:
        X_comp = X_comp.reshape(1, -1)
    N = X_comp.shape[0]
    L_block = np.repeat(L_scaled_1x1.astype(np.float64), repeats=N, axis=0)
    return np.hstack([X_comp, L_block]).astype(np.float64)

def gp_predict_at_design_load(X_comp: np.ndarray):
    X_gp = build_gp_input_from_comp(X_comp, L_design_scaled)
    mu, std = gp_model.predict(X_gp, return_std=True)
    std = np.maximum(std, 1e-12)
    return mu.reshape(-1), std.reshape(-1)

def fantasy_evaluate(x_comp: np.ndarray):
    mu, std = gp_predict_at_design_load(x_comp.reshape(1, -1))
    mu0, std0 = float(mu[0]), float(std[0])
    if FANTASY_MODE == "posterior_mean":
        return mu0
    return float(FANTASY_RNG.normal(loc=mu0, scale=max(std0, 1e-12)))

# Simplex parameterization (logits -> softmax) with floor
def softmax(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    u = u - np.max(u)
    e = np.exp(u)
    return e / np.sum(e)

def to_simplex(u: np.ndarray, x_min: float) -> np.ndarray:
    x = softmax(u)
    x = np.maximum(x, x_min)
    x = x / np.sum(x)
    return x

def comp_to_logits(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.log(np.maximum(x, eps))

# Acquisition
Z_LCB = 1.645
X_MIN_FLOOR = 1e-6

def objective_minus_lcb(u: np.ndarray) -> float:
    x = to_simplex(u, x_min=X_MIN_FLOOR)
    mu, std = gp_predict_at_design_load(x.reshape(1, -1))
    lcb = mu[0] - Z_LCB * std[0]
    return float(-lcb)

# Trust region anchors around best observed points near fixed load
train_fit_loads_raw = X_load_all_raw[train_fit_idx].reshape(-1).astype(np.float64)
train_fit_comps = X_comp_all[train_fit_idx].astype(np.float64)
train_fit_y = y_all[train_fit_idx].astype(np.float64)

abs_dL = np.abs(train_fit_loads_raw - L_design_raw)

POOL_FRAC = 0.20
POOL_MIN = 50
pool_size = int(max(POOL_MIN, POOL_FRAC * len(train_fit_idx)))
pool_idx = np.argsort(abs_dL)[:pool_size]

K_ANCHORS = 15
top_in_pool = pool_idx[np.argsort(train_fit_y[pool_idx])[::-1][:K_ANCHORS]]

anchors_comp = train_fit_comps[top_in_pool].copy()
anchors_y = train_fit_y[top_in_pool].copy()
anchors_load = train_fit_loads_raw[top_in_pool].copy()

anchors_df = pd.DataFrame({
    "anchor_id": np.arange(len(top_in_pool)),
    "load_raw": anchors_load,
    "y_measured_HV": anchors_y,
})
for j, col in enumerate(composition_cols):
    anchors_df[col] = anchors_comp[:, j]
anchors_df.to_csv("gpbo_anchors_L1N.csv", index=False)

TRUST_R = 0.12
TRUST_PENALTY = 1e4

def min_dist_to_anchors_L2(x: np.ndarray) -> float:
    d = np.sqrt(np.sum((anchors_comp - x.reshape(1, -1))**2, axis=1))
    return float(np.min(d))

def trust_penalty(x: np.ndarray) -> float:
    dmin = min_dist_to_anchors_L2(x)
    if dmin <= TRUST_R:
        return 0.0
    return float(TRUST_PENALTY * (dmin - TRUST_R)**2)

# Novelty diagnostic (distance to nearest train_fit point)
def novelty_to_trainfit_L2(x: np.ndarray) -> float:
    d = np.sqrt(np.sum((train_fit_comps - x.reshape(1, -1))**2, axis=1))
    return float(np.min(d))

# Diversity/repulsion against previously proposed candidates
REPULSE_R = 0.02
REPULSE_PENALTY = 2e4

def min_dist_to_seen_L2(x: np.ndarray, X_seen_mat: np.ndarray) -> float:
    if X_seen_mat is None or len(X_seen_mat) == 0:
        return float("inf")
    X_seen_mat = np.asarray(X_seen_mat, dtype=np.float64)
    d = np.sqrt(np.sum((X_seen_mat - x.reshape(1, -1))**2, axis=1))
    return float(np.min(d))

def repulsion_penalty(x: np.ndarray, X_seen_mat: np.ndarray) -> float:
    dmin = min_dist_to_seen_L2(x, X_seen_mat)
    if not np.isfinite(dmin) or dmin >= REPULSE_R:
        return 0.0
    return float(REPULSE_PENALTY * (REPULSE_R - dmin)**2)

def objective_minus_lcb_with_trust_and_repulsion(u: np.ndarray, X_seen_mat: np.ndarray) -> float:
    x = to_simplex(u, x_min=X_MIN_FLOOR)
    base = objective_minus_lcb(u)
    return float(base + trust_penalty(x) + repulsion_penalty(x, X_seen_mat))


# ==========================================
# Acquisition optimization (multi-start)
# ==========================================
M_RESTARTS = 100
NOISE_SCALE = 0.30
MAXITER = 1000
FTOL = 1e-10

rng = np.random.RandomState(seed)

def propose_next(X_seen_mat: np.ndarray):
    starts = []

    for a in anchors_comp:
        starts.append(comp_to_logits(np.maximum(a, X_MIN_FLOOR)))

    while len(starts) < M_RESTARTS:
        a = anchors_comp[rng.randint(0, anchors_comp.shape[0])]
        u0 = comp_to_logits(np.maximum(a, X_MIN_FLOOR)) + rng.normal(scale=NOISE_SCALE, size=a.shape[0])
        starts.append(u0)

    best_fun = np.inf
    best_u = None
    best_res = None

    obj = lambda u: objective_minus_lcb_with_trust_and_repulsion(u, X_seen_mat)

    for u0 in starts[:M_RESTARTS]:
        res = minimize(
            fun=obj,
            x0=u0,
            method="L-BFGS-B",
            options={"maxiter": MAXITER, "ftol": FTOL},
        )
        if np.isfinite(res.fun) and float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_u = res.x.copy()
            best_res = res

    x_best = to_simplex(best_u, x_min=X_MIN_FLOOR)
    mu_best, std_best = gp_predict_at_design_load(x_best.reshape(1, -1))
    lcb_best = float(mu_best[0] - Z_LCB * std_best[0])

    info = {
        "opt_success": int(bool(best_res.success)),
        "opt_nit": int(best_res.nit),
        "objective_value": float(best_fun),
        "min_dist_anchor_L2": min_dist_to_anchors_L2(x_best),
        "novelty_L2_to_trainfit": novelty_to_trainfit_L2(x_best),
        "min_dist_seen_L2": min_dist_to_seen_L2(x_best, X_seen_mat),
    }
    return x_best, float(mu_best[0]), float(std_best[0]), float(lcb_best), info


def refit_gp_on_augmented_data(X_obs_gp: np.ndarray, y_obs: np.ndarray):
    return fit_gp_fixed(
        kernel_amp=float(best_gp_params["kernel_amp"]),
        length_scale=float(best_gp_params["length_scale"]),
        noise_level=float(best_gp_params["noise_level"]),
        X_train=X_obs_gp,
        y_train=y_obs,
        alpha_jitter=float(best_gp_params.get("alpha_jitter", 1e-10)),
        nu=2.5
    )


# ================
# Run BO loop
# ================
T = 200

X_obs_gp = X_train_fit_gp.copy()
y_obs = y_train_fit.copy().reshape(-1)

X_seen = []
trace_rows = []

for t in range(T):
    X_seen_mat = np.asarray(X_seen, dtype=np.float64) if len(X_seen) > 0 else np.empty((0, anchors_comp.shape[1]), dtype=np.float64)
    x_next, mu_next, std_next, lcb_next, info = propose_next(X_seen_mat)
    X_seen.append(x_next.copy())
    y_next_f = np.nan
    if SEQUENTIAL_UPDATE:
        y_next_f = fantasy_evaluate(x_next)
        X_next_gp = build_gp_input_from_comp(x_next.reshape(1, -1), L_design_scaled)
        X_obs_gp = np.vstack([X_obs_gp, X_next_gp])
        y_obs = np.concatenate([y_obs, np.array([y_next_f], dtype=np.float64)])

        if ((t + 1) % REFIT_EVERY) == 0:
            gp_model = refit_gp_on_augmented_data(X_obs_gp, y_obs)

    trace_rows.append({
        "iter": int(t),
        "mu_HV": float(mu_next),
        "std_HV": float(std_next),
        "LCB_HV": float(lcb_next),
        "fantasy_y_HV": float(y_next_f) if np.isfinite(y_next_f) else np.nan,
        "min_dist_anchor_L2": float(info["min_dist_anchor_L2"]),
        "novelty_L2_to_trainfit": float(info["novelty_L2_to_trainfit"]),
        "min_dist_seen_L2": float(info["min_dist_seen_L2"]),
        "opt_success": int(info["opt_success"]),
        "opt_nit": int(info["opt_nit"]),
        "objective_value": float(info["objective_value"]),
    })

    if (t + 1) % 10 == 0:
        best_so_far = float(np.max([r["LCB_HV"] for r in trace_rows]))
        print(f"[GP-BO @1N] iter {t+1:3d}/{T} | current LCB={lcb_next:8.2f} | best LCB so far={best_so_far:8.2f}")

trace_df = pd.DataFrame(trace_rows)
trace_df.to_csv("gpbo_trace_L1N.csv", index=False)
trace_df.to_csv("origin_gpbo_trace_L1N.csv", index=False)

X_seen = np.asarray(X_seen, dtype=np.float64)
X_round = np.round(X_seen, 6)
_, unique_idx = np.unique(X_round, axis=0, return_index=True)
unique_idx = np.sort(unique_idx)

cand_rows = []
for rank_id, idx in enumerate(unique_idx):
    x = X_seen[idx]
    mu, std = gp_predict_at_design_load(x.reshape(1, -1))
    lcb = float(mu[0] - Z_LCB * std[0])
    cand_rows.append({
        "candidate_id": int(rank_id),
        "iter_first_seen": int(idx),
        "mu_HV": float(mu[0]),
        "std_HV": float(std[0]),
        "LCB_HV": float(lcb),
        "min_dist_anchor_L2": float(min_dist_to_anchors_L2(x)),
        "novelty_L2_to_trainfit": float(novelty_to_trainfit_L2(x)),
        "min_dist_seen_L2": float(min_dist_to_seen_L2(x, X_seen)),
    })

cand_df = pd.DataFrame(cand_rows).sort_values("LCB_HV", ascending=False).reset_index(drop=True)
for j, col in enumerate(composition_cols):
    cand_df[col] = [X_seen[unique_idx[i], j] for i in range(len(unique_idx))]

cand_df.to_csv("gpbo_candidates_L1N.csv", index=False)

TOPN = min(50, len(cand_df))
cand_df.head(TOPN).to_csv("origin_gpbo_top_candidates_L1N.csv", index=False)

print("[GP-BO @1N] Saved: gpbo_anchors_L1N.csv, gpbo_trace_L1N.csv, gpbo_candidates_L1N.csv")

# ====================
# Visualizations
# ====================
best_lcb = np.maximum.accumulate(trace_df["LCB_HV"].values)
plt.figure(figsize=(7.2, 4.8))
plt.plot(trace_df["iter"].values, best_lcb, linewidth=2)
plt.xlabel("BO iteration")
plt.ylabel(f"Best LCB (HV), z={Z_LCB:g}")
plt.title("Sequential local GP-BO @ 1 N: best conservative hardness vs iteration")
plt.tight_layout()
plt.savefig("gpbo_bestLCB_trace.png", dpi=300)
plt.close()

plt.figure(figsize=(7.0, 5.2))
plt.scatter(cand_df["mu_HV"].values[:TOPN], cand_df["std_HV"].values[:TOPN], s=28)
plt.xlabel("Predicted mean hardness μ (HV)")
plt.ylabel("Predictive std σ (HV)")
plt.title("Top sequential local GP-BO candidates @ 1 N (ranked by LCB)")
plt.tight_layout()
plt.savefig("gpbo_mu_vs_sigma.png", dpi=300)
plt.close()

x_axis = np.arange(TOPN)
plt.figure(figsize=(7.2, 5.0))
plt.plot(x_axis, cand_df["mu_HV"].values[:TOPN], marker="o", linewidth=1.6, label="μ")
plt.plot(x_axis, cand_df["LCB_HV"].values[:TOPN], marker="o", linewidth=1.6, label=f"LCB (z={Z_LCB:g})")
plt.xlabel("Candidate rank (by LCB)")
plt.ylabel("Hardness (HV)")
plt.title("Mean vs conservative bound (LCB) for top sequential GP-BO candidates @ 1 N")
plt.legend()
plt.tight_layout()
plt.savefig("gpbo_mean_vs_lcb_topN.png", dpi=300)
plt.close()

plt.figure(figsize=(7.0, 5.2))
plt.scatter(cand_df["novelty_L2_to_trainfit"].values[:TOPN], cand_df["LCB_HV"].values[:TOPN], s=28)
plt.xlabel("Novelty distance to nearest train_fit point (L2 in composition)")
plt.ylabel(f"LCB (HV), z={Z_LCB:g}")
plt.title("Sequential local GP-BO @ 1 N: novelty vs conservative performance (top candidates)")
plt.tight_layout()
plt.savefig("gpbo_novelty_vs_lcb.png", dpi=300)
plt.close()

print("[GP-BO @1N] Saved figures:",
      "gpbo_bestLCB_trace.png, gpbo_mu_vs_sigma.png, gpbo_mean_vs_lcb_topN.png, gpbo_novelty_vs_lcb.png")


TOPM = min(200, len(cand_df))
diag_df = cand_df.head(TOPM).copy()
diag_df.to_csv("gpbo_diagnostics_topM.csv", index=False)

# novelty distribution
plt.figure(figsize=(7.0, 4.8))
plt.hist(diag_df["novelty_L2_to_trainfit"].values, bins=30)
plt.xlabel("Novelty distance to nearest train_fit point (L2 in composition)")
plt.ylabel("Count")
plt.title(f"Sequential local GP-BO @1N: novelty distribution (top {TOPM} by LCB)")
plt.tight_layout()
plt.savefig("gpbo_hist_novelty_topM.png", dpi=300)
plt.close()

# sigma distribution
plt.figure(figsize=(7.0, 4.8))
plt.hist(diag_df["std_HV"].values, bins=30)
plt.xlabel("GP predictive std σ (HV)")
plt.ylabel("Count")
plt.title(f"Sequential local GP-BO @1N: uncertainty distribution (top {TOPM} by LCB)")
plt.tight_layout()
plt.savefig("gpbo_hist_sigma_topM.png", dpi=300)
plt.close()

# clustering near training points: proposals -> nearest train_fit vs train_fit NN spacing reference
d_to_trainfit_all = np.array([novelty_to_trainfit_L2(x) for x in X_seen], dtype=np.float64)
pd.DataFrame({"dist_to_nearest_trainfit_L2_proposals": d_to_trainfit_all}).to_csv(
    "gpbo_cluster_distances_all_proposals.csv", index=False
)

subN = min(600, train_fit_comps.shape[0])
rng2 = np.random.RandomState(seed + 999)
sub_idx = rng2.choice(np.arange(train_fit_comps.shape[0]), size=subN, replace=False)
X_sub = train_fit_comps[sub_idx]

D = cdist(X_sub, X_sub, metric="euclidean")
np.fill_diagonal(D, np.inf)
nn_trainfit = D.min(axis=1)

pd.DataFrame({"trainfit_nn_distance_L2_reference": nn_trainfit}).to_csv(
    "gpbo_trainfit_nn_reference.csv", index=False
)

plt.figure(figsize=(7.0, 4.8))
plt.hist(nn_trainfit, bins=30, alpha=0.7, label="Train_fit NN spacing (reference)")
plt.hist(d_to_trainfit_all, bins=30, alpha=0.7, label="Proposals → nearest train_fit")
plt.xlabel("L2 distance in composition space")
plt.ylabel("Count")
plt.title("Do sequential GP-BO proposals cluster near the training set?")
plt.legend()
plt.tight_layout()
plt.savefig("gpbo_hist_proposals_vs_trainfit_spacing.png", dpi=300)
plt.close()
