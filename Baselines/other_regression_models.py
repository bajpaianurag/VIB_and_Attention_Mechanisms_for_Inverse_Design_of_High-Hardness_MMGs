"""Baseline regressors for hardness prediction.

This script trains a set of conventional regressors on the same input features
(composition + load) used by the VIB+Attention model and reports comparable
metrics on a held-out test split.

Key points (to match the paper setup):
- Uses a fixed random seed for reproducibility.
- Reads the shared dataset from ../data/H_v_dataset.csv.
- Performs a single train/test split (default 80/20).
- Optionally performs Bayesian hyperparameter search (scikit-optimize).

Outputs:
- results/baselines_metrics.json
- results/<model_name>_parity.png

Run:
    python baselines/other_regression_models.py

Optional:
    python baselines/other_regression_models.py --n-iter 30 --cv 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor

# Optional dependencies (kept optional to reduce install friction)
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical

    _HAS_SKOPT = True
except Exception:
    BayesSearchCV = None
    Real = Integer = Categorical = None
    _HAS_SKOPT = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam, SGD
    from scikeras.wrappers import KerasRegressor

    _HAS_TF = True
except Exception:
    tf = None
    Sequential = Dense = Adam = SGD = KerasRegressor = None
    _HAS_TF = False


SEED = 42


def build_mlp(input_dim: int, hidden_layers: int = 2, hidden_units: int = 64,
              learning_rate: float = 1e-3, activation: str = "relu",
              optimizer: str = "adam"):
    if not _HAS_TF:
        raise RuntimeError("TensorFlow + SciKeras are required for the MLP baseline.")

    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(input_dim,), activation=activation))
    for _ in range(max(0, hidden_layers - 1)):
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(1))

    opt = Adam(learning_rate=learning_rate) if optimizer == "adam" else SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=opt)
    return model


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, outpath: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=30)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True HV")
    plt.ylabel("Predicted HV")
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def get_search_spaces(input_dim: int):
    spaces = {
        "RF": (
            RandomForestRegressor(random_state=SEED),
            {
                "n_estimators": Integer(200, 1200),
                "max_depth": Integer(2, 30),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf": Integer(1, 10),
                "max_features": Real(0.2, 1.0),
                "bootstrap": Categorical([True, False]),
            },
        ),
        "GB": (
            GradientBoostingRegressor(random_state=SEED),
            {
                "n_estimators": Integer(200, 1500),
                "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
                "max_depth": Integer(1, 6),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf": Integer(1, 10),
                "subsample": Real(0.5, 1.0),
            },
        ),
        "Lasso": (
            Lasso(random_state=SEED, max_iter=20000),
            {"alpha": Real(1e-6, 1.0, prior="log-uniform")},
        ),
        "Ridge": (
            Ridge(random_state=SEED, max_iter=20000),
            {"alpha": Real(1e-6, 100.0, prior="log-uniform")},
        ),
        "KNN": (
            KNeighborsRegressor(),
            {
                "n_neighbors": Integer(2, 80),
                "weights": Categorical(["uniform", "distance"]),
                "p": Categorical([1, 2]),
            },
        ),
    }

    if _HAS_TF:
        spaces["MLP"] = (
            KerasRegressor(
                model=lambda hidden_layers=2, hidden_units=64, learning_rate=1e-3,
                            activation="relu", optimizer="adam":
                build_mlp(
                    input_dim=input_dim,
                    hidden_layers=hidden_layers,
                    hidden_units=hidden_units,
                    learning_rate=learning_rate,
                    activation=activation,
                    optimizer=optimizer,
                ),
                verbose=0,
            ),
            {
                "model__hidden_layers": Integer(1, 4),
                "model__hidden_units": Integer(16, 256),
                "model__learning_rate": Real(1e-4, 3e-2, prior="log-uniform"),
                "model__activation": Categorical(["relu", "tanh"]),
                "model__optimizer": Categorical(["adam", "sgd"]),
                "batch_size": Integer(16, 64),
                "epochs": Integer(100, 500),
            },
        )

    return spaces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-iter", type=int, default=25, help="BayesSearchCV iterations")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--no-bayes", action="store_true", help="Disable Bayesian search; fit defaults")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "H_v_dataset.csv"

    df = pd.read_csv(data_path)
    if "HV" not in df.columns:
        raise ValueError("Expected column 'HV' in dataset.")

    X = df.drop(columns=["HV"]).to_numpy(dtype=float)
    y = df["HV"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED
    )

    out_dir = root / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}

    if not args.no_bayes and not _HAS_SKOPT:
        raise RuntimeError(
            "scikit-optimize is not installed, but Bayesian search was requested. "
            "Install extras (see requirements) or use --no-bayes."
        )

    spaces = get_search_spaces(input_dim=X.shape[1]) if _HAS_SKOPT else {}

    for name in ["RF", "GB", "Lasso", "Ridge", "KNN", "MLP"]:
        if name not in spaces:
            continue

        base_model, search_space = spaces[name]

        if args.no_bayes:
            model = base_model
            model.fit(X_train, y_train)
        else:
            opt = BayesSearchCV(
                base_model,
                search_space,
                n_iter=args.n_iter,
                cv=args.cv,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
                random_state=SEED,
            )
            opt.fit(X_train, y_train)
            model = opt.best_estimator_
            metrics[name + "_best_params"] = opt.best_params_

        y_pred = model.predict(X_test)

        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

        parity_plot(y_test, y_pred, f"{name} baseline", out_dir / f"{name}_parity.png")

    (out_dir / "baselines_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
