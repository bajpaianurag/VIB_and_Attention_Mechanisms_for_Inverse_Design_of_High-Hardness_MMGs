# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.calibration import calibration_curve
from sklearn.mixture import GaussianMixture
import scipy.stats
from scipy.stats import norm, pearsonr, spearmanr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tqdm import trange
import optuna
from optuna.samplers import TPESampler
from collections import Counter
import random
import os
from sklearn.utils import shuffle
import csv
import json

# Introduce fixed seed (run for 10 seeds and average the whole pipeline)
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Data Ingestion
from pathlib import Path

# -------------------------
# Data ingestion
# -------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'H_v_dataset.csv'
data = pd.read_csv(DATA_PATH)
print("\nBasic Statistics:")
print(data.describe())

composition_cols = [col for col in data.columns if col not in ['Load', 'HV']]
load_col = 'Load'
target_col = 'HV'

# Hardness Distribution Plot
plt.style.use('default')
sns.set_context("talk")
plt.figure(figsize=(10, 8))
sns.histplot(data['HV'], kde=True, color='grey', bins=15, edgecolor='black')
plt.xlabel('Hardness (HV)', fontsize=26)
plt.ylabel('Frequency', fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Distribution of Hardness (HV).jpg", dpi=600, format='jpeg')
plt.show()

# Hardness-Load Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['Load'], y=data['HV'], color='steelblue', s=180, edgecolor='black')
plt.xlabel('Load (N)', fontsize=26)
plt.ylabel('Hardness (HV)', fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Relationship Between Load and Hardness (HV).jpg", dpi=600, format='jpeg')
plt.show()


# =========================================================
# Data Preparation (with family-aware debiasing weights)
# =========================================================
X_comp_all = data[composition_cols].values.astype(np.float32)
row_sum = X_comp_all.sum(axis=1, keepdims=True)
row_sum = np.where(row_sum <= 0, 1.0, row_sum)
X_comp_all = X_comp_all / row_sum

X_load_all_raw = data[[load_col]].values.astype(np.float32)
y_all = data["HV"].values.astype(np.float32)

# Family-aware debiasing weights (computed on normalized X_comp_all)
EPS_NONZERO = 1e-12

nonzero_mask = X_comp_all > EPS_NONZERO
family_keys = []
for i in range(nonzero_mask.shape[0]):
    elems = tuple(np.array(composition_cols, dtype=object)[nonzero_mask[i]].tolist())
    family_keys.append(elems)
family_keys = np.array(family_keys, dtype=object)

family_counts = Counter(family_keys.tolist())

w_base = np.array([1.0 / np.sqrt(family_counts[k]) for k in family_keys], dtype=np.float32)

BFeNb_ONLY_FACTOR = 0.35
is_bfenb_only = np.array([set(k) == {"B", "Fe", "Nb"} for k in family_keys], dtype=bool)
w_base[is_bfenb_only] *= float(BFeNb_ONLY_FACTOR)

w_all = (w_base / (np.mean(w_base) + 1e-12)).astype(np.float32)

W_MIN, W_MAX = 0.25, 4.0
w_all = np.clip(w_all, W_MIN, W_MAX).astype(np.float32)

def make_sample_weight_dict(w: np.ndarray) -> dict:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    return {
        "Output_Layer": w,
        "Comp_Recon": w,
        "Comp_Attention": np.ones_like(w, dtype=np.float32),
        "Load_Attention": np.ones_like(w, dtype=np.float32),
    }

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
cluster_labels = kmeans.fit_predict(X_comp_all)

combined_labels = np.array([f"{cluster_labels[i]}|{family_keys[i]}" for i in range(len(cluster_labels))], dtype=object)

min_count = 5
combo_counts = Counter(combined_labels.tolist())
combined_labels = np.array(
    [lab if combo_counts[lab] >= min_count else "RARE" for lab in combined_labels],
    dtype=object
)

_, combined_codes = np.unique(combined_labels, return_inverse=True)

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
for train_idx, test_idx in strat_split.split(X_comp_all, combined_codes):
    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

X_comp_train = X_comp_all[train_idx].astype(np.float32)
X_comp_test  = X_comp_all[test_idx].astype(np.float32)

X_load_train_raw = X_load_all_raw[train_idx].astype(np.float32)
X_load_test_raw  = X_load_all_raw[test_idx].astype(np.float32)

y_train = y_all[train_idx].astype(np.float32)
y_test  = y_all[test_idx].astype(np.float32)

w_train = w_all[train_idx].astype(np.float32)
w_test  = w_all[test_idx].astype(np.float32)

scaler_load = StandardScaler()
scaler_load.fit(X_load_train_raw)

X_load_train = scaler_load.transform(X_load_train_raw).astype(np.float32)
X_load_test  = scaler_load.transform(X_load_test_raw).astype(np.float32)

train_df = data.iloc[train_idx].reset_index(drop=True)
test_df  = data.iloc[test_idx].reset_index(drop=True)

train_mean = X_comp_train.mean(axis=0)
test_mean  = X_comp_test.mean(axis=0)

compare_df = pd.DataFrame(
    {"Train": train_mean, "Test": test_mean},
    index=composition_cols
)
compare_df.plot(kind="bar", figsize=(14, 6))
plt.ylabel("Mean Fraction")
plt.title("Elemental Fraction Distribution: Train vs Test")
plt.tight_layout()
plt.show()


# ============================
# Create a CALIBRATION split
# ============================
ALPHA = 0.10
CAL_FRAC = 0.20

sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=seed)

train_fit_rel, cal_rel = next(sss_cal.split(X_comp_all[train_idx], combined_codes[train_idx]))

train_fit_idx = train_idx[train_fit_rel]
cal_idx       = train_idx[cal_rel]

X_comp_train_fit = X_comp_all[train_fit_idx].astype(np.float32)
X_load_train_fit_raw = X_load_all_raw[train_fit_idx].astype(np.float32)
y_train_fit = y_all[train_fit_idx].astype(np.float32).reshape(-1)

X_comp_cal = X_comp_all[cal_idx].astype(np.float32)
X_load_cal_raw = X_load_all_raw[cal_idx].astype(np.float32)
y_cal = y_all[cal_idx].astype(np.float32).reshape(-1)

X_comp_test = X_comp_all[test_idx].astype(np.float32)
X_load_test_raw = X_load_all_raw[test_idx].astype(np.float32)
y_test_arr = y_all[test_idx].astype(np.float32).reshape(-1)

w_train_fit = w_all[train_fit_idx].astype(np.float32)
w_cal       = w_all[cal_idx].astype(np.float32)
w_test      = w_all[test_idx].astype(np.float32)

debias_audit = pd.DataFrame({
    "family_key": [",".join(list(k)) for k in family_keys],
    "is_BFeNb_only": is_bfenb_only.astype(int),
    "weight": w_all.astype(np.float32),
})
debias_audit.to_csv("debias_family_weights_audit.csv", index=False)

family_summary = (
    pd.DataFrame({"family_key": [",".join(list(k)) for k in family_counts.keys()],
                  "count": list(family_counts.values())})
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)
family_summary.to_csv("debias_family_counts.csv", index=False)


# ===================================
# VIBANN Model Architecture
# ===================================
# VIB Layer
class VIBLayer(layers.Layer):
    def __init__(self, latent_dim=16, beta_init=1e-3, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        self.eps = float(eps)
        self.beta = tf.Variable(beta_init, trainable=False, dtype=tf.float32, name="beta")
        self.kl_per_dim_last = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_per_dim_last")
        self.to_mu = layers.Dense(self.latent_dim, name="z_mu")
        self.to_logvar = layers.Dense(self.latent_dim, name="z_logvar")

    def build(self, input_shape):
        self.to_mu.build(input_shape)
        self.to_logvar.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.latent_dim)

    def call(self, inputs, training=None):
        x = tf.convert_to_tensor(inputs)

        mu = self.to_mu(x)
        logvar = self.to_logvar(x)

        if training:
            eps = tf.random.normal(shape=tf.shape(mu), dtype=mu.dtype)
            z = mu + tf.exp(0.5 * logvar) * eps
        else:
            z = mu

        kl_per_sample = 0.5 * tf.reduce_sum(
            tf.exp(logvar) + tf.square(mu) - 1.0 - logvar,
            axis=-1
        )

        kl_per_dim = kl_per_sample / tf.cast(self.latent_dim, tf.float32)
        kl_per_dim_mean = tf.reduce_mean(kl_per_dim)

        self.kl_per_dim_last.assign(tf.cast(kl_per_dim_mean, tf.float32))
        self.add_loss(self.beta * tf.cast(kl_per_dim_mean, tf.float32))

        return z


# Feature-Wise Attention Layer
class FeatureWiseAttention(layers.Layer):

    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        self.attn_logits = layers.Dense(self.n_features, activation=None, name="attn_logits")

    def call(self, inputs):
        logits = self.attn_logits(inputs)
        attention_scores = tf.nn.softmax(logits, axis=-1)
        weighted_inputs = inputs * attention_scores
        return weighted_inputs, attention_scores

class AttentionScoreLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        X_comp_ref,
        X_load_ref,
        comp_feature_count: int,
        load_feature_count: int,
        fold_id: int,
        output_dir: str = ".",
        output_prefix: str = "attention_scores",
        sample_size: int | None = 2048,
        seed: int = 1,
        mc_samples: int = 16,
        mc_batch_size: int = 512,
    ):
        super().__init__()
        self.comp_feature_count = int(comp_feature_count)
        self.load_feature_count = int(load_feature_count)
        self.fold_id = int(fold_id)

        Xc = np.asarray(X_comp_ref, dtype=np.float32)
        Xl = np.asarray(X_load_ref, dtype=np.float32)

        if sample_size is not None and Xc.shape[0] > sample_size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(Xc.shape[0], size=sample_size, replace=False)
            Xc = Xc[idx]
            Xl = Xl[idx]

        self.X_comp_ref = Xc
        self.X_load_ref = Xl

        self.mc_samples = int(mc_samples)
        self.mc_batch_size = int(mc_batch_size)

        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f"{output_prefix}_fold_{self.fold_id}.csv")
        self.attention_scores = []

        header = (
            [f"Comp_Feature_{i+1}" for i in range(self.comp_feature_count)]
            + [f"Load_Feature_{j+1}" for j in range(self.load_feature_count)]
        )

        with open(self.output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch"] + header)

    def _mc_average_attention(self):
        Xc = self.X_comp_ref
        Xl = self.X_load_ref
        n = Xc.shape[0]

        ds = tf.data.Dataset.from_tensor_slices((Xc, Xl)).batch(self.mc_batch_size)

        comp_acc = np.zeros((self.comp_feature_count,), dtype=np.float64)
        load_acc = np.zeros((self.load_feature_count,), dtype=np.float64)

        for _ in range(self.mc_samples):
            comp_epoch = []
            load_epoch = []
            for xb, xl in ds:
                outputs = self.model([xb, xl], training=False)
                comp_att = outputs["Comp_Attention"].numpy()
                load_att = outputs["Load_Attention"].numpy() 
                comp_epoch.append(comp_att)
                load_epoch.append(load_att)

            comp_all = np.concatenate(comp_epoch, axis=0)
            load_all = np.concatenate(load_epoch, axis=0)

            comp_acc += comp_all.mean(axis=0)
            load_acc += load_all.mean(axis=0)

        comp_acc /= float(self.mc_samples)
        load_acc /= float(self.mc_samples)

        return comp_acc.astype(np.float32), load_acc.astype(np.float32)

    def on_epoch_end(self, epoch, logs=None):
        avg_comp_attention, avg_load_attention = self._mc_average_attention()

        all_attention_scores = np.concatenate([avg_comp_attention, avg_load_attention], axis=0)
        self.attention_scores.append(all_attention_scores)

        with open(self.output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1] + all_attention_scores.tolist())


# ------------------------
# Other helpers
# ------------------------
class PairwiseProducts(tf.keras.layers.Layer):
    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        if self.n_features < 2:
            raise ValueError("PairwiseProducts requires n_features >= 2.")
        self.n_pairs = self.n_features * (self.n_features - 1) // 2
        idx_i, idx_j = [], []
        for i in range(self.n_features - 1):
            for j in range(i + 1, self.n_features):
                idx_i.append(i)
                idx_j.append(j)
        self.idx_i = tf.constant(idx_i, dtype=tf.int32)
        self.idx_j = tf.constant(idx_j, dtype=tf.int32)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError("PairwiseProducts requires known last dimension.")
        if int(input_shape[-1]) != self.n_features:
            raise ValueError(f"Expected input last-dim {self.n_features}, got {int(input_shape[-1])}.")
        super().build(input_shape)

    def call(self, x):
        x = tf.convert_to_tensor(x)
        xi = tf.gather(x, self.idx_i, axis=-1)
        xj = tf.gather(x, self.idx_j, axis=-1)
        return xi * xj

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_pairs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features})
        return cfg


class MCDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, mc_active=True, **kwargs):
        super().__init__(rate, **kwargs)
        self.mc_active = bool(mc_active)

    def call(self, inputs, training=None):
        if training is True:
            return super().call(inputs, training=True)

        if training is None:
            return super().call(inputs, training=False)

        return super().call(inputs, training=self.mc_active)


def set_mc_dropout(model: tf.keras.Model, active: bool) -> None:
    active = bool(active)

    def _recurse(layer):
        if layer.__class__.__name__ == "MCDropout" and hasattr(layer, "mc_active"):
            layer.mc_active = active

        if isinstance(layer, tf.keras.Model):
            for sub in layer.layers:
                _recurse(sub)

    for lyr in model.layers:
        _recurse(lyr)


# -------------------------
# Build vibann model
# -------------------------
def build_vib_attention_model(
    input_shape_comp,
    input_shape_load,
    latent_dim=16,
    dropout_rate=0.3,
    recon_weight=0.2,
    load_recon_weight=0.0
):

    # Simplex projection layer
    class SimplexNormalize(tf.keras.layers.Layer):
        def __init__(self, eps=1e-8, **kwargs):
            super().__init__(**kwargs)
            self.eps = float(eps)

        def call(self, x):
            x = tf.nn.relu(x)
            s = tf.reduce_sum(x, axis=-1, keepdims=True)
            s = tf.maximum(s, self.eps)
            return x / s

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"eps": self.eps})
            return cfg

    input_shape_comp = int(input_shape_comp)
    input_shape_load = int(input_shape_load)

    # Inputs
    comp_input = layers.Input(shape=(int(input_shape_comp),), name="Composition_Input")
    load_input = layers.Input(shape=(int(input_shape_load),), name="Load_Input")

    comp_simplex = SimplexNormalize(name="Comp_Simplex")(comp_input)

    # Feature-wise attention
    comp_weighted, comp_attention_scores = FeatureWiseAttention(
        n_features=int(input_shape_comp), name="Comp_Attention"
    )(comp_simplex)

    load_weighted, load_attention_scores = FeatureWiseAttention(
        n_features=int(input_shape_load), name="Load_Attention"
    )(load_input)

    # Encoders
    x_comp_main = layers.Dense(64, activation="relu", name="Comp_Enc_D1")(comp_weighted)
    x_comp_main = layers.BatchNormalization(name="Comp_Enc_BN1")(x_comp_main)
    x_comp_main = MCDropout(dropout_rate, name="Comp_Enc_DO1")(x_comp_main)

    pair_feats = PairwiseProducts(n_features=int(input_shape_comp), name="Comp_Pairwise")(comp_simplex)

    x_comp_pair = layers.Dense(128, activation="relu", name="Comp_Pair_D1")(pair_feats)
    x_comp_pair = layers.BatchNormalization(name="Comp_Pair_BN1")(x_comp_pair)
    x_comp_pair = MCDropout(dropout_rate, name="Comp_Pair_DO1")(x_comp_pair)

    x_comp_pair = layers.Dense(64, activation="relu", name="Comp_Pair_D2")(x_comp_pair)
    x_comp_pair = layers.BatchNormalization(name="Comp_Pair_BN2")(x_comp_pair)
    x_comp_pair = MCDropout(dropout_rate, name="Comp_Pair_DO2")(x_comp_pair)

    x_comp = layers.Concatenate(name="Comp_Fuse")([x_comp_main, x_comp_pair])
    x_comp = layers.Dense(64, activation="relu", name="Comp_Fuse_D1")(x_comp)
    x_comp = layers.BatchNormalization(name="Comp_Fuse_BN1")(x_comp)
    x_comp = MCDropout(dropout_rate, name="Comp_Fuse_DO1")(x_comp)

    x_load = layers.Dense(32, activation="relu", name="Load_Enc_D1")(load_weighted)
    x_load = layers.BatchNormalization(name="Load_Enc_BN1")(x_load)
    x_load = MCDropout(dropout_rate, name="Load_Enc_DO1")(x_load)

    combined = x_comp

    # VIB latent
    z = VIBLayer(latent_dim=int(latent_dim), name="vib_layer")(combined)

    # Predictor head: z -> hardness
    hard_in = layers.Concatenate(name="Hard_Input")([z, x_load])
    
    h = layers.Dense(64, activation="relu", name="Hard_Trunk_D1")(hard_in)
    h = layers.BatchNormalization(name="Hard_Trunk_BN1")(h)
    h = MCDropout(dropout_rate, name="Hard_Trunk_DO1")(h)
    hardness_out = layers.Dense(1, activation="linear", name="Output_Layer")(h)

    # Decoder head: z -> composition reconstruction
    d = layers.Dense(128, activation="relu", name="Comp_Dec_D1")(z)
    d = layers.BatchNormalization(name="Comp_Dec_BN1")(d)
    d = MCDropout(dropout_rate, name="Comp_Dec_DO1")(d)
    
    d = layers.Dense(64, activation="relu", name="Comp_Dec_D2")(d)
    d = layers.BatchNormalization(name="Comp_Dec_BN2")(d)
    d = MCDropout(dropout_rate, name="Comp_Dec_DO2")(d)
    
    comp_recon = layers.Dense(int(input_shape_comp), activation="softmax", name="Comp_Recon")(d)

    model = models.Model(
        inputs=[comp_input, load_input],
        outputs={
            "Output_Layer": hardness_out,
            "Comp_Attention": comp_attention_scores,
            "Load_Attention": load_attention_scores,
            "Comp_Recon": comp_recon,
        },
        name="VIB_Attention_With_Trained_Decoder"
    )
    return model


# Adaptive Beta Callback
class AdaptiveBetaCallback(tf.keras.callbacks.Callback):
    def __init__(self, vib_layer, target_kl_per_dim=0.15, beta_adjustment_factor=1.05,
                 beta_min=1e-6, beta_max=1e+2, verbose=1):
        super().__init__()
        self.vib_layer = vib_layer
        self.target = float(target_kl_per_dim)
        self.factor = float(beta_adjustment_factor)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.verbose = int(verbose)

    def on_epoch_end(self, epoch, logs=None):
        kl = float(self.vib_layer.kl_per_dim_last.numpy())
        beta = float(self.vib_layer.beta.numpy())
        
        if kl < self.target:
            beta_new = beta / self.factor
        else:
            beta_new = beta * self.factor

        beta_new = float(np.clip(beta_new, self.beta_min, self.beta_max))
        self.vib_layer.beta.assign(beta_new)

        if self.verbose:
            print(f"Epoch {epoch+1}: kl_per_dim={kl:.6f}, beta={beta_new:.6e}")


# beta adjustment
def adjust_beta(vib_layer, kl_per_dim_mean, target_kl_per_dim=0.15, beta_adjustment_factor=1.05, beta_min=1e-6, beta_max=1e2):
    kl_per_dim_mean = float(kl_per_dim_mean)

    if kl_per_dim_mean < target_kl_per_dim:
        new_beta = vib_layer.beta / beta_adjustment_factor
    else:
        new_beta = vib_layer.beta * beta_adjustment_factor

    new_beta = float(np.clip(new_beta, beta_min, beta_max))
    vib_layer.beta.assign(new_beta)
    return new_beta

class BetaKLController(tf.keras.callbacks.Callback):
    def __init__(self, vib_layer, target_kl_per_dim=0.15, beta_adjustment_factor=1.05,
                 beta_min=1e-6, beta_max=1e2):
        super().__init__()
        self.vib_layer = vib_layer
        self.target_kl_per_dim = float(target_kl_per_dim)
        self.beta_adjustment_factor = float(beta_adjustment_factor)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

        self._kl_vals = []
        self.kl_trace = []
        self.beta_trace = []

    def on_train_batch_end(self, batch, logs=None):
        self._kl_vals.append(float(self.vib_layer.kl_per_dim_last.numpy()))

    def on_epoch_end(self, epoch, logs=None):
        kl_epoch = float(np.mean(self._kl_vals)) if self._kl_vals else float("nan")
        self._kl_vals.clear()

        if np.isfinite(kl_epoch):
            adjust_beta(
                self.vib_layer,
                kl_epoch,
                target_kl_per_dim=self.target_kl_per_dim,
                beta_adjustment_factor=self.beta_adjustment_factor,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
            )

        beta_now = float(self.vib_layer.beta.numpy())

        self.kl_trace.append(kl_epoch)
        self.beta_trace.append(beta_now)

        if logs is not None:
            logs["kl_per_dim_epoch"] = kl_epoch
            logs["beta_epoch"] = beta_now


def predict_mc_mean_std(model, X_comp, X_load_scaled, S=50, batch_size=512, seed=0, return_samples=True):
    X_comp = np.asarray(X_comp, dtype=np.float32)
    X_load_scaled = np.asarray(X_load_scaled, dtype=np.float32)

    set_mc_dropout(model, True)

    ds = tf.data.Dataset.from_tensor_slices((X_comp, X_load_scaled)).batch(batch_size)

    samples = []
    for s in range(S):
        tf.random.set_seed(int(seed) + int(s))

        preds = []
        for xb, xl in ds:
            out = model([xb, xl], training=False)
            yb = out["Output_Layer"]
            preds.append(tf.reshape(yb, (-1,)).numpy().astype(np.float32))

        samples.append(np.concatenate(preds, axis=0))

    samples = np.stack(samples, axis=0)
    mu = samples.mean(axis=0).astype(np.float32)
    sig = samples.std(axis=0).astype(np.float32)

    if return_samples:
        return mu, sig, samples
    return mu, sig


def mc_mean_attention_scores(
    model: tf.keras.Model,
    X_comp: np.ndarray,
    X_load_scaled: np.ndarray,
    mc_samples: int = 200,
    batch_size: int = 512,
    seed: int = 0,
):
    X_comp = np.asarray(X_comp, dtype=np.float32)
    X_load_scaled = np.asarray(X_load_scaled, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X_comp, X_load_scaled)).batch(batch_size)

    prev_states = [lyr.mc_active for lyr in model.layers if isinstance(lyr, MCDropout)]
    set_mc_dropout(model, True)

    comp_means = []
    load_means = []

    for s in range(int(mc_samples)):
        tf.random.set_seed(int(seed) + int(s))

        comp_batches = []
        load_batches = []

        for xb, xl in ds:
            outs = model([xb, xl], training=False)
            comp_att = outs["Comp_Attention"]
            load_att = outs["Load_Attention"]

            comp_batches.append(comp_att.numpy().astype(np.float32))
            load_batches.append(load_att.numpy().astype(np.float32))

        comp_all = np.concatenate(comp_batches, axis=0)
        load_all = np.concatenate(load_batches, axis=0)

        comp_means.append(comp_all.mean(axis=0))
        load_means.append(load_all.mean(axis=0))

    comp_means = np.stack(comp_means, axis=0)
    load_means = np.stack(load_means, axis=0)

    comp_att_mean = comp_means.mean(axis=0).astype(np.float32)
    load_att_mean = load_means.mean(axis=0).astype(np.float32)
    comp_att_std  = comp_means.std(axis=0).astype(np.float32)
    load_att_std  = load_means.std(axis=0).astype(np.float32)

    i = 0
    for lyr in model.layers:
        if isinstance(lyr, MCDropout):
            lyr.mc_active = bool(prev_states[i])
            i += 1

    return comp_att_mean, load_att_mean, comp_att_std, load_att_std


# ============================
# Model Training
# ============================
epochs     = 1000
k_outer    = 5
n_trials   = 50
batch_size = 32

def ensure_2d_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1, 1) if x.ndim == 1 else x

def normalize_simplex_np(x, eps=1e-8):
    x = np.clip(x, 0.0, None)
    s = x.sum(axis=1, keepdims=True)
    s[s < eps] = 1.0
    return x / s

def make_sample_weight_dict(w: np.ndarray) -> dict:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    ones = np.ones_like(w, dtype=np.float32)
    return {
        "Output_Layer": w,
        "Comp_Recon": w,
        "Comp_Attention": ones,
        "Load_Attention": ones,
    }

X_comp_train_full      = np.asarray(X_comp_train_fit, dtype=np.float32)
X_load_train_full_raw  = ensure_2d_col(X_load_train_fit_raw)
y_train_full           = np.asarray(y_train_fit, dtype=np.float32).reshape(-1)

w_train_full = np.asarray(w_all[train_fit_idx], dtype=np.float32).reshape(-1)

cluster_labels_train = np.asarray(cluster_labels[train_fit_idx])
skf = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=seed)

fold_results = {
    "best_latent_dim": [],
    "best_dropout_rate": [],

    "train_losses": [],
    "train_output_losses": [],
    "val_losses": [],
    "val_output_losses": [],
    "fold_best_val_outloss": [],

    "r2_scores": [],
    "mse_scores": [],

    "beta_traces": [],
    "attention_scores": [],

    "latent_train": [],
    "latent_val": [],
    "y_train": [],
    "y_val": [],

    "load_scaler_mean": [],
    "load_scaler_scale": [],

    "kl_per_dim_traces": [],
}

inner_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

for fold, (tr_i, va_i) in enumerate(skf.split(X_comp_train_full, cluster_labels_train), start=1):
    print(f"\nFold {fold}/{k_outer}")

    tr_i = np.asarray(tr_i)
    va_i = np.asarray(va_i)

    X_comp_tr = X_comp_train_full[tr_i].astype(np.float32)
    X_comp_va = X_comp_train_full[va_i].astype(np.float32)

    X_load_tr_raw = ensure_2d_col(X_load_train_full_raw[tr_i])
    X_load_va_raw = ensure_2d_col(X_load_train_full_raw[va_i])

    y_tr_full = y_train_full[tr_i].astype(np.float32).reshape(-1)
    y_va_full = y_train_full[va_i].astype(np.float32).reshape(-1)

    w_tr_full = w_train_full[tr_i].astype(np.float32).reshape(-1)
    w_va_full = w_train_full[va_i].astype(np.float32).reshape(-1)

    y_mu_fold = float(y_tr_full.mean())
    y_sd_fold = float(y_tr_full.std() + 1e-8)

    y_tr_full_s = ((y_tr_full - y_mu_fold) / y_sd_fold).astype(np.float32)
    y_va_full_s = ((y_va_full - y_mu_fold) / y_sd_fold).astype(np.float32)

    scaler_load_fold = StandardScaler()
    scaler_load_fold.fit(X_load_tr_raw)

    X_load_tr = scaler_load_fold.transform(X_load_tr_raw).astype(np.float32)
    X_load_va = scaler_load_fold.transform(X_load_va_raw).astype(np.float32)

    fold_results["load_scaler_mean"].append(scaler_load_fold.mean_.copy())
    fold_results["load_scaler_scale"].append(scaler_load_fold.scale_.copy())

    comp_target_tr_full = normalize_simplex_np(X_comp_tr)
    comp_target_va_full = normalize_simplex_np(X_comp_va)

    load_target_tr_full = X_load_tr
    load_target_va_full = X_load_va

    # Dummy targets for attention outputs
    dummy_comp_tr_full = np.zeros((X_comp_tr.shape[0], X_comp_tr.shape[1]), dtype=np.float32)
    dummy_load_tr_full = np.zeros((X_load_tr.shape[0], X_load_tr.shape[1]), dtype=np.float32)
    dummy_comp_va_full = np.zeros((X_comp_va.shape[0], X_comp_va.shape[1]), dtype=np.float32)
    dummy_load_va_full = np.zeros((X_load_va.shape[0], X_load_va.shape[1]), dtype=np.float32)

    cluster_labels_fold_tr = cluster_labels_train[tr_i]
    tr_in_idx, va_in_idx = next(inner_split.split(X_comp_tr, cluster_labels_fold_tr))

    tr_in_idx = np.asarray(tr_in_idx)
    va_in_idx = np.asarray(va_in_idx)

    Xc_in_tr = X_comp_tr[tr_in_idx].astype(np.float32)
    Xl_in_tr = X_load_tr[tr_in_idx].astype(np.float32)
    y_in_tr  = y_tr_full[tr_in_idx].astype(np.float32).reshape(-1)

    Xc_in_va = X_comp_tr[va_in_idx].astype(np.float32)
    Xl_in_va = X_load_tr[va_in_idx].astype(np.float32)
    y_in_va  = y_tr_full[va_in_idx].astype(np.float32).reshape(-1)

    w_in_tr = w_tr_full[tr_in_idx].astype(np.float32).reshape(-1)
    w_in_va = w_tr_full[va_in_idx].astype(np.float32).reshape(-1)

    y_in_tr_s = ((y_in_tr - y_mu_fold) / y_sd_fold).astype(np.float32)
    y_in_va_s = ((y_in_va - y_mu_fold) / y_sd_fold).astype(np.float32)

    dummy_comp_in_tr = np.zeros((Xc_in_tr.shape[0], Xc_in_tr.shape[1]), dtype=np.float32)
    dummy_load_in_tr = np.zeros((Xl_in_tr.shape[0], Xl_in_tr.shape[1]), dtype=np.float32)
    dummy_comp_in_va = np.zeros((Xc_in_va.shape[0], Xc_in_va.shape[1]), dtype=np.float32)
    dummy_load_in_va = np.zeros((Xl_in_va.shape[0], Xl_in_va.shape[1]), dtype=np.float32)

    comp_target_in_tr = normalize_simplex_np(Xc_in_tr)
    comp_target_in_va = normalize_simplex_np(Xc_in_va)

    load_target_in_tr = Xl_in_tr
    load_target_in_va = Xl_in_va

    # Optuna objective (VIBANN Architechture Hyperparameters tuning)
    def objective(trial):
        tf.keras.backend.clear_session()

        latent_dim   = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

        model = build_vib_attention_model(
            input_shape_comp=Xc_in_tr.shape[1],
            input_shape_load=Xl_in_tr.shape[1],
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )

        set_mc_dropout(model, False)
        n_mc = sum(1 for lyr in model.layers if lyr.__class__.__name__ == "MCDropout")
        if n_mc == 0:
            def count_mc(m):
                c = 0
                for l in m.layers:
                    if l.__class__.__name__ == "MCDropout":
                        c += 1
                    if isinstance(l, tf.keras.Model):
                        c += count_mc(l)
                return c
            n_mc = count_mc(model)
        if n_mc == 0:
            raise RuntimeError("No MCDropout layers found. MC-dropout toggle is irrelevant.")

        vib_layer_local = next(layer for layer in model.layers if isinstance(layer, VIBLayer))

        model.compile(
            optimizer="adam",
            loss={
                "Output_Layer": tf.keras.losses.MeanSquaredError(),
                "Comp_Attention": lambda y_true, y_pred: 0.0,
                "Load_Attention": lambda y_true, y_pred: 0.0,
                "Comp_Recon": tf.keras.losses.KLDivergence(),
            },
            loss_weights={
                "Output_Layer": 1.0,
                "Comp_Attention": 0.0,
                "Load_Attention": 0.0,
                "Comp_Recon": 0.5,
            },
        )

        es = EarlyStopping(
            monitor="val_Output_Layer_loss",
            patience=100,
            restore_best_weights=True,
            mode="min",
        )

        beta_controller_local = BetaKLController(
            vib_layer_local,
            target_kl_per_dim=0.15,
            beta_adjustment_factor=1.05,
        )

        y_train_dict = {
            "Output_Layer": y_in_tr_s,
            "Comp_Attention": dummy_comp_in_tr,
            "Load_Attention": dummy_load_in_tr,
            "Comp_Recon": comp_target_in_tr,
        }
        y_val_dict = {
            "Output_Layer": y_in_va_s,
            "Comp_Attention": dummy_comp_in_va,
            "Load_Attention": dummy_load_in_va,
            "Comp_Recon": comp_target_in_va,
        }

        hist = model.fit(
            [Xc_in_tr, Xl_in_tr],
            y_train_dict,
            sample_weight=make_sample_weight_dict(w_in_tr),
            validation_data=(
                [Xc_in_va, Xl_in_va],
                y_val_dict,
                make_sample_weight_dict(w_in_va),
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, beta_controller_local],
            verbose=0,
        )

        return float(np.min(hist.history["val_Output_Layer_loss"]))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_latent_dim_fold   = int(study.best_trial.params["latent_dim"])
    best_dropout_rate_fold = float(study.best_trial.params["dropout_rate"])

    fold_results["best_latent_dim"].append(best_latent_dim_fold)
    fold_results["best_dropout_rate"].append(best_dropout_rate_fold)

    print(f"Fold {fold}: best_latent_dim={best_latent_dim_fold}, best_dropout={best_dropout_rate_fold:.3f}")

    # ------------------------------------------------
    # Train fold-final model ON FULL fold-train
    # ------------------------------------------------
    tf.keras.backend.clear_session()

    vib_attention_model = build_vib_attention_model(
        input_shape_comp=X_comp_tr.shape[1],
        input_shape_load=X_load_tr.shape[1],
        latent_dim=best_latent_dim_fold,
        dropout_rate=best_dropout_rate_fold,
    )

    set_mc_dropout(vib_attention_model, False)

    vib_layer = next(layer for layer in vib_attention_model.layers if isinstance(layer, VIBLayer))

    vib_attention_model.compile(
        optimizer="adam",
        loss={
            "Output_Layer": tf.keras.losses.MeanSquaredError(),
            "Comp_Attention": lambda y_true, y_pred: 0.0,
            "Load_Attention": lambda y_true, y_pred: 0.0,
            "Comp_Recon": tf.keras.losses.KLDivergence(),
        },
        loss_weights={
            "Output_Layer": 1.0,
            "Comp_Attention": 0.0,
            "Load_Attention": 0.0,
            "Comp_Recon": 0.5,
        },
    )

    attention_logger = AttentionScoreLogger(
        X_comp_ref=X_comp_tr,
        X_load_ref=X_load_tr,
        comp_feature_count=X_comp_tr.shape[1],
        load_feature_count=X_load_tr.shape[1],
        fold_id=fold,
        output_dir="attention_logs",
        output_prefix="attention_scores",
        sample_size=None,
        seed=seed,
    )

    beta_controller = BetaKLController(
        vib_layer,
        target_kl_per_dim=0.15,
        beta_adjustment_factor=1.05,
    )

    es_fold = EarlyStopping(
        monitor="val_Output_Layer_loss",
        patience=100,
        restore_best_weights=True,
        mode="min",
    )

    y_tr_dict = {
        "Output_Layer": y_tr_full_s,
        "Comp_Attention": dummy_comp_tr_full,
        "Load_Attention": dummy_load_tr_full,
        "Comp_Recon": comp_target_tr_full,
    }
    y_va_dict = {
        "Output_Layer": y_va_full_s,
        "Comp_Attention": dummy_comp_va_full,
        "Load_Attention": dummy_load_va_full,
        "Comp_Recon": comp_target_va_full,
    }

    hist_fold = vib_attention_model.fit(
        [X_comp_tr, X_load_tr],
        y_tr_dict,
        sample_weight=make_sample_weight_dict(w_tr_full),
        validation_data=(
            [X_comp_va, X_load_va],
            y_va_dict,
            make_sample_weight_dict(w_va_full),
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[attention_logger, beta_controller, es_fold],
        verbose=0,
    )

    train_losses = [float(x) for x in hist_fold.history.get("loss", [])]
    train_outloss = [float(x) for x in hist_fold.history.get("Output_Layer_loss", [])]
    val_losses   = [float(x) for x in hist_fold.history.get("val_loss", [])]
    val_outloss  = [float(x) for x in hist_fold.history.get("val_Output_Layer_loss", [])]

    fold_best = float(np.min(val_outloss)) if len(val_outloss) else float("inf")
    fold_results["fold_best_val_outloss"].append(fold_best)

    beta_values = beta_controller.beta_trace.copy()
    kl_values   = beta_controller.kl_trace.copy()

    # Fold evaluation (MC mean in scaled space -> invert to HV for metrics)
    set_mc_dropout(vib_attention_model, True)
    y_pred_mu_s, y_pred_sig_s, _ = predict_mc_mean_std(
        vib_attention_model, X_comp_va, X_load_va, S=50, batch_size=512, seed=seed
    )
    set_mc_dropout(vib_attention_model, False)

    y_pred_mu_hv = (y_pred_mu_s * y_sd_fold + y_mu_fold).astype(np.float32)
    r2  = r2_score(y_va_full, y_pred_mu_hv)
    mse = mean_squared_error(y_va_full, y_pred_mu_hv)

    # Deterministic latents
    set_mc_dropout(vib_attention_model, False)

    latent_extractor_fold = tf.keras.Model(
        inputs=vib_attention_model.input,
        outputs=vib_attention_model.get_layer("vib_layer").output,
        name=f"latent_extractor_fold_{fold}",
    )

    latent_vectors_tr = latent_extractor_fold.predict([X_comp_tr, X_load_tr], verbose=0).astype(np.float32)
    latent_vectors_va = latent_extractor_fold.predict([X_comp_va, X_load_va], verbose=0).astype(np.float32)

    # Store fold outputs
    fold_results["train_losses"].append(train_losses)
    fold_results["train_output_losses"].append(train_outloss)
    fold_results["val_losses"].append(val_losses)
    fold_results["val_output_losses"].append(val_outloss)
    fold_results["beta_traces"].append(beta_values)
    fold_results["kl_per_dim_traces"].append(kl_values)
    fold_results["r2_scores"].append(float(r2))
    fold_results["mse_scores"].append(float(mse))
    fold_results["attention_scores"].append(attention_logger.attention_scores.copy())

    fold_results["latent_train"].append(latent_vectors_tr)
    fold_results["latent_val"].append(latent_vectors_va)
    fold_results["y_train"].append(y_tr_full.copy())
    fold_results["y_val"].append(y_va_full.copy())

    vib_attention_model.save(f"vib_model_fold_{fold}.h5")
    print(f"Fold {fold} R²: {r2:.4f}, MSE: {mse:.2f}, best_val_outloss={fold_best:.6f}")


# ============================
# FINAL deployable model
# ============================
best_fold_idx = int(np.argmin(np.asarray(fold_results["fold_best_val_outloss"], dtype=np.float64)))
final_latent_dim = int(fold_results["best_latent_dim"][best_fold_idx])
final_dropout    = float(fold_results["best_dropout_rate"][best_fold_idx])
final_dropout = float(np.clip(final_dropout, 0.05, 0.30))

print(f"\n[FINAL] selected from best fold {best_fold_idx+1}: latent_dim={final_latent_dim}, dropout={final_dropout:.3f}")

scaler_load_final = StandardScaler()
scaler_load_final.fit(ensure_2d_col(X_load_train_fit_raw))

X_load_train_fit = scaler_load_final.transform(ensure_2d_col(X_load_train_fit_raw)).astype(np.float32)
X_load_cal       = scaler_load_final.transform(ensure_2d_col(X_load_cal_raw)).astype(np.float32)
X_load_test      = scaler_load_final.transform(ensure_2d_col(X_load_test_raw)).astype(np.float32)

y_train_fit_arr = np.asarray(y_train_fit, dtype=np.float32).reshape(-1)
y_cal_arr       = np.asarray(y_cal, dtype=np.float32).reshape(-1)

y_mu_final = float(y_train_fit_arr.mean())
y_sd_final = float(y_train_fit_arr.std() + 1e-8)

y_train_fit_s = ((y_train_fit_arr - y_mu_final) / y_sd_final).astype(np.float32)
y_cal_s       = ((y_cal_arr       - y_mu_final) / y_sd_final).astype(np.float32)

comp_target_train_fit = normalize_simplex_np(np.asarray(X_comp_train_fit, dtype=np.float32))
comp_target_cal       = normalize_simplex_np(np.asarray(X_comp_cal, dtype=np.float32))

dummy_comp_train_fit = np.zeros((X_comp_train_fit.shape[0], X_comp_train_fit.shape[1]), dtype=np.float32)
dummy_load_train_fit = np.zeros((X_load_train_fit.shape[0], X_load_train_fit.shape[1]), dtype=np.float32)

dummy_comp_cal = np.zeros((X_comp_cal.shape[0], X_comp_cal.shape[1]), dtype=np.float32)
dummy_load_cal = np.zeros((X_load_cal.shape[0], X_load_cal.shape[1]), dtype=np.float32)

tf.keras.backend.clear_session()

final_model = build_vib_attention_model(
    input_shape_comp=X_comp_train_fit.shape[1],
    input_shape_load=X_load_train_fit.shape[1],
    latent_dim=final_latent_dim,
    dropout_rate=final_dropout,
)

set_mc_dropout(final_model, False)

final_vib_layer = next(layer for layer in final_model.layers if isinstance(layer, VIBLayer))

final_model.compile(
    optimizer="adam",
    loss={
        "Output_Layer": tf.keras.losses.MeanSquaredError(),
        "Comp_Attention": lambda y_true, y_pred: 0.0,
        "Load_Attention": lambda y_true, y_pred: 0.0,
        "Comp_Recon": tf.keras.losses.KLDivergence(),
    },
    loss_weights={
        "Output_Layer": 1.0,
        "Comp_Attention": 0.0,
        "Load_Attention": 0.0,
        "Comp_Recon": 0.5,
    },
)

es_final = EarlyStopping(
    monitor="val_Output_Layer_loss",
    patience=100,
    restore_best_weights=True,
    mode="min",
)

beta_controller_final = BetaKLController(
    final_vib_layer,
    target_kl_per_dim=0.15,
    beta_adjustment_factor=1.05,
)

y_train_final = {
    "Output_Layer": y_train_fit_s.reshape(-1),
    "Comp_Attention": dummy_comp_train_fit,
    "Load_Attention": dummy_load_train_fit,
    "Comp_Recon": comp_target_train_fit,
}
y_cal_final = {
    "Output_Layer": y_cal_s.reshape(-1),
    "Comp_Attention": dummy_comp_cal,
    "Load_Attention": dummy_load_cal,
    "Comp_Recon": comp_target_cal,
}

final_model.fit(
    [np.asarray(X_comp_train_fit, dtype=np.float32), X_load_train_fit],
    y_train_final,
    sample_weight=make_sample_weight_dict(w_train_fit),
    validation_data=(
        [np.asarray(X_comp_cal, dtype=np.float32), X_load_cal],
        y_cal_final,
        make_sample_weight_dict(w_cal),
    ),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es_final, beta_controller_final],
    verbose=1,
)

# Save models
final_model.save("vib_attention_FINAL_model.keras")
np.savez("load_scaler_FINAL.npz", mean=scaler_load_final.mean_, scale=scaler_load_final.scale_)
np.savez("y_scaler_FINAL.npz", mean=y_mu_final, scale=y_sd_final)
print("Saved: vib_attention_FINAL_model.keras, load_scaler_FINAL.npz, y_scaler_FINAL.npz")


# ========================
# Visualizatons
# ========================
plt.figure(figsize=(10, 8))
for i in range(k_outer):
    tr = fold_results["train_output_losses"][i]
    va = fold_results["val_output_losses"][i]
    plt.plot(range(1, len(tr) + 1), tr, label=f"Train Fold {i+1}")
    plt.plot(range(1, len(va) + 1), va, linestyle="--", label=f"Val Fold {i+1}")

plt.xlabel("Epoch")
plt.ylabel("Hardness loss (MSE on y_scaled)")
plt.title("Hardness Training and Validation Loss per Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot Training and Validation Loss for best fold
best_fold_idx = int(np.argmax(fold_results["r2_scores"]))
train_losses = fold_results["train_output_losses"][best_fold_idx]
val_losses   = fold_results["val_output_losses"][best_fold_idx]

epochs_vec = np.arange(1, min(len(train_losses), len(val_losses)) + 1)
train_losses = train_losses[:len(epochs_vec)]
val_losses   = val_losses[:len(epochs_vec)]

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(epochs_vec, train_losses, label="Training Loss", linewidth=3, marker=".", markersize=2)
ax.plot(epochs_vec, val_losses,   label="Validation Loss", linewidth=3, marker=".", markersize=2)

ax.set_xlabel("Epochs", fontsize=22, weight="bold", color="black")
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=22, weight="bold", color="black")

legend = ax.legend(fontsize=14, loc="upper right", fancybox=True, shadow=True)
legend.get_frame().set_facecolor("white")

ax.tick_params(axis="both", which="major", labelsize=15, color="black")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("Model_Training_and_Validation_Loss_Over_Epochs.jpeg", dpi=600, format="jpeg")
plt.show()


# beta values plot
best_fold_idx = int(np.argmax(fold_results["r2_scores"]))
beta_values = np.asarray(fold_results["beta_traces"][best_fold_idx], dtype=np.float64)

plt.figure(figsize=(10, 8))
plt.plot(np.arange(1, len(beta_values)+1), beta_values,
         label=r'$\beta$ (KL controller)', linewidth=3, marker='.', markersize=3)
plt.xlabel('Epochs', fontsize=22, labelpad=10, weight='bold', color='black')
plt.ylabel(r'$\beta$ Value', fontsize=22, labelpad=10, weight='bold', color='black')
plt.grid(False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig("beta_value_dynamics_during_training.jpeg", dpi=600, format='jpeg')
plt.show()


# Validation loss over folds
avg_val_hv = [float(np.mean(x)) for x in fold_results["val_output_losses"]]
std_val_hv = [float(np.std(x))  for x in fold_results["val_output_losses"]]

fold_ids = np.arange(1, len(avg_val_hv) + 1)

plt.figure(figsize=(10, 8))
plt.plot(fold_ids, avg_val_hv, "o--", label="Val Output_Layer loss (HV)")
plt.fill_between(
    fold_ids,
    np.array(avg_val_hv) - np.array(std_val_hv),
    np.array(avg_val_hv) + np.array(std_val_hv),
    alpha=0.3
)

plt.xlabel("Fold", fontsize=14)
plt.ylabel("Mean val HV loss across epochs", fontsize=14)
plt.xticks(fold_ids)
plt.legend(frameon=True, fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig("Val_HV_Loss_vs_Fold.jpeg", dpi=600, format="jpeg")
plt.show()


# Plot RMSE vs folds
rmse = np.sqrt(np.array(fold_results["mse_scores"], dtype=np.float64))
fold_ids = np.arange(1, len(rmse) + 1)

rmse_mean = float(np.mean(rmse))
rmse_std  = float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0

plt.figure(figsize=(10, 8))
plt.plot(fold_ids, rmse, "o--", label="Val RMSE (MC-mean)")
plt.axhline(rmse_mean, linestyle="--", linewidth=2, label=f"Mean={rmse_mean:.2f}")
if len(rmse) > 1:
    plt.fill_between(
        fold_ids,
        rmse_mean - rmse_std,
        rmse_mean + rmse_std,
        alpha=0.2,
        label=f"±1σ={rmse_std:.2f}"
    )

plt.xlabel("Fold", fontsize=14)
plt.ylabel("RMSE (HV)", fontsize=14)
plt.xticks(fold_ids)
plt.legend(frameon=True, fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig("Val_RMSE_vs_Fold_MCmean.jpeg", dpi=600, format="jpeg")
plt.show()

print(f"CV RMSE: {rmse_mean:.3f} ± {rmse_std:.3f} (HV)")


# =====================
# FINAL MODEL SAVE
# =====================
set_mc_dropout(final_model, False)

# Save full model (architecture + weights)
MODEL_PATH = "vib_attention_FINAL_model.keras"
final_model.save(MODEL_PATH)
print(f"Saved full model: {MODEL_PATH}")

# Save weights separately
WEIGHTS_PATH = "vib_attention_FINAL_model.weights.h5"
final_model.save_weights(WEIGHTS_PATH)
print(f"Saved model weights: {WEIGHTS_PATH}")

# Save load scaler (mean/scale)
SCALER_PATH = "load_scaler_FINAL.npz"
np.savez(
    SCALER_PATH,
    mean=np.asarray(scaler_load_final.mean_, dtype=np.float64).reshape(-1),
    scale=np.asarray(scaler_load_final.scale_, dtype=np.float64).reshape(-1),
)
print(f"Saved load scaler: {SCALER_PATH}")

META_PATH = "vib_attention_FINAL_metadata.json"
meta = {
    "model_path": MODEL_PATH,
    "weights_path": WEIGHTS_PATH,
    "scaler_path": SCALER_PATH,
    "composition_cols": list(composition_cols),
    "final_latent_dim": int(final_latent_dim),
    "final_dropout": float(final_dropout),
    "input_shape_comp": int(X_comp_train_fit.shape[1]),
    "input_shape_load": int(X_load_train_fit.shape[1]),
    "notes": {
        "mc_dropout_default": "OFF (set_mc_dropout(model, False) before saving)",
        "inference_requires_scaled_load": True,
        "load_scaler": "StandardScaler(mean_, scale_) saved in NPZ",
    },
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved metadata: {META_PATH}")


# =======================================
# Monte-Carlo Dropout sampling helpers
# =======================================
def _extract_output_layer(out):
    if isinstance(out, dict):
        return out["Output_Layer"]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

def mc_dropout_samples(model, X_comp, X_load_scaled, num_samples=200, batch_size=512, seed=0):
    X_comp = np.asarray(X_comp, dtype=np.float32)
    X_load_scaled = np.asarray(X_load_scaled, dtype=np.float32)
    if X_load_scaled.ndim == 1:
        X_load_scaled = X_load_scaled.reshape(-1, 1)

    n = X_comp.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((X_comp, X_load_scaled)).batch(batch_size)

    rng = np.random.default_rng(seed)
    samples = np.empty((int(num_samples), n), dtype=np.float32)

    set_mc_dropout(model, True)
    try:
        for s in range(int(num_samples)):
            tf.keras.utils.set_random_seed(int(rng.integers(0, 2**31 - 1)))

            preds = []
            for xb, xl in ds:
                out = model([xb, xl], training=False)
                yb = _extract_output_layer(out)
                preds.append(tf.reshape(yb, (-1,)).numpy().astype(np.float32))

            samples[s, :] = np.concatenate(preds, axis=0)
    finally:
        set_mc_dropout(model, False)

    return samples


def bootstrap_rmse_ci(y_true, y_pred_samples, n_bootstrap=1000, ci=0.95, seed=1):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_samples = np.asarray(y_pred_samples, dtype=np.float64)

    if y_pred_samples.ndim != 2:
        raise ValueError("y_pred_samples must be 2D with shape (S, n).")
    if y_pred_samples.shape[1] != y_true.shape[0]:
        raise ValueError("y_pred_samples second dimension must match len(y_true).")

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]

    # Point estimate using MC mean predictor
    y_pred_mean_full = y_pred_samples.mean(axis=0)
    rmse_point = float(np.sqrt(mean_squared_error(y_true, y_pred_mean_full)))

    rmse_list = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_b = y_true[idx]
        y_pred_b = y_pred_samples[:, idx].mean(axis=0)
        rmse_list[b] = np.sqrt(mean_squared_error(y_true_b, y_pred_b))

    alpha = 1.0 - ci
    lower = float(np.percentile(rmse_list, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(rmse_list, 100.0 * (1.0 - alpha / 2.0)))
    rmse_boot_mean = float(rmse_list.mean())

    return rmse_point, rmse_boot_mean, lower, upper, rmse_list


# ==================================
# MC Dropout UQ (FINAL model)
# ==================================
S = 100

X_load_test_scaled = np.asarray(X_load_test, dtype=np.float32).reshape(-1, 1)

y_pred_samples_s = mc_dropout_samples(
    final_model,
    X_comp_test,
    X_load_test_scaled,
    num_samples=S,
    batch_size=512,
    seed=seed
)

y_test_hv = np.asarray(y_test, dtype=np.float32).reshape(-1)
y_pred_samples_hv = (y_pred_samples_s * y_sd_final + y_mu_final).astype(np.float32)
y_pred_mean_hv = y_pred_samples_hv.mean(axis=0).astype(np.float32)

mu_pred_hv = y_pred_samples_hv.mean(axis=0).astype(np.float32)
resid = (y_test_hv - mu_pred_hv).astype(np.float32)

mad = np.median(np.abs(resid - np.median(resid))).astype(np.float32)
sigma_noise = float(1.4826 * mad)

if not np.isfinite(sigma_noise) or sigma_noise <= 1e-6:
    sigma_noise = float(np.sqrt(np.mean(resid**2)))

print(f"[UQ DIAG] sigma_noise (HV) = {sigma_noise:.3f}")

rng = np.random.default_rng(int(seed) + 2025)
y_pred_samples_hv = (y_pred_samples_hv + rng.normal(0.0, sigma_noise, size=y_pred_samples_hv.shape)).astype(np.float32)

ALPHA = 0.90
q_lo = (1.0 - ALPHA) / 2.0
q_hi = 1.0 - q_lo

y_lo_hv = np.quantile(y_pred_samples_hv, q_lo, axis=0).astype(np.float32)
y_hi_hv = np.quantile(y_pred_samples_hv, q_hi, axis=0).astype(np.float32)

if y_test_hv.shape[0] != y_pred_mean_hv.shape[0]:
    raise ValueError(
        f"Length mismatch: y_test has {y_test_hv.shape[0]} but predictions have {y_pred_mean_hv.shape[0]}."
    )

rmse_point, rmse_boot_mean, rmse_lower, rmse_upper, rmse_list = bootstrap_rmse_ci(
    y_true=y_test_hv,
    y_pred_samples=y_pred_samples_hv,
    n_bootstrap=1000,
    ci=0.95,
    seed=seed
)

print(f"Test RMSE (MC-mean predictor, HV): {rmse_point:.4f}")
print(f"Bootstrap RMSE mean: {rmse_boot_mean:.4f} (95% CI: [{rmse_lower:.4f}, {rmse_upper:.4f}])")


# ====================
# Visualizations
# ====================
# Predicted vs Actual Hardness with MC Prediction Intervals (HV space)
y_test_arr  = np.asarray(y_test_hv, dtype=np.float32).reshape(-1)
y_pred_mean = np.asarray(y_pred_mean_hv, dtype=np.float32).reshape(-1)
y_lo        = np.asarray(y_lo_hv, dtype=np.float32).reshape(-1)
y_hi        = np.asarray(y_hi_hv, dtype=np.float32).reshape(-1)

if np.any(y_lo > y_pred_mean) or np.any(y_pred_mean > y_hi):
    raise ValueError("Invalid PI ordering: expected y_lo <= y_pred_mean <= y_hi for all points.")

yerr_lower = y_pred_mean - y_lo
yerr_upper = y_hi - y_pred_mean

minv = float(np.min([y_test_arr.min(), y_pred_mean.min(), y_lo.min()]))
maxv = float(np.max([y_test_arr.max(), y_pred_mean.max(), y_hi.max()]))

pad  = 0.05 * (maxv - minv + 1e-6)
lo_ax = minv - pad
hi_ax = maxv + pad

fig, ax = plt.subplots(figsize=(10, 8))
ax.errorbar(
    y_test_arr, y_pred_mean,
    yerr=np.vstack([yerr_lower, yerr_upper]),
    fmt='o',
    ecolor='red',
    alpha=0.8,
    capsize=4,
    markeredgewidth=1,
    markersize=8,
    label=f"MC PI ({int(ALPHA*100)}%)"
)
ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], 'k--', lw=2)

ax.set_xlim(lo_ax, hi_ax)
ax.set_ylim(lo_ax, hi_ax)

ax.set_xlabel('Actual Hardness (HV)', fontsize=22, weight='bold')
ax.set_ylabel('Predicted Hardness (HV)', fontsize=22, weight='bold')
ax.grid(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(loc='upper left', fontsize=14)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig(f"pred_vs_actual_with_MC_PI_{int(ALPHA*100)}.jpeg", dpi=600, format='jpeg')
plt.show()


# Uncertainty distribution: use PI width or sigma
y_lo_arr = np.asarray(y_lo, dtype=np.float32).reshape(-1)
y_hi_arr = np.asarray(y_hi, dtype=np.float32).reshape(-1)

if np.any(y_hi_arr < y_lo_arr):
    raise ValueError("Invalid PI bounds: found y_hi < y_lo for some samples.")

pi_width = y_hi_arr - y_lo_arr

fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(pi_width, bins=30, alpha=0.7, density=False)

ax.set_xlabel(f'Predictive Interval Width ({int(ALPHA*100)}% PI)', fontsize=20, weight='bold')
ax.set_ylabel('Frequency', fontsize=20, weight='bold')
ax.grid(False)
ax.tick_params(axis='both', which='major', labelsize=18)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig(f"distribution_PI_width_{int(ALPHA*100)}.jpeg", dpi=600, format='jpeg')
plt.show()



# Frequency vs. RMSE
rmse_arr = np.asarray(rmse_list, dtype=np.float64).reshape(-1)

plt.figure(figsize=(10, 8))
counts, bin_edges, _ = plt.hist(rmse_arr, bins=30, alpha=0.7, density=True)

plt.axvline(rmse_mean, linestyle='--', linewidth=2, label=f"Mean RMSE = {rmse_mean:.2f}")
plt.axvline(rmse_lower, linestyle='--', linewidth=2, label=f"Lower 95% CI = {rmse_lower:.2f}")
plt.axvline(rmse_upper, linestyle='--', linewidth=2, label=f"Upper 95% CI = {rmse_upper:.2f}")

plt.xlabel("RMSE", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Bootstrap Distribution of RMSE", fontsize=14)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("bootstrap_rmse_distribution.jpeg", dpi=600, format="jpeg")
plt.show()


# ============================
# Final Metrics
# ============================
y_true = np.asarray(y_test_arr, dtype=np.float32).reshape(-1)
y_pred_mean_vec = np.asarray(y_pred_mean, dtype=np.float32).reshape(-1)
y_pred = y_pred_mean_vec

# Finite checks
if y_true.shape[0] != y_pred.shape[0]:
    raise ValueError(f"Length mismatch: y_true={y_true.shape[0]} vs y_pred={y_pred.shape[0]}")

if not np.all(np.isfinite(y_true)):
    bad = np.where(~np.isfinite(y_true))[0][:10]
    raise ValueError(f"Non-finite values in y_true at indices {bad} (showing up to 10).")

if not np.all(np.isfinite(y_pred)):
    bad = np.where(~np.isfinite(y_pred))[0][:10]
    raise ValueError(f"Non-finite values in y_pred at indices {bad} (showing up to 10).")

# Core metrics
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae  = float(mean_absolute_error(y_true, y_pred))
r2   = float(r2_score(y_true, y_pred))

# MAPE
eps = 1e-6
mask = np.abs(y_true) > eps
mape = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask])) if np.any(mask) else float("nan")

# Normalized RMSE
yrng = float(np.max(y_true) - np.min(y_true))
nrmse_range = float(rmse / (yrng + 1e-12))

# Tail performance
q = 0.85
thr = float(np.quantile(y_true, q))
tail = y_true >= thr

if np.sum(tail) >= 5:
    rmse_tail = float(np.sqrt(mean_squared_error(y_true[tail], y_pred[tail])))
    mae_tail  = float(mean_absolute_error(y_true[tail], y_pred[tail]))
else:
    rmse_tail, mae_tail = float("nan"), float("nan")

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R²:   {r2:.4f}")
print(f"Test NRMSE (range): {nrmse_range:.4f}")
print(f"Test MAPE: {mape:.4f}")
print(f"Tail metrics (y >= q{int(q*100)}={thr:.3f}, n={int(np.sum(tail))}): RMSE={rmse_tail:.4f}, MAE={mae_tail:.4f}")


# =================================
# Attention Scores: MC-averaged
# =================================
def scale_load_np(x_raw, scaler, dtype=np.float32):
    x = np.asarray(x_raw, dtype=dtype)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return scaler.transform(x).astype(dtype)

X_comp_eval = np.asarray(X_comp_test, dtype=np.float32)

X_load_eval_scaled = np.asarray(X_load_test, dtype=np.float32)
if X_load_eval_scaled.ndim == 1:
    X_load_eval_scaled = X_load_eval_scaled.reshape(-1, 1)

if X_load_eval_scaled.shape[1] != 1:
    raise ValueError(f"Expected load shape (N,1). Got {X_load_eval_scaled.shape}.")

comp_att_mean, load_att_mean, comp_att_std, load_att_std = mc_mean_attention_scores(
    model=final_model,
    X_comp=X_comp_eval,
    X_load_scaled=X_load_eval_scaled,
    mc_samples=500,
    batch_size=512,
    seed=seed
)

feature_names = list(composition_cols)
comp_att_mean = np.asarray(comp_att_mean, dtype=np.float32).reshape(-1)
comp_att_std  = np.asarray(comp_att_std,  dtype=np.float32).reshape(-1)

if comp_att_mean.shape[0] != len(feature_names):
    raise ValueError(
        f"Comp attention dim {comp_att_mean.shape[0]} != n_features {len(feature_names)}. "
        "Check FeatureWiseAttention(n_features=...) wiring."
    )

plt.figure(figsize=(14, 6))
plt.bar(feature_names, comp_att_mean, edgecolor="black", linewidth=1.0)

plt.errorbar(
    np.arange(len(feature_names)),
    comp_att_mean,
    yerr=comp_att_std,
    fmt="none",
    ecolor="black",
    elinewidth=1.0,
    capsize=2,
    alpha=0.9,
)

plt.xlabel("Composition Feature", fontsize=24, labelpad=10, color="black")
plt.ylabel("MC-Mean Attention Score", fontsize=24, labelpad=10, color="black")
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("Average_Attention_Weights_for_Composition_Features_MCmean.jpeg", dpi=600, format="jpeg")
plt.show()


# ============================
# Integrated Gradients
# ============================
required = [
    "final_model",
    "scaler_load_final",
    "composition_cols",
    "X_comp_train_fit",
    "X_load_train_fit_raw",
    "X_comp_cal",
    "X_load_cal_raw",
    "seed",
]
for name in required:
    if name not in globals():
        raise RuntimeError(f"Missing required object: {name}")

def ensure_2d_col_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"Expected (N,1). Got {x.shape}.")
    return x

def scale_load_final_np(x_raw: np.ndarray, scaler) -> np.ndarray:
    x_raw = ensure_2d_col_np(x_raw)
    return scaler.transform(x_raw).astype(np.float32)

def _model_output_to_tensor(out):
    """Unify model output into a float32 tensor."""
    if isinstance(out, dict):
        out = out["Output_Layer"]
    elif isinstance(out, (list, tuple)):
        out = out[0]
    return tf.convert_to_tensor(out, dtype=tf.float32)

def integrated_gradients(
    model: tf.keras.Model,
    baseline_inputs,
    input_inputs,
    steps: int = 200,
):
    """
    Returns IG attributions for each input tensor (comp, load).
    Robust to disconnected graphs: if gradient is None, attribution is 0 for that input.
    BN-safe: always calls model(..., training=False).
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")

    baseline_comp, baseline_load = baseline_inputs
    input_comp, input_load       = input_inputs

    baseline_comp = tf.cast(baseline_comp, tf.float32)
    baseline_load = tf.cast(baseline_load, tf.float32)
    input_comp    = tf.cast(input_comp, tf.float32)
    input_load    = tf.cast(input_load, tf.float32)

    baseline_comp = tf.reshape(baseline_comp, (1, -1))
    input_comp    = tf.reshape(input_comp, (1, -1))
    baseline_load = tf.reshape(baseline_load, (1, 1))
    input_load    = tf.reshape(input_load, (1, 1))

    alphas = tf.linspace(0.0, 1.0, steps + 1)

    acc_comp = tf.zeros_like(input_comp, dtype=tf.float32)
    acc_load = tf.zeros_like(input_load, dtype=tf.float32)

    prev_g_comp = None
    prev_g_load = None

    for a in alphas:
        comp_a = baseline_comp + a * (input_comp - baseline_comp)
        load_a = baseline_load + a * (input_load - baseline_load)

        with tf.GradientTape() as tape:
            tape.watch([comp_a, load_a])
            out = model([comp_a, load_a], training=False)
            out = _model_output_to_tensor(out)
            obj = tf.reduce_sum(out)

        g_comp, g_load = tape.gradient(obj, [comp_a, load_a])

        if g_comp is None:
            g_comp = tf.zeros_like(comp_a, dtype=tf.float32)
        if g_load is None:
            g_load = tf.zeros_like(load_a, dtype=tf.float32)

        if prev_g_comp is not None:
            acc_comp += 0.5 * (prev_g_comp + g_comp)
            acc_load += 0.5 * (prev_g_load + g_load)

        prev_g_comp, prev_g_load = g_comp, g_load

    avg_g_comp = acc_comp / tf.cast(steps, tf.float32)
    avg_g_load = acc_load / tf.cast(steps, tf.float32)

    ig_comp = (input_comp - baseline_comp) * avg_g_comp
    ig_load = (input_load - baseline_load) * avg_g_load
    return ig_comp, ig_load

Xc_train = np.asarray(X_comp_train_fit, dtype=np.float32)
Xl_train_raw = ensure_2d_col_np(np.asarray(X_load_train_fit_raw, dtype=np.float32))

comp_baseline_np = Xc_train.mean(axis=0, keepdims=True).astype(np.float32)
load_baseline_raw = Xl_train_raw.mean(axis=0, keepdims=True).astype(np.float32)
load_baseline_scaled = scale_load_final_np(load_baseline_raw, scaler_load_final)

sample_idx = 0
Xc_ref = np.asarray(X_comp_cal, dtype=np.float32)
Xl_ref_raw = ensure_2d_col_np(np.asarray(X_load_cal_raw, dtype=np.float32))

if sample_idx < 0 or sample_idx >= Xc_ref.shape[0]:
    raise IndexError(f"sample_idx={sample_idx} out of range for X_comp_cal with N={Xc_ref.shape[0]}")

comp_row = Xc_ref[sample_idx:sample_idx+1].astype(np.float32)
load_row_raw = Xl_ref_raw[sample_idx:sample_idx+1].astype(np.float32)
load_row_scaled = scale_load_final_np(load_row_raw, scaler_load_final)
comp_baseline_tf = tf.constant(comp_baseline_np, dtype=tf.float32)
load_baseline_tf = tf.constant(load_baseline_scaled, dtype=tf.float32)
comp_input_tf    = tf.constant(comp_row, dtype=tf.float32)
load_input_tf    = tf.constant(load_row_scaled, dtype=tf.float32)

tf.keras.utils.set_random_seed(int(seed))
ig_comp_tf, ig_load_tf = integrated_gradients(
    final_model,
    baseline_inputs=[comp_baseline_tf, load_baseline_tf],
    input_inputs=[comp_input_tf, load_input_tf],
    steps=200,
)

ig_comp = ig_comp_tf.numpy().reshape(-1).astype(np.float32)
ig_load = ig_load_tf.numpy().reshape(-1).astype(np.float32)

print(
    "IG(comp) min/max:",
    float(np.nanmin(ig_comp)),
    float(np.nanmax(ig_comp)),
    "nonfinite:",
    bool(np.any(~np.isfinite(ig_comp))),
)
print("IG(load) value:", float(ig_load[0]))

feature_names = list(composition_cols)
mx = float(np.max(np.abs(ig_comp))) + 1e-12

plt.figure(figsize=(16, 8))
plt.bar(feature_names, ig_comp, edgecolor="black", linewidth=1.0)
plt.ylim(-mx, mx)
plt.xlabel("Compositional Constituent", fontsize=18, weight="bold")
plt.ylabel("Integrated Gradients Attribution (model output y_scaled)", fontsize=18, weight="bold")
plt.xticks(range(len(ig_comp)), feature_names, rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
for spine in plt.gca().spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig("Integrated_Gradients_FINALModel_Composition_yScaled.jpeg", dpi=600, format="jpeg")
plt.show()


# ============================
# Regression UQ Calibration
# ============================
X_comp_te     = X_comp_test
X_load_te_raw = X_load_test_raw
y_te_full     = y_test_arr

for name in ["final_model", "scaler_load_final", "y_mu_final", "y_sd_final", "seed",
             "X_comp_te", "X_load_te_raw", "y_te_full"]:
    if name not in globals():
        raise RuntimeError(f"Missing required object: {name}")

X_comp_eval = np.asarray(X_comp_te, dtype=np.float32)
X_load_eval_raw = np.asarray(X_load_te_raw, dtype=np.float32)
y_eval_vec = np.asarray(y_te_full, dtype=np.float32).reshape(-1)

if X_load_eval_raw.ndim == 1:
    X_load_eval_raw = X_load_eval_raw.reshape(-1, 1)
if X_load_eval_raw.ndim != 2 or X_load_eval_raw.shape[1] != 1:
    raise ValueError(f"X_load_eval_raw must be (N,1). Got {X_load_eval_raw.shape}")

N = X_comp_eval.shape[0]
if (X_load_eval_raw.shape[0] != N) or (y_eval_vec.shape[0] != N):
    raise ValueError(
        f"Length mismatch: X_comp N={N}, X_load N={X_load_eval_raw.shape[0]}, y N={y_eval_vec.shape[0]}"
    )

def _iter_layers(obj):
    """Yield layers recursively for Keras/TF-Keras model-like objects."""
    if hasattr(obj, "layers"):
        for lyr in obj.layers:
            yield lyr
            if hasattr(lyr, "layers"):
                yield from _iter_layers(lyr)

def set_mc_dropout(model, active: bool) -> int:
    """
    Toggle MC-dropout layers (custom MCDropout with attribute mc_active).
    Returns number of toggled layers. Does NOT use model.submodules.
    """
    active = bool(active)
    n = 0
    for m in _iter_layers(model):
        if m.__class__.__name__ == "MCDropout" and hasattr(m, "mc_active"):
            m.mc_active = active
            n += 1
    return n

def _extract_y_scaled(out):
    """
    final_model sometimes returns a dict with keys:
      'Output_Layer', 'Comp_Attention', 'Load_Attention', ...
    Handle dict/list/tuple/tensor robustly.
    """
    if isinstance(out, dict):
        if "Output_Layer" not in out:
            raise KeyError(f"Model returned dict but missing 'Output_Layer'. Keys: {list(out.keys())}")
        return out["Output_Layer"]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

def scale_load_final_np(x_raw: np.ndarray) -> np.ndarray:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim == 1:
        x_raw = x_raw.reshape(-1, 1)
    if x_raw.ndim != 2 or x_raw.shape[1] != 1:
        raise ValueError(f"Expected load shape (N,1). Got {x_raw.shape}")
    return scaler_load_final.transform(x_raw).astype(np.float32)

def yscaled_to_hv_np(y_scaled: np.ndarray, y_mu: float, y_sd: float) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=np.float32).reshape(-1)
    return (y_scaled * float(y_sd) + float(y_mu)).astype(np.float32)

ORIGINAL_Y_IS_SCALED = False
y_eval_hv = yscaled_to_hv_np(y_eval_vec, y_mu_final, y_sd_final) if ORIGINAL_Y_IS_SCALED else y_eval_vec.astype(np.float32)
X_load_eval_scaled = scale_load_final_np(X_load_eval_raw)

def mc_predict_hv(
    model,
    Xc: np.ndarray,
    Xl_scaled: np.ndarray,
    S: int = 500,
    batch_size: int = 512,
    seed: int = 0,
) -> np.ndarray:  
    Xc = np.asarray(Xc, dtype=np.float32)
    Xl_scaled = np.asarray(Xl_scaled, dtype=np.float32)
    n = Xc.shape[0]

    ds = tf.data.Dataset.from_tensor_slices((Xc, Xl_scaled)).batch(batch_size)
    samples = np.empty((int(S), n), dtype=np.float32)

    n_toggled = set_mc_dropout(model, True)
    if n_toggled == 0:
        raise RuntimeError(
            "No MCDropout layers with attribute 'mc_active' were found. "
            "MC-dropout sampling will be deterministic. Ensure final_model uses MCDropout."
        )

    try:
        for s in range(int(S)):
            tf.keras.utils.set_random_seed(int(seed) + 10000 + s)
            preds = []
            for xb, xl in ds:
                out = model([xb, xl], training=False)
                yb = _extract_y_scaled(out)
                yb = tf.reshape(tf.cast(yb, tf.float32), (-1,)).numpy().astype(np.float32)
                preds.append(yscaled_to_hv_np(yb, y_mu_final, y_sd_final))
            samples[s, :] = np.concatenate(preds, axis=0)
    finally:
        set_mc_dropout(model, False)

    return samples

S_MC = 500
y_pred_samples_hv = mc_predict_hv(
    final_model, X_comp_eval, X_load_eval_scaled, S=S_MC, batch_size=512, seed=int(seed)
)

std_per_point = np.std(y_pred_samples_hv, axis=0)
mc_std_mean = float(np.mean(std_per_point))
mc_std_p50 = float(np.median(std_per_point))
print(f"[UQ DIAG] MC std: mean={mc_std_mean:.6f} HV, median={mc_std_p50:.6f} HV")

if mc_std_mean < 1e-3:
    raise RuntimeError(
        "MC predictive samples are essentially deterministic (std ~ 0). "
        "Fix upstream: ensure final_model uses MCDropout and set_mc_dropout toggles mc_active."
    )

# PIT histogram
y_eval = y_eval_hv.reshape(-1)
S = y_pred_samples_hv.shape[0]

pit = (np.sum(y_pred_samples_hv < y_eval[None, :], axis=0) +
       0.5 * np.sum(y_pred_samples_hv == y_eval[None, :], axis=0)) / float(S)
pit = pit.astype(np.float32)

plt.figure(figsize=(10, 7))
plt.hist(pit, bins=20, edgecolor="black")
plt.xlabel("PIT value", fontsize=16, weight="bold")
plt.ylabel("Count", fontsize=16, weight="bold")
plt.tight_layout()
plt.savefig("uq_pit_histogram_HV_FINAL.jpeg", dpi=600, format="jpeg")
plt.show()

# Coverage vs nominal + sharpness
confidence_levels = np.linspace(0.05, 0.95, 10).astype(np.float32)
coverage_probabilities = []
interval_widths = []

for c in confidence_levels:
    lower_q = float((1.0 - c) / 2.0)
    upper_q = float(1.0 - lower_q)

    lower_bound = np.quantile(y_pred_samples_hv, lower_q, axis=0).astype(np.float32)
    upper_bound = np.quantile(y_pred_samples_hv, upper_q, axis=0).astype(np.float32)

    cov = np.mean((y_eval >= lower_bound) & (y_eval <= upper_bound))
    coverage_probabilities.append(float(cov))

    width = np.mean(upper_bound - lower_bound)
    interval_widths.append(float(width))

coverage_probabilities = np.asarray(coverage_probabilities, dtype=np.float32)
interval_widths = np.asarray(interval_widths, dtype=np.float32)

abs_miscal = np.abs(coverage_probabilities - confidence_levels)
mace = float(np.mean(abs_miscal))
imce = float(np.trapz(abs_miscal, confidence_levels))
print(f"Calibration (HV): MACE={mace:.4f}, IMCE={imce:.4f}")

# Reliability diagram
plt.figure(figsize=(10, 8))
plt.plot(confidence_levels, coverage_probabilities, "o-", label="Empirical coverage")
plt.plot([confidence_levels.min(), confidence_levels.max()],
         [confidence_levels.min(), confidence_levels.max()],
         "k--", label="Perfect calibration")
plt.xlabel("Nominal confidence level")
plt.ylabel("Empirical coverage")
plt.legend()
plt.tight_layout()
plt.savefig("uq_reliability_coverage_vs_nominal_HV_FINAL.jpeg", dpi=600, format="jpeg")
plt.show()

# Sharpness
plt.figure(figsize=(10, 8))
plt.plot(confidence_levels, interval_widths, "o-")
plt.xlabel("Nominal confidence level")
plt.ylabel("Mean prediction-interval width (HV)")
plt.tight_layout()
plt.savefig("uq_sharpness_interval_width_HV_FINAL.jpeg", dpi=600, format="jpeg")
plt.show()

calib_df = pd.DataFrame({
    "confidence_level": confidence_levels,
    "empirical_coverage": coverage_probabilities,
    "abs_miscalibration": abs_miscal.astype(np.float32),
    "mean_interval_width_HV": interval_widths.astype(np.float32),
})
calib_df.to_csv("uq_calibration_coverage_curve_HV_FINAL.csv", index=False)

summary_df = pd.DataFrame([{
    "MACE": mace,
    "IMCE": imce,
    "MC_std_mean_HV": mc_std_mean,
    "MC_std_median_HV": mc_std_p50
}])
summary_df.to_csv("uq_calibration_summary_HV_FINAL.csv", index=False)

print("Saved: uq_calibration_coverage_curve_HV_FINAL.csv, uq_calibration_summary_HV_FINAL.csv")


# ===========================
# Latent Space Analysis
# ===========================
def ensure_scaled_load(x_load, *, assume_scaled: bool, scaler, dtype=np.float32):
    x = np.asarray(x_load, dtype=dtype)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if assume_scaled:
        return x.astype(dtype)
    return scale_load_np(x, scaler, dtype=dtype)

set_mc_dropout(final_model, False)

Xc = np.asarray(X_comp_all, dtype=np.float32)
xl_raw = np.asarray(X_load_all_raw, dtype=np.float32)
yy = np.asarray(y_all, dtype=np.float32).reshape(-1)

assert Xc.ndim == 2, f"X_comp_all must be 2D, got shape {Xc.shape}"
n = Xc.shape[0]
xl_raw = xl_raw.reshape(-1)
assert xl_raw.shape[0] == n, f"Load length {xl_raw.shape[0]} != n {n}"
assert yy.shape[0] == n, f"y length {yy.shape[0]} != n {n}"

data = pd.DataFrame(Xc, columns=composition_cols)
data["Load_raw"] = xl_raw
data["HV"] = yy

# Latent extractor model
latent_extractor_final = tf.keras.Model(
    inputs=final_model.input,
    outputs=final_model.get_layer("vib_layer").output,
    name="latent_extractor_final",
)

ASSUME_SPLIT_LOADS_ARE_SCALED = True

X_load_train_fit_scaled = ensure_scaled_load(
    X_load_train_fit, assume_scaled=ASSUME_SPLIT_LOADS_ARE_SCALED, scaler=scaler_load_final
)
X_load_cal_scaled = ensure_scaled_load(
    X_load_cal, assume_scaled=ASSUME_SPLIT_LOADS_ARE_SCALED, scaler=scaler_load_final
)
X_load_test_scaled = ensure_scaled_load(
    X_load_test, assume_scaled=ASSUME_SPLIT_LOADS_ARE_SCALED, scaler=scaler_load_final
)

# Extract deterministic latents
Z_train = latent_extractor_final.predict(
    [np.asarray(X_comp_train_fit, dtype=np.float32), X_load_train_fit_scaled],
    verbose=0
).astype(np.float32)

Z_cal = latent_extractor_final.predict(
    [np.asarray(X_comp_cal, dtype=np.float32), X_load_cal_scaled],
    verbose=0
).astype(np.float32)

Z_test = latent_extractor_final.predict(
    [np.asarray(X_comp_test, dtype=np.float32), X_load_test_scaled],
    verbose=0
).astype(np.float32)

# Assemble for PCA/GMM
Z = np.vstack([Z_train, Z_cal, Z_test]).astype(np.float32)

y_latent = np.concatenate([
    np.asarray(y_train_fit, dtype=np.float32).reshape(-1),
    np.asarray(y_cal, dtype=np.float32).reshape(-1),
    np.asarray(y_test, dtype=np.float32).reshape(-1),
]).astype(np.float32)

split_ids = np.concatenate([
    np.full(Z_train.shape[0], 1, dtype=int),
    np.full(Z_cal.shape[0],   2, dtype=int),
    np.full(Z_test.shape[0],  3, dtype=int),
])

if Z.shape[0] != y_latent.shape[0]:
    raise ValueError(f"Mismatch: Z N={Z.shape[0]} vs y N={y_latent.shape[0]}")

print("Latent extraction complete:")
print(f"  Z_train: {Z_train.shape}, Z_cal: {Z_cal.shape}, Z_test: {Z_test.shape}, Z_all: {Z.shape}")


# Elbow method in FINAL-model latent space
set_mc_dropout(final_model, False)

# Latent extractor
Xc_train = np.asarray(X_comp_train_fit, dtype=np.float32)
Xl_train = np.asarray(X_load_train_fit, dtype=np.float32)

if Xl_train.ndim == 1:
    Xl_train = Xl_train.reshape(-1, 1)

# Deterministic latent embedding
Z_train = latent_extractor_final.predict([Xc_train, Xl_train], verbose=0).astype(np.float32)

Zk = StandardScaler().fit_transform(Z_train).astype(np.float32)

cluster_range = range(1, 11)
wcss = []
for k in cluster_range:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(Zk)
    wcss.append(float(km.inertia_))

plt.figure(figsize=(10, 8))
plt.plot(list(cluster_range), wcss, marker="o", markersize=8, linestyle="--", linewidth=2)
plt.xlabel("Number of Clusters (k)", fontsize=22, weight="bold")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=22, weight="bold")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("Elbow_Method_for_Optimal_Number_of_Clusters.jpeg", dpi=600, format="jpeg")
plt.show()


# Design the latent space
set_mc_dropout(final_model, False)

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

Xc_train = np.asarray(X_comp_train_fit, dtype=np.float32)
Xc_cal   = np.asarray(X_comp_cal, dtype=np.float32)
Xc_test  = np.asarray(X_comp_test, dtype=np.float32)

Xl_train = _ensure_2d(X_load_train_fit)
Xl_cal   = _ensure_2d(X_load_cal)
Xl_test  = _ensure_2d(X_load_test)

Z_train = latent_extractor_final.predict([Xc_train, Xl_train], verbose=0).astype(np.float32)
Z_cal   = latent_extractor_final.predict([Xc_cal,   Xl_cal],   verbose=0).astype(np.float32)
Z_test  = latent_extractor_final.predict([Xc_test,  Xl_test],  verbose=0).astype(np.float32)

Z = np.vstack([Z_train, Z_cal, Z_test]).astype(np.float32)

split_id = np.concatenate([
    np.full(len(Z_train), 1, dtype=int),
    np.full(len(Z_cal),   2, dtype=int),
    np.full(len(Z_test),  3, dtype=int),
])

scaler_z = StandardScaler()
scaler_z.fit(Z_train)
Z_train_std = scaler_z.transform(Z_train).astype(np.float32)
Z_std       = scaler_z.transform(Z).astype(np.float32)

# KMeans clustering
K_CLUSTERS = 3
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=seed, n_init=10)
kmeans.fit(Z_train_std)
clusters = kmeans.predict(Z_std).astype(int)

# t-SNE
tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
Z_2d = tsne.fit_transform(Z_std).astype(np.float32)

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(
    Z_2d[:, 0], Z_2d[:, 1],
    c=clusters, cmap="plasma", s=200, edgecolor="k", alpha=0.7
)

legend1 = ax.legend(*sc.legend_elements(), title="Clusters")
ax.add_artist(legend1)

ax.set_xlabel("t-SNE 1", fontsize=22, weight="bold")
ax.set_ylabel("t-SNE 2", fontsize=22, weight="bold")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("tSNE_LatentSpace_KMeansClusters.jpeg", dpi=600, format="jpeg")
plt.show()

unique_clusters = np.unique(clusters)
print(f"Found {len(unique_clusters)} clusters:", unique_clusters)


# t-SNE plot colored by hardness (aligned + deterministic)
set_mc_dropout(final_model, False)

y_latent = np.concatenate([
    np.asarray(y_train_fit, dtype=np.float32).reshape(-1),
    np.asarray(y_cal,       dtype=np.float32).reshape(-1),
    np.asarray(y_test,      dtype=np.float32).reshape(-1),
]).astype(np.float32)

Z_2d = np.asarray(Z_2d, dtype=np.float32)

if Z_2d.shape[0] != y_latent.shape[0]:
    raise ValueError(f"Mismatch: Z_2d N={Z_2d.shape[0]} vs y_latent N={y_latent.shape[0]}")

vmin = float(np.min(y_latent))
vmax = float(np.max(y_latent))

fig, ax = plt.subplots(figsize=(12, 8))
sc_h = ax.scatter(
    Z_2d[:, 0], Z_2d[:, 1],
    c=y_latent, cmap="coolwarm",
    vmin=vmin, vmax=vmax,
    s=200, edgecolor="k", alpha=0.7
)

cbar = plt.colorbar(sc_h)
cbar.set_label("Hardness (HV)", fontsize=22, weight="bold")

ax.set_xlabel("t-SNE 1", fontsize=22, weight="bold")
ax.set_ylabel("t-SNE 2", fontsize=22, weight="bold")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("tSNE_LatentSpace_ColorByHardness.jpeg", dpi=600, format="jpeg")
plt.show()


#  Hardness distributions across clusters
clusters = np.asarray(clusters, dtype=int).reshape(-1)

y_latent = np.concatenate([
    np.asarray(y_train_fit, dtype=np.float32).reshape(-1),
    np.asarray(y_cal,       dtype=np.float32).reshape(-1),
    np.asarray(y_test,      dtype=np.float32).reshape(-1),
]).astype(np.float32)

if clusters.shape[0] != y_latent.shape[0]:
    raise ValueError(f"Mismatch: clusters N={clusters.shape[0]} vs y_latent N={y_latent.shape[0]}")

# Group HV by cluster
uniq = np.unique(clusters)

cluster_medians = {k: float(np.median(y_latent[clusters == k])) for k in uniq}
uniq_sorted = np.array(sorted(uniq, key=lambda k: cluster_medians[k]), dtype=int)

data_by_cluster = [y_latent[clusters == k] for k in uniq_sorted]
labels = [str(k) for k in uniq_sorted]

plt.figure(figsize=(10, 6))
plt.boxplot(data_by_cluster, labels=labels, showfliers=True)

plt.xlabel("Cluster Label (KMeans)", fontsize=22, weight="bold")
plt.ylabel("Hardness (HV)", fontsize=22, weight="bold")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("Hardness_Distribution_Across_Clusters.jpeg", dpi=600, format="jpeg")
plt.show()


# Global GMM in latent space
latent_extractor_final = tf.keras.Model(
    inputs=final_model.input,
    outputs=final_model.get_layer("vib_layer").output,
    name="latent_extractor_final"
)

Z_train = latent_extractor_final.predict(
    [X_comp_train_fit.astype(np.float32), X_load_train_fit.astype(np.float32)],
    verbose=0
).astype(np.float32)

Z_cal = latent_extractor_final.predict(
    [X_comp_cal.astype(np.float32), X_load_cal.astype(np.float32)],
    verbose=0
).astype(np.float32)

Z_test = latent_extractor_final.predict(
    [X_comp_test.astype(np.float32), X_load_test.astype(np.float32)],
    verbose=0
).astype(np.float32)

Z = np.vstack([Z_train, Z_cal, Z_test]).astype(np.float32)

y_latent = np.concatenate([
    np.asarray(y_train_fit, dtype=np.float32).reshape(-1),
    np.asarray(y_cal,       dtype=np.float32).reshape(-1),
    np.asarray(y_test,      dtype=np.float32).reshape(-1),
]).astype(np.float32)

if Z.shape[0] != y_latent.shape[0]:
    raise ValueError(f"Mismatch: Z N={Z.shape[0]} vs y_latent N={y_latent.shape[0]}")

Z_prior = Z_train

K = 4 
gmm_latent = GaussianMixture(
    n_components=K,
    covariance_type="full",
    reg_covar=1e-6,
    n_init=10,
    random_state=seed
)
gmm_latent.fit(Z_prior)

labels_all = gmm_latent.predict(Z)
resp_all   = gmm_latent.predict_proba(Z)

best_k = int(np.argmax([y_latent[labels_all == k].mean() for k in range(K)]))
print("Selected component (highest mean HV):", best_k)


# Average composition per cluster
cluster_labels_used = np.asarray(clusters, dtype=int).reshape(-1)

X_comp_stack = np.vstack([
    X_comp_train_fit.astype(np.float32),
    X_comp_cal.astype(np.float32),
    X_comp_test.astype(np.float32),
]).astype(np.float32)

if X_comp_stack.shape[0] != cluster_labels_used.shape[0]:
    raise ValueError(f"Alignment error: X_comp_stack N={X_comp_stack.shape[0]} vs labels N={cluster_labels_used.shape[0]}")

data_local = pd.DataFrame(X_comp_stack, columns=composition_cols)
data_local["Cluster_Label"] = cluster_labels_used

cluster_compositions = data_local.groupby("Cluster_Label")[composition_cols].mean()
print(cluster_compositions)
cluster_compositions.to_csv("cluster_mean_compositions.csv", index=True)

Z_train = np.asarray(Z_train, dtype=np.float32)
Z = np.asarray(Z, dtype=np.float32)

K_candidates = [1, 2, 3, 4, 5, 6]
bics = []
gmms = []

for K in K_candidates:
    g = GaussianMixture(
        n_components=K,
        covariance_type="full",
        reg_covar=1e-6,
        n_init=10,
        random_state=seed
    )
    g.fit(Z_train)
    bics.append(float(g.bic(Z_train)))
    gmms.append(g)

best_idx = int(np.argmin(bics))
gmm_latent = gmms[best_idx]
K_best = K_candidates[best_idx]
print(f"[GMM prior] Selected K={K_best} by BIC on Z_train.")

logp_all = gmm_latent.score_samples(Z).astype(np.float32)
logp_train = gmm_latent.score_samples(Z_train).astype(np.float32)

logp_thr = float(np.percentile(logp_train, 1.0))
print(f"[GMM prior] logp threshold (1st percentile of train): {logp_thr:.3f}")


# Fit a separate GMM on the 2D latent space (for visualization only)
Z_prior = np.asarray(Z_prior)

if Z_prior.ndim != 2:
    raise ValueError(f"Z_prior must be 2D, got {Z_prior.shape}")

if Z_prior.shape[1] != 2:
    if "latent_tsne" in globals():
        Z_prior_2d = np.asarray(latent_tsne)
        if Z_prior_2d.ndim != 2 or Z_prior_2d.shape[1] != 2:
            raise ValueError(f"latent_tsne must be (N,2), got {Z_prior_2d.shape}")
    else:
        Z_prior_2d = PCA(n_components=2, random_state=1).fit_transform(Z_prior)
else:
    Z_prior_2d = Z_prior

Z_prior_2d = np.asarray(Z_prior_2d, dtype=np.float64)

gmm_temp = GaussianMixture(n_components=3, covariance_type="full", random_state=1, n_init=10)
gmm_temp.fit(Z_prior_2d)

x_temp = np.linspace(Z_prior_2d[:, 0].min() - 1, Z_prior_2d[:, 0].max() + 1, 100)
y_temp = np.linspace(Z_prior_2d[:, 1].min() - 1, Z_prior_2d[:, 1].max() + 1, 100)
X_temp, Y_temp = np.meshgrid(x_temp, y_temp)
grid_points_temp = np.column_stack([X_temp.ravel(), Y_temp.ravel()])

log_likelihood_temp = gmm_temp.score_samples(grid_points_temp)
Z_temp = log_likelihood_temp.reshape(X_temp.shape)

x_smooth = np.linspace(X_temp.min(), X_temp.max(), 300)
y_smooth = np.linspace(Y_temp.min(), Y_temp.max(), 300)
X_smooth_temp, Y_smooth_temp = np.meshgrid(x_smooth, y_smooth)

Z_smooth_temp = griddata(
    (X_temp.ravel(), Y_temp.ravel()),
    Z_temp.ravel(),
    (X_smooth_temp, Y_smooth_temp),
    method="cubic",
)

plt.figure(figsize=(12, 8))
contour = plt.contourf(X_smooth_temp, Y_smooth_temp, Z_smooth_temp, levels=50, cmap="coolwarm", alpha=1)
plt.contour(X_smooth_temp, Y_smooth_temp, Z_smooth_temp, levels=50, colors="grey", linewidths=0.5, alpha=0.5)

plt.scatter(Z_prior_2d[:, 0], Z_prior_2d[:, 1], c="black", s=10, alpha=0.3)

cbar = plt.colorbar(contour)
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Log Likelihood", fontsize=22, weight="bold")
plt.xlabel("Dimension 1", fontsize=22, weight="bold")
plt.ylabel("Dimension 2", fontsize=22, weight="bold")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("gmm_probability_density_with_scatter_points.png", dpi=600, format="png")
plt.show()


# ================================
# SHAP on latent-GMM clusters
# ================================
def scale_load_np_new(x_load_raw: np.ndarray, scaler) -> np.ndarray:
    """Always returns (N,1) float32 scaled load as FINAL model expects."""
    x = np.asarray(x_load_raw, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim == 2 and x.shape[1] != 1:
        raise ValueError(f"Load must be shape (N,1). Got {x.shape}.")
    return scaler.transform(x).astype(np.float32)

def predict_final_np(model: tf.keras.Model, X_comp: np.ndarray, X_load_raw: np.ndarray) -> np.ndarray:
    comp = np.asarray(X_comp, dtype=np.float32)
    load_scaled = scale_load_np_new(X_load_raw, scaler_load_final)
    out = model([comp, load_scaled], training=False)

    if isinstance(out, dict):
        y = out["Output_Layer"]
    elif isinstance(out, (list, tuple)):
        y = out[0]
    else:
        y = out

    return tf.reshape(y, (-1,)).numpy().astype(np.float32)

def get_latents_np(model: tf.keras.Model, X_comp: np.ndarray, X_load_raw: np.ndarray) -> np.ndarray:
    """Latents from FINAL model, always with FINAL-scaled load."""
    comp = np.asarray(X_comp, dtype=np.float32)
    load_scaled = scale_load_np_new(X_load_raw, scaler_load_final)
    latent_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("vib_layer").output,
        name="latent_model_final"
    )
    Z = latent_model.predict([comp, load_scaled], verbose=0)
    return np.asarray(Z, dtype=np.float32)


train_idx = np.asarray(train_idx, dtype=int)
test_idx  = np.asarray(test_idx, dtype=int)

X_comp_all_f     = np.asarray(X_comp_all, dtype=np.float32)
X_load_all_raw_f = np.asarray(X_load_all_raw, dtype=np.float32).reshape(-1, 1)
y_all_f          = np.asarray(y_all, dtype=np.float32).reshape(-1)

n_all, n_comp = X_comp_all_f.shape[0], X_comp_all_f.shape[1]
if X_load_all_raw_f.shape[0] != n_all:
    raise ValueError(f"Load length {X_load_all_raw_f.shape[0]} != X_comp length {n_all}")
if y_all_f.shape[0] != n_all:
    raise ValueError(f"y length {y_all_f.shape[0]} != X_comp length {n_all}")

feature_names = list(composition_cols) + ["Load_raw"]

X_full_train = np.concatenate([X_comp_all_f[train_idx], X_load_all_raw_f[train_idx]], axis=1).astype(np.float32)
X_full_test  = np.concatenate([X_comp_all_f[test_idx],  X_load_all_raw_f[test_idx]],  axis=1).astype(np.float32)

latent_train = get_latents_np(final_model, X_comp_all_f[train_idx], X_load_all_raw_f[train_idx])
latent_test  = get_latents_np(final_model, X_comp_all_f[test_idx],  X_load_all_raw_f[test_idx])

gmm_shap = GaussianMixture(
    n_components=3,
    covariance_type="full",
    reg_covar=1e-6,
    n_init=5,
    random_state=seed
)
gmm_shap.fit(latent_train.astype(np.float64))

cluster_labels_train = gmm_shap.predict(latent_train.astype(np.float64))
cluster_labels_test  = gmm_shap.predict(latent_test.astype(np.float64))

# SHAP prediction wrapper
set_mc_dropout(final_model, False)

def shap_predict(X_full: np.ndarray) -> np.ndarray:
    X_full = np.asarray(X_full, dtype=np.float32)
    if X_full.ndim != 2 or X_full.shape[1] != (n_comp + 1):
        raise ValueError(f"Expected X_full shape (N,{n_comp+1}), got {X_full.shape}")

    comp = X_full[:, :n_comp].astype(np.float32)
    load_raw = X_full[:, n_comp:].astype(np.float32).reshape(-1, 1)
    return predict_final_np(final_model, comp, load_raw)

# SHAP explainer
BG_SIZE = int(min(500, X_full_train.shape[0]))
X_bg = shap.sample(X_full_train, BG_SIZE, random_state=seed)

explainer = shap.Explainer(shap_predict, X_bg, feature_names=feature_names)

os.makedirs("shap_export", exist_ok=True)

cluster_means = {}

y_test_true = y_all_f[test_idx]

for cid in np.unique(cluster_labels_test):
    idx_local = np.where(cluster_labels_test == cid)[0]
    if idx_local.size == 0:
        continue

    X_cluster = X_full_test[idx_local]
    y_cluster = y_test_true[idx_local]

    shap_values = explainer(X_cluster)

    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title(f"SHAP Bar Plot (HELD-OUT) - Cluster {cid}")
    plt.tight_layout()
    plt.savefig(f"shap_export/Cluster_{cid}_SHAP_Bar.png", dpi=300)
    plt.close()

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    bar_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    bar_df.to_csv(f"shap_export/Cluster_{cid}_SHAP_Bar_Data.csv", index=False)

    top_features = np.argsort(mean_abs)[::-1][:5]
    for i in top_features:
        shap.dependence_plot(
            i,
            shap_values.values,
            X_cluster,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"Cluster {cid} (HELD-OUT) - SHAP Dependence: {feature_names[i]}")
        plt.tight_layout()
        safe_name = str(feature_names[i]).replace("/", "_").replace(" ", "_")
        plt.savefig(f"shap_export/Cluster_{cid}_SHAP_Dep_{safe_name}.png", dpi=300)
        plt.close()

    mean_feat = pd.DataFrame(X_cluster, columns=feature_names).mean()
    mean_feat["Cluster_ID"] = cid
    mean_feat["N_points"] = int(X_cluster.shape[0])
    mean_feat["HV_mean_true"] = float(np.mean(y_cluster)) if y_cluster.size else np.nan
    cluster_means[f"Cluster_{cid}"] = mean_feat

means_df = pd.DataFrame(cluster_means).T
means_df.to_csv("shap_export/Cluster_Physical_Feature_Means_HELDOUT.csv", index=True)
print(means_df)


# ============================
# Latent Traversal Analysis
# ============================
def ensure_2d_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1, 1) if x.ndim == 1 else x

def get_latents_np(model: tf.keras.Model, X_comp: np.ndarray, X_load_scaled: np.ndarray) -> np.ndarray:
    comp = np.asarray(X_comp, dtype=np.float32)
    load = ensure_2d_col(np.asarray(X_load_scaled, dtype=np.float32))

    latent_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("vib_layer").output,
        name="latent_extractor_final"
    )
    Z = latent_model.predict([comp, load], verbose=0)
    Z = np.asarray(Z, dtype=np.float32)

    if Z.ndim != 2 or Z.shape[0] != comp.shape[0]:
        raise ValueError(f"Latent extraction failed: got Z shape {Z.shape}, expected (N, d).")
    if np.any(~np.isfinite(Z)):
        raise ValueError("Non-finite values in extracted latents.")
    return Z

try:
    set_mc_dropout(final_model, False)
except Exception:
    pass

# Extract latents from the training-fit set
Z_train = get_latents_np(
    final_model,
    np.asarray(X_comp_train_fit, dtype=np.float32),
    np.asarray(X_load_train_fit, dtype=np.float32)
)
y_train_vec = np.asarray(y_train_fit, dtype=np.float32).reshape(-1)

if Z_train.shape[0] != y_train_vec.shape[0]:
    raise ValueError(f"Mismatch: Z_train N={Z_train.shape[0]} vs y_train_fit N={y_train_vec.shape[0]}")

latent_dim = int(Z_train.shape[1])


# ------------------------------------------------------------
# Build a head model: (z, load_scaled) -> hardness_scaled
# ------------------------------------------------------------
# Inputs
z_in = tf.keras.Input(shape=(latent_dim,), dtype=tf.float32, name="z_in")
load_in = tf.keras.Input(shape=(1,), dtype=tf.float32, name="load_in_scaled")

# Reuse the trained load branch layers
load_weighted, _ = final_model.get_layer("Load_Attention")(load_in)
x_load = final_model.get_layer("Load_Enc_D1")(load_weighted)
x_load = final_model.get_layer("Load_Enc_BN1")(x_load)
x_load = final_model.get_layer("Load_Enc_DO1")(x_load)

hard_in = final_model.get_layer("Hard_Input")([z_in, x_load])

h = final_model.get_layer("Hard_Trunk_D1")(hard_in)
h = final_model.get_layer("Hard_Trunk_BN1")(h)
h = final_model.get_layer("Hard_Trunk_DO1")(h)
y_out_scaled = final_model.get_layer("Output_Layer")(h)

hardness_from_zL_scaled = tf.keras.Model(
    inputs=[z_in, load_in],
    outputs=y_out_scaled,
    name="hardness_from_z_and_load_scaled"
)

# -----------------------------------------
# Choose candidate latent anchors
# -----------------------------------------
center_lv = Z_train.mean(axis=0).astype(np.float32)

top_percentile = 92
thr = float(np.percentile(y_train_vec, top_percentile))
high_idx = np.where(y_train_vec >= thr)[0]
if high_idx.size == 0:
    raise ValueError(f"No points found above {top_percentile}th percentile in y_train_fit.")
rng = np.random.default_rng(seed)
high_lv = Z_train[int(rng.choice(high_idx))].astype(np.float32)

candidates = {"Center": center_lv, "High": high_lv}

CLIP_Q = 0.01
lb = np.quantile(Z_train, CLIP_Q, axis=0).astype(np.float32)
ub = np.quantile(Z_train, 1.0 - CLIP_Q, axis=0).astype(np.float32)

L_star = float(np.median(np.asarray(X_load_train_fit, dtype=np.float32)))
L_star_arr = np.array([[L_star]], dtype=np.float32)


plt.rcParams["figure.facecolor"]  = "white"
plt.rcParams["axes.facecolor"]    = "white"
plt.rcParams["savefig.facecolor"] = "white"

N_STEPS = 25


# Latent traversal loops
for name, base_lv in candidates.items():
    rows = []

    ncols = 4
    nrows = int(np.ceil(latent_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows), facecolor="white")
    axes = np.array(axes).reshape(-1)

    base = np.asarray(base_lv, dtype=np.float32).copy()

    for dim in range(latent_dim):
        grid = np.linspace(lb[dim], ub[dim], N_STEPS).astype(np.float32)

        traj_hv = np.empty((N_STEPS,), dtype=np.float32)
        traj_scaled = np.empty((N_STEPS,), dtype=np.float32)

        for i, val in enumerate(grid):
            lv = base.copy()
            lv[dim] = val

            y_s = hardness_from_zL_scaled([lv[None, :], L_star_arr], training=False)
            y_s = float(tf.reshape(y_s, (-1,))[0].numpy())
            hv = float(y_s * y_sd_final + y_mu_final)

            traj_scaled[i] = y_s
            traj_hv[i] = hv

            rows.append({
                "candidate": name,
                "dim": int(dim + 1),
                "step": int(i),
                "latent_value": float(val),
                "fixed_load_scaled": float(L_star),
                "predicted_y_scaled": float(y_s),
                "predicted_hv": float(hv),
            })

        ax = axes[dim]
        ax.plot(grid, traj_hv, marker="o", markersize=5, linewidth=1.5, alpha=1.0)
        ax.set_title(f"Dim {dim+1}", fontsize=11, pad=6)
        ax.set_xlabel("Latent Value", fontsize=9)
        ax.set_ylabel("Predicted Hardness (HV)", fontsize=9)
        ax.grid(False)

    for j in range(latent_dim, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Latent Traversal — {name} (fixed load scaled={L_star:.3f}; bounds=q[{int(CLIP_Q*100)}%, {int((1-CLIP_Q)*100)}%])",
        fontsize=20,
        y=1.01
    )
    plt.tight_layout()

    fig.savefig(
        f"latent_traversal_{name.lower()}_hv.jpg",
        format="jpg",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()

    df_long = pd.DataFrame(
        rows,
        columns=[
            "candidate", "dim", "step", "latent_value",
            "fixed_load_scaled", "predicted_y_scaled", "predicted_hv"
        ]
    )
    csv_name = f"latent_traversal_{name.lower()}_long.csv"
    df_long.to_csv(csv_name, index=False)
    print(f"Saved long-format traversal data: {csv_name}")


# =======================================
# Global Latent–Hardness Correlations
# =======================================
vib_layer = final_model.get_layer("vib_layer")
latent_extractor = tf.keras.Model(
    inputs=final_model.input,
    outputs=vib_layer.output,
    name="latent_extractor_final"
)

latent_dim = None
for attr in ("latent_dim", "units", "dim"):
    if hasattr(vib_layer, attr):
        try:
            latent_dim = int(getattr(vib_layer, attr))
            break
        except Exception:
            pass

X_comp_all_f = np.asarray(X_comp_all, dtype=np.float32)
X_load_all_raw_f = np.asarray(X_load_all_raw, dtype=np.float32).reshape(-1, 1)
X_load_all_scaled = scaler_load_final.transform(X_load_all_raw_f).astype(np.float32)

try:
    set_mc_dropout(final_model, False)
except Exception:
    pass

all_latents = latent_extractor.predict([X_comp_all_f, X_load_all_scaled], verbose=0).astype(np.float32)

if all_latents.ndim != 2:
    raise ValueError(f"Latents must be 2D; got {all_latents.shape}")
if np.any(~np.isfinite(all_latents)):
    raise ValueError("Non-finite values in extracted latents.")

if latent_dim is None:
    latent_dim = int(all_latents.shape[1])
else:
    if all_latents.shape[1] != latent_dim:
        raise ValueError(f"Latent dim mismatch: vib_layer says {latent_dim}, Z has {all_latents.shape[1]}")

print("Detected latent_dim =", latent_dim)

z_in = tf.keras.Input(shape=(latent_dim,), dtype=tf.float32, name="z_in")
load_in = tf.keras.Input(shape=(1,), dtype=tf.float32, name="load_in_scaled")

load_weighted, _ = final_model.get_layer("Load_Attention")(load_in)
x_load = final_model.get_layer("Load_Enc_D1")(load_weighted)
x_load = final_model.get_layer("Load_Enc_BN1")(x_load)
x_load = final_model.get_layer("Load_Enc_DO1")(x_load)

hard_in = final_model.get_layer("Hard_Input")([z_in, x_load])

h = final_model.get_layer("Hard_Trunk_D1")(hard_in)
h = final_model.get_layer("Hard_Trunk_BN1")(h)
h = final_model.get_layer("Hard_Trunk_DO1")(h)
y_scaled_out = final_model.get_layer("Output_Layer")(h)

hardness_from_zL_scaled = tf.keras.Model(
    inputs=[z_in, load_in],
    outputs=y_scaled_out,
    name="hardness_from_z_and_load_scaled"
)

y_scaled_emp = hardness_from_zL_scaled([all_latents, X_load_all_scaled], training=False)
y_scaled_emp = y_scaled_emp.numpy().reshape(-1).astype(np.float64)

hv_emp = (y_scaled_emp * float(y_sd_final) + float(y_mu_final)).astype(np.float64)


# =============================================================
# Correlations of each latent dimension with predicted HV
# =============================================================
pearson_coeffs  = np.empty(latent_dim, dtype=np.float64)
spearman_coeffs = np.empty(latent_dim, dtype=np.float64)

for d in range(latent_dim):
    c = all_latents[:, d].astype(np.float64)
    pearson_coeffs[d]  = pearsonr(c, hv_emp)[0]
    spearman_coeffs[d] = spearmanr(c, hv_emp)[0]

dims = np.arange(1, latent_dim + 1)

plt.figure(figsize=(12, 8))
plt.plot(dims, pearson_coeffs,  marker="o", linestyle="-",  label="Pearson")
plt.plot(dims, spearman_coeffs, marker="s", linestyle="--", label="Spearman")
plt.xlabel("Latent Dimension", fontsize=14, weight="bold")
plt.ylabel("Correlation with Predicted Hardness (HV)", fontsize=14, weight="bold")
plt.title("Global Latent–Hardness Correlations (empirical latents; FINAL model)", fontsize=16, weight="bold")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

corr_df = pd.DataFrame({
    "Dim": dims,
    "Pearson": pearson_coeffs,
    "Spearman": spearman_coeffs
})
corr_df.to_csv("latent_hardness_correlations_empirical_latents.csv", index=False)
print("Exported: latent_hardness_correlations_empirical_latents.csv")


# ====================================================
# PCA in FINAL latent space vs predicted hardness
# ====================================================
def ensure_2d_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1, 1) if x.ndim == 1 else x

def invert_y_to_hv(y_scaled: np.ndarray, y_mu: float, y_sd: float) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=np.float32).reshape(-1)
    return (y_scaled * float(y_sd) + float(y_mu)).astype(np.float32)

def get_output_layer_tensor(model_out):
    """Robustly extract Output_Layer tensor from a Keras model output (dict/list/tuple/tensor)."""
    if isinstance(model_out, dict):
        if "Output_Layer" not in model_out:
            raise KeyError(f"Model output dict keys: {list(model_out.keys())}; expected 'Output_Layer'.")
        return model_out["Output_Layer"]
    if isinstance(model_out, (list, tuple)):
        return model_out[0]
    return model_out

# Latent extractor
Xc_test = np.asarray(X_comp_test, dtype=np.float32)
Xl_test = ensure_2d_col(scale_load_np_new(np.asarray(X_load_test_raw, dtype=np.float32), scaler_load_final))

try:
    set_mc_dropout(final_model, False)
except Exception:
    pass

latent_vectors_test = latent_model.predict([Xc_test, Xl_test], verbose=0).astype(np.float32)

out_det = final_model([Xc_test, Xl_test], training=False)
y_tensor = get_output_layer_tensor(out_det)
y_det_scaled = y_tensor.numpy().reshape(-1).astype(np.float32)

hardness_preds_hv = invert_y_to_hv(y_det_scaled, y_mu_final, y_sd_final)

if latent_vectors_test.shape[0] != hardness_preds_hv.shape[0]:
    raise ValueError(f"Mismatch: latents N={latent_vectors_test.shape[0]} vs preds N={hardness_preds_hv.shape[0]}")

# PCA basis fit on TRAIN-FIT latents
Z_train = latent_model.predict(
    [np.asarray(X_comp_train_fit, dtype=np.float32),
     ensure_2d_col(np.asarray(X_load_train_fit, dtype=np.float32))],
    verbose=0
).astype(np.float32)

pca = PCA(n_components=2, random_state=seed)
pca.fit(Z_train)
pc_scores = pca.transform(latent_vectors_test).astype(np.float32)

# Linear association of PCs with predicted hardness (HV)
lr1 = LinearRegression().fit(pc_scores[:, [0]], hardness_preds_hv)
r2_pc1 = r2_score(hardness_preds_hv, lr1.predict(pc_scores[:, [0]]))

lr2 = LinearRegression().fit(pc_scores[:, [1]], hardness_preds_hv)
r2_pc2 = r2_score(hardness_preds_hv, lr2.predict(pc_scores[:, [1]]))

plt.figure(figsize=(8, 6))
plt.scatter(pc_scores[:, 0], hardness_preds_hv, s=100, edgecolor="k", alpha=0.6)
x1 = np.array([pc_scores[:, 0].min(), pc_scores[:, 0].max()], dtype=np.float32)[:, None]
plt.plot(x1, lr1.predict(x1), "r--", lw=2, label=f"Fit R²={r2_pc1:.3f}")
plt.xlabel("PC 1", fontsize=14)
plt.ylabel("Predicted Hardness (HV)", fontsize=14)
plt.title("PC1 vs Predicted Hardness (FINAL)", fontsize=16)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(pc_scores[:, 1], hardness_preds_hv, s=100, edgecolor="k", alpha=0.6)
x2 = np.array([pc_scores[:, 1].min(), pc_scores[:, 1].max()], dtype=np.float32)[:, None]
plt.plot(x2, lr2.predict(x2), "r--", lw=2, label=f"Fit R²={r2_pc2:.3f}")
plt.xlabel("PC 2", fontsize=14)
plt.ylabel("Predicted Hardness (HV)", fontsize=14)
plt.title("PC2 vs Predicted Hardness (FINAL)", fontsize=16)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")


# ===================================================
# Gradient-Based Importance of Latent Dimensions
# ===================================================
def ensure_2d_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1, 1) if x.ndim == 1 else x

try:
    set_mc_dropout(final_model, False)
except Exception:
    pass

Xc_test = np.asarray(X_comp_test, dtype=np.float32)
Xl_test = ensure_2d_col(scale_load_np_new(np.asarray(X_load_test_raw, dtype=np.float32), scaler_load_final))

latent_vectors_test = latent_model.predict([Xc_test, Xl_test], verbose=0).astype(np.float32)

if latent_vectors_test.ndim != 2 or latent_vectors_test.shape[0] != Xc_test.shape[0]:
    raise ValueError(f"Unexpected latent shape: {latent_vectors_test.shape}")

latent_dim = int(latent_vectors_test.shape[1])

z_in = tf.keras.Input(shape=(latent_dim,), dtype=tf.float32, name="z_in")
load_in = tf.keras.Input(shape=(1,), dtype=tf.float32, name="load_in_scaled")

load_weighted, _ = final_model.get_layer("Load_Attention")(load_in)
x_load = final_model.get_layer("Load_Enc_D1")(load_weighted)
x_load = final_model.get_layer("Load_Enc_BN1")(x_load)
x_load = final_model.get_layer("Load_Enc_DO1")(x_load)

hard_in = final_model.get_layer("Hard_Input")([z_in, x_load])

h = final_model.get_layer("Hard_Trunk_D1")(hard_in)
h = final_model.get_layer("Hard_Trunk_BN1")(h)
y_out_scaled = final_model.get_layer("Output_Layer")(h)

hardness_from_zL_det = tf.keras.Model(
    inputs=[z_in, load_in],
    outputs=y_out_scaled,
    name="hardness_from_z_and_load_det"
)

# ============================
# Compute gradients d(HV)/d(z)
# ============================
N_MAX = 2048
rng = np.random.default_rng(seed)

if latent_vectors_test.shape[0] > N_MAX:
    idx = rng.choice(latent_vectors_test.shape[0], size=N_MAX, replace=False)
    Z_use = latent_vectors_test[idx]
    L_use = Xl_test[idx]
else:
    Z_use = latent_vectors_test
    L_use = Xl_test

z_var = tf.Variable(np.asarray(Z_use, dtype=np.float32))
l_tensor = tf.convert_to_tensor(np.asarray(L_use, dtype=np.float32))

with tf.GradientTape() as tape:
    y_pred_scaled = hardness_from_zL_det([z_var, l_tensor], training=False)
    y_pred_scaled = tf.squeeze(y_pred_scaled, axis=-1)
    obj = tf.reduce_sum(y_pred_scaled)

grads = tape.gradient(obj, z_var)

if grads is None:
    raise RuntimeError("GradientTape returned None gradients. Check that y depends on z in the head model.")

y_sd = float(y_sd_final) if "y_sd_final" in globals() else 1.0
sensitivities = (y_sd * tf.reduce_mean(tf.abs(grads), axis=0)).numpy().astype(np.float32)  # (latent_dim,)

dims = np.arange(1, latent_dim + 1)

# ============
# Plotting
# ============
plt.figure(figsize=(12, 6))
plt.bar(dims, sensitivities, edgecolor="black", linewidth=1.0)
plt.xticks(dims, [f"z{i}" for i in dims], rotation=90, fontsize=10)
plt.ylabel(r"Mean $|\partial HV / \partial z|$", fontsize=14, weight="bold")
plt.xlabel("Latent Dimension", fontsize=14, weight="bold")
plt.title("Gradient-Based Latent Dimension Importance (deterministic head; load-conditioned)", fontsize=16, weight="bold")
plt.grid(False)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("latent_gradient_importance_det_head.png", dpi=600)
plt.show()

imp_df = pd.DataFrame({
    "LatentDim": dims.astype(int),
    "MeanAbs_dHV_dz": sensitivities.reshape(-1)
})
imp_df.to_csv("latent_gradient_importance_det_head.csv", index=False)
print("Saved: latent_gradient_importance_det_head.png and latent_gradient_importance_det_head.csv")


# ================================
# Inverse Design
# ================================

# Latent prior (Weighted GMM) + Mixture-aware Multi-chain MCMC sampling
for name in [
    "final_model", "scaler_load_final",
    "X_comp_train_fit", "X_load_train_fit_raw", "y_train_fit",
    "composition_cols", "seed", "y_mu_final", "y_sd_final",
    "w_train_fit"
]:
    if name not in globals():
        raise RuntimeError(f"Missing required object: {name}")

# Helpers
def ensure_2d_col_inv(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"Expected (N,1). Got {x.shape}.")
    return x

def scale_load_np_inv(x_load_raw: np.ndarray, scaler) -> np.ndarray:
    x = ensure_2d_col_inv(x_load_raw)
    return scaler.transform(x).astype(np.float32)

def invert_y_to_hv(y_scaled: np.ndarray, y_mu: float, y_sd: float) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=np.float32).reshape(-1)
    return (y_scaled * float(y_sd) + float(y_mu)).astype(np.float32)

def project_comp_to_simplex_np(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    X = np.clip(X, 0.0, None)
    s = X.sum(axis=1, keepdims=True)
    s[s < eps] = 1.0
    return X / s

def _seed_stream(base_seed: int, n: int) -> np.ndarray:
    ss = np.random.SeedSequence(int(base_seed))
    return ss.generate_state(int(n), dtype=np.uint32)

def set_mc_dropout_inv(model: tf.keras.Model, active: bool) -> None:
    active = bool(active)

    def walk_layers(layer):
        yield layer
        if hasattr(layer, "layers"):
            for ll in layer.layers:
                yield from walk_layers(ll)

    for m in walk_layers(model):
        if m.__class__.__name__ == "MCDropout" and hasattr(m, "mc_active"):
            m.mc_active = active

def _walk_layers(layer):
    yield layer
    if hasattr(layer, "layers"):
        for ll in layer.layers:
            yield from _walk_layers(ll)

n_mc_layers = sum(1 for lyr in _walk_layers(final_model) if lyr.__class__.__name__ == "MCDropout")
if n_mc_layers == 0:
    raise RuntimeError(
        "final_model contains no MCDropout layers. "
        "BN-safe MC-dropout requires MCDropout layers active under training=False."
    )

def assert_mc_dropout_configured(
    model: tf.keras.Model,
    DESIGN_LOAD_VALUE: float,
    n_checks: int = 6,
    atol: float = 1e-8
) -> None:
    b = 8
    Xc = tf.ones((b, len(composition_cols)), dtype=tf.float32) / tf.cast(len(composition_cols), tf.float32)
    Xl_raw = tf.fill((b, 1), tf.constant(float(DESIGN_LOAD_VALUE), tf.float32))
    mean_tf  = tf.constant(np.asarray(scaler_load_final.mean_,  dtype=np.float32).reshape(1, 1), dtype=tf.float32)
    scale_tf = tf.constant(np.asarray(scaler_load_final.scale_, dtype=np.float32).reshape(1, 1), dtype=tf.float32)
    Xl = (Xl_raw - mean_tf) / scale_tf

    draw_seeds = _seed_stream(int(seed) + 12345, int(n_checks))

    set_mc_dropout_inv(model, True)
    ys = []
    for s in range(int(n_checks)):
        tf.keras.utils.set_random_seed(int(draw_seeds[s]))
        out = model([Xc, Xl], training=False)
        if isinstance(out, dict):
            y = out["Output_Layer"]
        elif isinstance(out, (list, tuple)):
            y = out[0]
        else:
            y = out
        ys.append(tf.reshape(tf.cast(y, tf.float32), (-1,)))
    ys = tf.stack(ys, axis=0)
    s_mean = float(tf.reduce_mean(tf.math.reduce_std(ys, axis=0)).numpy())
    set_mc_dropout_inv(model, False)

    if s_mean <= atol:
        raise RuntimeError(
            "MC-dropout appears OFF under training=False (sigma ~ 0). "
            "Ensure MCDropout layers exist and set_mc_dropout_inv(model, True) is called for MC sampling."
        )

DESIGN_LOAD_VALUE = float(0.5)
assert_mc_dropout_configured(final_model, DESIGN_LOAD_VALUE, n_checks=6, atol=1e-8)

# Encoder: (x_comp, x_load_scaled) -> z
encoder_model = tf.keras.Model(
    inputs=final_model.input,
    outputs=final_model.get_layer("vib_layer").output,
    name="encoder_z_final_for_sampling"
)

Xc_train_for_prior = np.asarray(X_comp_train_fit, dtype=np.float32)
Xl_train_fixed_raw = np.full((Xc_train_for_prior.shape[0], 1), DESIGN_LOAD_VALUE, dtype=np.float32)
Xl_train_fixed_scaled = scale_load_np_inv(Xl_train_fixed_raw, scaler_load_final)

z_train_fixedload = encoder_model.predict([Xc_train_for_prior, Xl_train_fixed_scaled], verbose=0).astype(np.float64)
latent_dim = int(z_train_fixedload.shape[1])

def _weighted_quantile(x, q, w):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if x.shape[0] != w.shape[0]:
        raise ValueError("x and w must have same length")
    idx = np.argsort(x)
    x_s = x[idx]
    w_s = w[idx]
    cw = np.cumsum(w_s)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, x_s))

# Weighted GMM prior in latent
K = int(3)
gmm_latent_train = GaussianMixture(
    n_components=K,
    covariance_type="full",
    reg_covar=1e-6,
    n_init=10,
    random_state=int(seed),
)

w_prior = np.asarray(w_train_fit, dtype=np.float64).reshape(-1)
if w_prior.shape[0] != z_train_fixedload.shape[0]:
    raise RuntimeError(f"w_train_fit length {w_prior.shape[0]} != z_train_fixedload N {z_train_fixedload.shape[0]}")

w_prior = np.clip(w_prior, 1e-8, np.inf)
w_prior = w_prior / np.mean(w_prior)

try:
    gmm_latent_train.fit(z_train_fixedload, sample_weight=w_prior)
except TypeError:
    rng_prior = np.random.default_rng(int(seed) + 991)
    p = w_prior / np.sum(w_prior)
    N = z_train_fixedload.shape[0]
    N_boot = int(min(max(5 * N, N), 200000))
    idx = rng_prior.choice(N, size=N_boot, replace=True, p=p)
    gmm_latent_train.fit(z_train_fixedload[idx])

def _log_gmm_pdf(gmm: GaussianMixture, Z: np.ndarray) -> np.ndarray:
    return gmm.score_samples(np.asarray(Z, dtype=np.float64)).astype(np.float64)

LOGP_Q = float(0.10)
train_logp = _log_gmm_pdf(gmm_latent_train, z_train_fixedload.reshape(-1, latent_dim))
logp_thr = _weighted_quantile(train_logp, LOGP_Q, w_prior)
print(f"[MANIFOLD GUARD] logp_thr = weighted q{LOGP_Q:.2f} = {logp_thr:.6f}")

labels_train = gmm_latent_train.predict(z_train_fixedload)
train_idx_by_comp = [np.where(labels_train == k)[0] for k in range(K)]

gmm_mu = [np.asarray(gmm_latent_train.means_[k], dtype=np.float64).reshape(-1) for k in range(K)]
gmm_cov_inv = []
for k in range(K):
    cov_k = np.asarray(gmm_latent_train.covariances_[k], dtype=np.float64)
    if cov_k.ndim == 1:
        cov_k = np.diag(cov_k)
    gmm_cov_inv.append(np.linalg.inv(cov_k + 1e-6 * np.eye(cov_k.shape[0], dtype=np.float64)))

# Decoder (latent -> composition)
vib_layer = final_model.get_layer("vib_layer")
latent_dim_check = int(getattr(vib_layer, "latent_dim", latent_dim))
if latent_dim_check != latent_dim:
    latent_dim = int(latent_dim_check)

latent_in = tf.keras.Input(shape=(latent_dim,), dtype=tf.float32, name="latent_in")
d = final_model.get_layer("Comp_Dec_D1")(latent_in)
d = final_model.get_layer("Comp_Dec_BN1")(d)
d = final_model.get_layer("Comp_Dec_DO1")(d)
d = final_model.get_layer("Comp_Dec_D2")(d)
d = final_model.get_layer("Comp_Dec_BN2")(d)
d = final_model.get_layer("Comp_Dec_DO2")(d)
comp_out = final_model.get_layer("Comp_Recon")(d)
decoder_model = tf.keras.Model(latent_in, comp_out, name="trained_decoder_comp_only_final")

Xl_design_scaled_1x1 = scale_load_np_inv(
    np.array([[DESIGN_LOAD_VALUE]], dtype=np.float32),
    scaler_load_final
).astype(np.float32)

# Hybrid responsibility-weighted Mahalanobis distance
RESP_TAU = float(2.0)
D2_EPS = float(1e-12)

def _maha_d2_vec(z_vec: np.ndarray, mu_vec: np.ndarray, cov_inv: np.ndarray) -> float:
    z_vec = np.asarray(z_vec, dtype=np.float64).reshape(-1)
    mu_vec = np.asarray(mu_vec, dtype=np.float64).reshape(-1)
    dz = z_vec - mu_vec
    return float(dz @ cov_inv @ dz)

ZCONF_FINAL = float(1.645)
ZCONF_MCMC_UTILITY = float(1.0)

# Calibrate strictness using train@DESIGN_LOAD_VALUE
LCB_CAL_S = int(300)
LCB_CAL_N = int(min(5000, Xc_train_for_prior.shape[0]))
rng_lcb = np.random.default_rng(int(seed) + 20231)
idx_lcb = rng_lcb.choice(Xc_train_for_prior.shape[0], size=LCB_CAL_N, replace=False)

Xc_lcb = Xc_train_for_prior[idx_lcb].astype(np.float32)
Xl_lcb = np.repeat(Xl_design_scaled_1x1.astype(np.float32), LCB_CAL_N, axis=0)

mc_seeds_lcb = _seed_stream(int(seed) + 91000, int(LCB_CAL_S))

set_mc_dropout_inv(final_model, True)
try:
    ys_lcb = []
    for s in range(LCB_CAL_S):
        tf.keras.utils.set_random_seed(int(mc_seeds_lcb[s]))
        out = final_model([Xc_lcb, Xl_lcb], training=False)
        if isinstance(out, dict):
            out = out["Output_Layer"]
        elif isinstance(out, (list, tuple)):
            out = out[0]
        ys_lcb.append(tf.reshape(tf.cast(out, tf.float32), (-1,)).numpy().astype(np.float32))
finally:
    set_mc_dropout_inv(final_model, False)

ys_lcb = np.stack(ys_lcb, axis=0)  # (S, N)
hv_lcb = invert_y_to_hv(ys_lcb.reshape(-1), y_mu_final, y_sd_final).reshape(ys_lcb.shape)

mu_train_1N = hv_lcb.mean(axis=0).astype(np.float64)
sig_train_1N = hv_lcb.std(axis=0).astype(np.float64)
lcb_train_1N = (mu_train_1N - ZCONF_FINAL * sig_train_1N).astype(np.float64)

MU_PEAK_Q = float(0.95)
mu_peak_thr = float(np.quantile(mu_train_1N, MU_PEAK_Q))
peak_mask = (mu_train_1N >= mu_peak_thr)
if int(peak_mask.sum()) < 200:
    peak_mask = np.ones_like(mu_train_1N, dtype=bool)

LCB_PEAK_Q = float(0.80)
LCB_TARGET = float(np.quantile(lcb_train_1N[peak_mask], LCB_PEAK_Q))

SIGMA_REF_Q75 = float(np.quantile(sig_train_1N, 0.75))

print(f"[TRAIN@1N] mu_peak_thr = q{MU_PEAK_Q:.3f}(mu) = {mu_peak_thr:.2f} HV")
print(f"[TRAIN@1N] LCB_TARGET  = q{LCB_PEAK_Q:.2f}(LCB | mu>=mu_peak_thr) = {LCB_TARGET:.2f} HV")
print(f"[TRAIN@1N] sigma_ref_q75 = {SIGMA_REF_Q75:.2f} HV")

# Target + penalties
BETA_TILT = float(0.25)
T_MC_IN_MCMC = int(20)
D2_FLOOR = float(0.50)

REPULSION_M = int(64)
REPULSION_GAMMA = float(0.15)
REPULSION_EPS = float(1e-6)

def _utility_risk(mu_hv: float, sigma_hv: float) -> float:
    return float(mu_hv - ZCONF_MCMC_UTILITY * sigma_hv)

def _eval_z_mc(
    z: np.ndarray,
    gmm: GaussianMixture,
    T_mc: int = 4,
    sigma_eps: float = 1e-6,
    seed_base: int = 0,
    precomputed_logp: float | None = None,
) -> dict:
    z = np.asarray(z, dtype=np.float32).reshape(1, -1)

    comp = decoder_model.predict(z, verbose=0).astype(np.float32)
    comp = project_comp_to_simplex_np(comp).astype(np.float32)
    comp_tf = tf.constant(comp, dtype=tf.float32)
    Xl_tf = tf.constant(Xl_design_scaled_1x1.astype(np.float32), dtype=tf.float32)

    mc_seeds = _seed_stream(int(seed_base), int(T_mc))

    set_mc_dropout_inv(final_model, True)
    try:
        ys = []
        for s in range(int(T_mc)):
            tf.keras.utils.set_random_seed(int(mc_seeds[s]))
            out = final_model([comp_tf, Xl_tf], training=False)
            if isinstance(out, dict):
                y = out["Output_Layer"]
            elif isinstance(out, (list, tuple)):
                y = out[0]
            else:
                y = out
            ys.append(tf.reshape(tf.cast(y, tf.float32), (-1,)))
        ys = tf.stack(ys, axis=0)
    finally:
        set_mc_dropout_inv(final_model, False)

    y_scaled = ys.numpy().astype(np.float32).reshape(-1)
    y_hv = invert_y_to_hv(y_scaled, y_mu_final, y_sd_final)

    mu_hv = float(np.mean(y_hv))
    sigma_hv = float(np.std(y_hv))
    sigma_hv = float(max(sigma_hv, sigma_eps))

    z64 = np.asarray(z, dtype=np.float64)
    logp = float(precomputed_logp) if precomputed_logp is not None else float(_log_gmm_pdf(gmm, z64)[0])

    r = gmm.predict_proba(z64).reshape(-1).astype(np.float64)
    r_sharp = np.power(np.clip(r, D2_EPS, 1.0), RESP_TAU)
    r_sharp = r_sharp / (np.sum(r_sharp) + D2_EPS)

    z_vec = z64.reshape(-1)
    d2_k = np.empty((gmm.n_components,), dtype=np.float64)
    for kk in range(gmm.n_components):
        d2_k[kk] = _maha_d2_vec(z_vec, gmm_mu[kk], gmm_cov_inv[kk])
    d2 = float(np.sum(r_sharp * d2_k))

    return {"mu_hv": mu_hv, "sigma_hv": sigma_hv, "logp": logp, "d2": d2}

def _repulsion_penalty(z_vec64: np.ndarray, z_memory: list) -> float:
    if len(z_memory) == 0:
        return 0.0
    Zm = np.stack(z_memory, axis=0)
    d2 = np.sum((Zm - z_vec64.reshape(1, -1))**2, axis=1)
    d2_min = float(np.min(d2))
    return float(REPULSION_GAMMA * np.exp(-d2_min / float(REPULSION_EPS + 1.0)))

def _log_target_from_eval(ev: dict, z_vec64: np.ndarray, z_memory: list, burnin: bool = False) -> float:
    if ev["logp"] < logp_thr:
        return -np.inf

    lcb_ev = float(ev["mu_hv"] - ZCONF_FINAL * ev["sigma_hv"])
    shortfall = max(0.0, float(LCB_TARGET) - lcb_ev)
    lcb_pen = 0.0020 * (shortfall ** 2)

    sig_excess = max(0.0, float(ev["sigma_hv"]) - float(SIGMA_REF_Q75))
    sig_pen = 0.0003 * (sig_excess ** 2)

    d2_low = max(0.0, float(D2_FLOOR) - float(ev["d2"]))
    d2_pen = 0.050 * (d2_low ** 2)

    rep_pen = _repulsion_penalty(z_vec64, z_memory)

    U = _utility_risk(ev["mu_hv"], ev["sigma_hv"])
    beta_tilt_eff = (0.10 if burnin else BETA_TILT)
    rep_pen_eff   = (0.0  if burnin else rep_pen)

    return float(ev["logp"] + beta_tilt_eff * U - lcb_pen - sig_pen - d2_pen - rep_pen_eff)

# MCMC hyperparameters
N_CHAINS = int(25)
STEPS_PER_CHAIN = int(700)
BURNIN = int(250)
THIN = int(5)
PER_CHAIN_KEEP = int(60)

PROP_SCALE = float(0.25)
P_GLOBAL   = float(0.10)

P_SMALL = float(0.70)
SMALL_SCALE = float(0.10)
TEMP_BURNIN = float(2.0)

rng = np.random.default_rng(int(seed))

comp_weights = np.asarray(gmm_latent_train.weights_, dtype=np.float64)
comp_weights = comp_weights / (comp_weights.sum() + 1e-12)

local_cov_by_k = []
for k in range(K):
    cov_k = np.asarray(gmm_latent_train.covariances_[k], dtype=np.float64)
    if cov_k.ndim == 1:
        cov_k = np.diag(cov_k)
    cov_k = cov_k + 1e-6 * np.eye(latent_dim, dtype=np.float64)
    local_cov_by_k.append((PROP_SCALE ** 2) * cov_k)

local_cov_inv_by_k = []
local_log_norm_by_k = []
for k in range(K):
    cov = np.asarray(local_cov_by_k[k], dtype=np.float64)
    inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet = np.log(np.maximum(np.linalg.det(cov), 1e-300))
    log_norm = -0.5 * (latent_dim * np.log(2.0 * np.pi) + logdet)
    local_cov_inv_by_k.append(inv)
    local_log_norm_by_k.append(float(log_norm))

def _k_of(z_center: np.ndarray) -> int:
    r = gmm_latent_train.predict_proba(np.asarray(z_center, dtype=np.float64).reshape(1, -1)).reshape(-1)
    return int(np.argmax(r))

def _log_mvnorm_k(z: np.ndarray, mean: np.ndarray, k: int) -> float:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    diff = (z - mean).reshape(-1, 1)
    quad = float(diff.T @ local_cov_inv_by_k[k] @ diff)
    return float(local_log_norm_by_k[k] - 0.5 * quad)

def log_q(z: np.ndarray, z_center: np.ndarray) -> float:
    """
    Correct proposal density for the actual propose_z kernel:
      q(.|z_center) = (1-P_GLOBAL)*N(z_center, local_cov_by_k[k(z_center)]) + P_GLOBAL*gmm(z)
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z_center = np.asarray(z_center, dtype=np.float64).reshape(-1)

    k = _k_of(z_center)
    log_local = np.log(1.0 - P_GLOBAL + 1e-12) + _log_mvnorm_k(z, z_center, k)
    log_gmm   = np.log(P_GLOBAL + 1e-12) + float(_log_gmm_pdf(gmm_latent_train, z.reshape(1, -1))[0])

    m = max(log_local, log_gmm)
    return float(m + np.log(np.exp(log_local - m) + np.exp(log_gmm - m) + 1e-300))

def propose_z(z_cur: np.ndarray) -> np.ndarray:
    """
    Mixture-aware proposal:
      - with prob P_GLOBAL: global draw from GMM
      - otherwise: component-aware local RW using cov of k(argmax p(k|z_cur))
      - two-scale local RW: mostly small steps, sometimes medium steps
    """
    z_cur = np.asarray(z_cur, dtype=np.float64).reshape(-1)

    if rng.random() < P_GLOBAL:
        return gmm_latent_train.sample(1)[0].reshape(-1).astype(np.float64)

    k = _k_of(z_cur)
    cov_loc = local_cov_by_k[k]

    if rng.random() < P_SMALL:
        factor = (SMALL_SCALE / max(PROP_SCALE, 1e-12)) ** 2
        cov_loc = factor * cov_loc

    return rng.multivariate_normal(z_cur, cov_loc).astype(np.float64)

# Multi-chain MCMC
latent_component_ids = []
all_latent_samples = []

all_diag_mu = []
all_diag_sigma = []
all_diag_logp = []
all_diag_d2 = []
all_diag_lcb = []

accept_counts = []
total_counts = []

precheck_rejects = 0
full_rejects = 0

z_memory = []

for chain_id in range(N_CHAINS):
    k_init = int(rng.choice(K, p=comp_weights))
    idx_k = train_idx_by_comp[k_init]
    if idx_k.size > 0:
        z_cur = z_train_fixedload[int(rng.choice(idx_k))].astype(np.float64)
    else:
        z_cur = gmm_latent_train.sample(1)[0].reshape(-1).astype(np.float64)

    logp_cur = float(_log_gmm_pdf(gmm_latent_train, np.asarray(z_cur, dtype=np.float64).reshape(1, -1))[0])
    ev_cur = _eval_z_mc(
        z_cur, gmm_latent_train, T_mc=T_MC_IN_MCMC,
        seed_base=int(seed) + 1000000 + 1000 * chain_id + 0,
        precomputed_logp=logp_cur
    )
    logt_cur = _log_target_from_eval(ev_cur, z_cur, z_memory, burnin=True)

    retry = 0
    while (not np.isfinite(logt_cur)) and (retry < 120):
        if idx_k.size > 0:
            z_cur = z_train_fixedload[int(rng.choice(idx_k))].astype(np.float64)
        else:
            z_cur = gmm_latent_train.sample(1)[0].reshape(-1).astype(np.float64)

        logp_cur = float(_log_gmm_pdf(gmm_latent_train, np.asarray(z_cur, dtype=np.float64).reshape(1, -1))[0])
        ev_cur = _eval_z_mc(
            z_cur, gmm_latent_train, T_mc=T_MC_IN_MCMC,
            seed_base=int(seed) + 1000000 + 1000 * chain_id + 10 + retry,
            precomputed_logp=logp_cur
        )
        logt_cur = _log_target_from_eval(ev_cur, z_cur, z_memory, burnin=True)
        retry += 1

    if not np.isfinite(logt_cur):
        print(f"[MCMC] chain {chain_id}: could not find feasible start. Skipping.")
        accept_counts.append(0)
        total_counts.append(0)
        continue

    acc = 0
    tot = 0
    kept = 0

    for t in range(STEPS_PER_CHAIN):
        tot += 1

        z_prop = propose_z(z_cur)

        logp_prop = float(_log_gmm_pdf(gmm_latent_train, np.asarray(z_prop, dtype=np.float64).reshape(1, -1))[0])
        if logp_prop < logp_thr:
            precheck_rejects += 1
            continue

        ev_prop = _eval_z_mc(
            z_prop, gmm_latent_train, T_mc=T_MC_IN_MCMC,
            seed_base=int(seed) + 2000000 + 100000 * chain_id + t,
            precomputed_logp=logp_prop
        )
        logt_prop = _log_target_from_eval(ev_prop, z_prop, z_memory, burnin=(t < BURNIN))

        if not np.isfinite(logt_prop):
            full_rejects += 1
            continue

        temp = TEMP_BURNIN if t < BURNIN else 1.0
        log_alpha = ((logt_prop - logt_cur) / temp) + (log_q(z_cur, z_prop) - log_q(z_prop, z_cur))
        if np.log(rng.random()) < log_alpha:
            z_cur = z_prop
            ev_cur = ev_prop
            logt_cur = logt_prop
            acc += 1

        if t >= BURNIN and ((t - BURNIN) % THIN == 0):
            all_latent_samples.append(np.asarray(z_cur, dtype=np.float64).copy())

            r_cur = gmm_latent_train.predict_proba(np.asarray(z_cur, dtype=np.float64).reshape(1, -1)).reshape(-1)
            latent_component_ids.append(int(np.argmax(r_cur)))

            all_diag_mu.append(ev_cur["mu_hv"])
            all_diag_sigma.append(ev_cur["sigma_hv"])
            all_diag_logp.append(ev_cur["logp"])
            all_diag_d2.append(ev_cur["d2"])
            all_diag_lcb.append(float(ev_cur["mu_hv"] - ZCONF_FINAL * ev_cur["sigma_hv"]))

            z_memory.append(np.asarray(z_cur, dtype=np.float64).copy())
            if len(z_memory) > int(REPULSION_M):
                z_memory.pop(0)

            kept += 1
            if kept >= PER_CHAIN_KEEP:
                break

    accept_counts.append(acc)
    total_counts.append(tot)

if len(all_latent_samples) == 0:
    raise RuntimeError(
        "Multi-chain MCMC produced zero feasible samples. "
        "Relax logp_thr (increase LOGP_Q), decrease MU_PEAK_Q / LCB_PEAK_Q, "
        "or reduce BETA_TILT."
    )

sampled_latent_vectors = np.asarray(all_latent_samples, dtype=np.float32)
latent_component_ids = np.asarray(latent_component_ids, dtype=np.int32)

acc_rate = float(np.sum(accept_counts) / max(1, np.sum(total_counts)))
print(
    f"[GMM+MCMC multi-chain] chains={N_CHAINS}, pooled_samples={sampled_latent_vectors.shape[0]}, "
    f"accept_rate={acc_rate:.3f}, DESIGN_LOAD_VALUE(raw)={DESIGN_LOAD_VALUE}"
)
print(f"[MCMC diagnostics] precheck_rejects(logp<thr)={precheck_rejects}, full_rejects={full_rejects}")

selected_gmm_component = int(np.bincount(latent_component_ids, minlength=K).argmax())
print(f"[GMM+MCMC] selected_gmm_component (mode over kept samples) = {selected_gmm_component}")

cov_k = np.asarray(gmm_latent_train.covariances_[selected_gmm_component], dtype=np.float64)
if cov_k.ndim == 1:
    cov_k = np.diag(cov_k)
mu_k = np.asarray(gmm_latent_train.means_[selected_gmm_component], dtype=np.float64)

cov_k_inv = np.linalg.inv(cov_k + 1e-6 * np.eye(cov_k.shape[0], dtype=np.float64)).astype(np.float32)
mu_k_tf = tf.constant(mu_k.reshape(1, -1).astype(np.float32), dtype=tf.float32)
cov_k_inv_tf = tf.constant(cov_k_inv, dtype=tf.float32)

mcmc_diag_df = pd.DataFrame({
    "chain_component": latent_component_ids.astype(int),
    "logp_gmm": np.asarray(all_diag_logp, dtype=np.float64),
    "maha_d2_hybrid": np.asarray(all_diag_d2, dtype=np.float64),
    "mu_hv_mccheap": np.asarray(all_diag_mu, dtype=np.float64),
    "sigma_hv_mccheap": np.asarray(all_diag_sigma, dtype=np.float64),
    "lcb_hv_mccheap": np.asarray(all_diag_lcb, dtype=np.float64),
    "mu_peak_thr": float(mu_peak_thr),
    "MU_PEAK_Q": float(MU_PEAK_Q),
    "LCB_TARGET": float(LCB_TARGET),
    "LCB_PEAK_Q": float(LCB_PEAK_Q),
    "SIGMA_REF_Q75": float(SIGMA_REF_Q75),
    "T_MC_IN_MCMC": int(T_MC_IN_MCMC),
    "PROP_SCALE": float(PROP_SCALE),
    "P_GLOBAL": float(P_GLOBAL),
    "P_SMALL": float(P_SMALL),
    "SMALL_SCALE": float(SMALL_SCALE),
    "REPULSION_M": int(REPULSION_M),
    "REPULSION_GAMMA": float(REPULSION_GAMMA),
    "D2_FLOOR": float(D2_FLOOR),
    "DESIGN_LOAD_VALUE_raw": float(DESIGN_LOAD_VALUE),
})
mcmc_diag_df.to_csv("mcmc_latent_diagnostics_cheap.csv", index=False)
print("Saved: mcmc_latent_diagnostics_cheap.csv")


# ------------------------------------------
# Inverse Design loop after sampling
# ------------------------------------------
def _seed_stream(base_seed: int, n: int) -> np.ndarray:
    ss = np.random.SeedSequence(int(base_seed))
    return ss.generate_state(int(n), dtype=np.uint32)

# MC-dropout inference
def mc_dropout_samples_scaled(
    model: tf.keras.Model,
    X_comp: np.ndarray,
    X_load_scaled: np.ndarray,
    num_samples: int = 200,
    batch_size: int = 512,
    seed: int = 0,
) -> np.ndarray:
    """
    Returns MC samples in *scaled-y space* of shape (S, N).
    BN-safe: training=False + toggled MCDropout via set_mc_dropout_inv.
    Uses a seed stream: independent draws, reproducible.
    """
    X_comp = np.asarray(X_comp, dtype=np.float32)
    X_load_scaled = ensure_2d_col_inv(X_load_scaled).astype(np.float32)
    n = X_comp.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((X_comp, X_load_scaled)).batch(int(batch_size))

    mc_seeds = _seed_stream(int(seed) + 10000, int(num_samples))

    set_mc_dropout_inv(model, True)
    try:
        samples = np.empty((int(num_samples), n), dtype=np.float32)
        for s in range(int(num_samples)):
            tf.keras.utils.set_random_seed(int(mc_seeds[s]))
            preds = []
            for xb, xl in ds:
                out = model([xb, xl], training=False)
                if isinstance(out, dict):
                    yb = out["Output_Layer"]
                elif isinstance(out, (list, tuple)):
                    yb = out[0]
                else:
                    yb = out
                preds.append(tf.reshape(tf.cast(yb, tf.float32), (-1,)).numpy().astype(np.float32))
            samples[s, :] = np.concatenate(preds, axis=0)
    finally:
        set_mc_dropout_inv(model, False)

    return samples

def summarize_mc_hv(samples_scaled: np.ndarray, y_mu: float, y_sd: float):
    """
    Convert scaled-y MC samples to HV and return:
      samples_hv (S,N), mu(N), sigma(N), q10(N), q80(N)
    """
    samples_scaled = np.asarray(samples_scaled, dtype=np.float32)
    samples_hv = invert_y_to_hv(samples_scaled.reshape(-1), y_mu, y_sd).reshape(samples_scaled.shape)
    mu = samples_hv.mean(axis=0).astype(np.float32)
    sigma = samples_hv.std(axis=0).astype(np.float32)
    q10 = np.quantile(samples_hv, 0.10, axis=0).astype(np.float32)
    q80 = np.quantile(samples_hv, 0.80, axis=0).astype(np.float32)
    return samples_hv, mu, sigma, q10, q80


X_train_comp = project_comp_to_simplex_np(np.asarray(X_comp_train_fit, dtype=np.float32))

NOV_SUB = int(min(5000, X_train_comp.shape[0]))
rng_nov = np.random.default_rng(int(seed) + 44551)
nov_idx = rng_nov.choice(X_train_comp.shape[0], size=NOV_SUB, replace=False)
X_train_sub = X_train_comp[nov_idx]

def _min_dist_to_train(Xcand: np.ndarray, Xtrain: np.ndarray, batch: int = 1024) -> np.ndarray:
    Xcand = np.asarray(Xcand, dtype=np.float32)
    Xtrain = np.asarray(Xtrain, dtype=np.float32)
    out = np.empty((Xcand.shape[0],), dtype=np.float32)
    for i in range(0, Xcand.shape[0], batch):
        xb = Xcand[i:i+batch]
        d2 = np.sum((xb[:, None, :] - Xtrain[None, :, :])**2, axis=2)
        out[i:i+batch] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
    return out

def _nn1_within_set(X: np.ndarray, batch: int = 512) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    out = np.empty((n,), dtype=np.float32)

    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        xb = X[i0:i1]  # (b, d)

        d2 = np.sum((xb[:, None, :] - X[None, :, :])**2, axis=2)
        rows = np.arange(i1 - i0)
        cols = np.arange(i0, i1)
        d2[rows, cols] = np.inf

        out[i0:i1] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)

    return out

ref_nn = _nn1_within_set(X_train_sub, batch=512)
NOVELTY_Q = float(0.75)
novel_thr = float(np.quantile(ref_nn, NOVELTY_Q))

print(f"[NOVELTY] novelty_thr = q{NOVELTY_Q:.2f}(leave-one-out NN in train_sub) = {novel_thr:.4f}")
print(
    f"[NOVELTY] ref_nn stats: min/mean/max = "
    f"{float(ref_nn.min()):.4f} / {float(ref_nn.mean()):.4f} / {float(ref_nn.max()):.4f}"
)

X_comp_train_design = np.asarray(X_comp_train_fit, dtype=np.float32)
Xl_const_scaled = np.repeat(Xl_design_scaled_1x1.astype(np.float32), X_comp_train_design.shape[0], axis=0)

S_TRAIN_MC = 300
train_samples_scaled = mc_dropout_samples_scaled(
    final_model, X_comp_train_design, Xl_const_scaled,
    num_samples=S_TRAIN_MC, batch_size=512, seed=int(seed) + 11
)
train_samples_hv, mu_train_mc_hv, sigma_train_hv, _, _ = summarize_mc_hv(
    train_samples_scaled, y_mu_final, y_sd_final
)

z_conf_seed = float(ZCONF_FINAL)
lcb_train_hv = (mu_train_mc_hv - z_conf_seed * sigma_train_hv).astype(np.float32)

MU_TARGET_Q_DIAG = float(0.99)
MU_TARGET_DIAG = float(np.quantile(mu_train_mc_hv, MU_TARGET_Q_DIAG))
print(f"[TRAIN@1N diag] MU_TARGET_DIAG = q{MU_TARGET_Q_DIAG:.2f}(mu) = {MU_TARGET_DIAG:.2f} HV")
print(f"[TRAIN@1N]      LCB_TARGET     = {float(LCB_TARGET):.2f} HV (from Block-1 calibration)")


# Decode MCMC latents -> compositions
set_mc_dropout_inv(final_model, False)
comp_pred = decoder_model.predict(sampled_latent_vectors.astype(np.float32), verbose=0)
comp_pred = project_comp_to_simplex_np(comp_pred.astype(np.float32))

comp_std = float(np.mean(np.std(comp_pred, axis=0)))
print(f"[DECODER DIAG] mean(std over elements) = {comp_std:.6e}")

DEDUP_DECIMALS = int(4)
comp_round = np.round(comp_pred, DEDUP_DECIMALS)
_, uniq_idx = np.unique(comp_round, axis=0, return_index=True)
uniq_idx = np.sort(uniq_idx)

if uniq_idx.size < comp_pred.shape[0]:
    print(f"[DEDUP] unique compositions: {uniq_idx.size} / {comp_pred.shape[0]} (after rounding {DEDUP_DECIMALS} d.p.)")

comp_pred_u = comp_pred[uniq_idx]
z_u = sampled_latent_vectors[uniq_idx]
latent_component_ids_u = latent_component_ids[uniq_idx]


# Candidate MC evaluation
X_load_design_raw = np.full((comp_pred_u.shape[0], 1), float(DESIGN_LOAD_VALUE), dtype=np.float32)
X_load_design_scaled = scale_load_np_inv(X_load_design_raw, scaler_load_final)

USE_PRESCREEN = True
S_CAND_FAST = 120
S_CAND_FULL = 500

if USE_PRESCREEN:
    cand_samples_scaled_fast = mc_dropout_samples_scaled(
        final_model, comp_pred_u, X_load_design_scaled,
        num_samples=S_CAND_FAST, batch_size=512, seed=int(seed) + 21
    )
    _, mu_fast, sig_fast, _, _ = summarize_mc_hv(cand_samples_scaled_fast, y_mu_final, y_sd_final)
    lcb_fast = (mu_fast - z_conf_seed * sig_fast).astype(np.float32)

    KEEP_FULL = int(min(400, comp_pred_u.shape[0]))
    keep_idx_full = np.argsort(-lcb_fast)[:KEEP_FULL]
else:
    keep_idx_full = np.arange(comp_pred_u.shape[0], dtype=int)

comp_eval = comp_pred_u[keep_idx_full]
z_eval = z_u[keep_idx_full]
latent_component_ids_eval = latent_component_ids_u[keep_idx_full]
X_load_eval = X_load_design_scaled[keep_idx_full]

cand_samples_scaled = mc_dropout_samples_scaled(
    final_model, comp_eval, X_load_eval,
    num_samples=S_CAND_FULL, batch_size=512, seed=int(seed) + 22
)
cand_samples_hv, mu_cand_hv, sigma_cand_hv, q10_cand_hv, q80_cand_hv = summarize_mc_hv(
    cand_samples_scaled, y_mu_final, y_sd_final
)

# Candidate diagnostics/gates
cand_logp = _log_gmm_pdf(gmm_latent_train, z_eval.astype(np.float64)).astype(np.float64)
lcb_cand_hv = (mu_cand_hv - z_conf_seed * sigma_cand_hv).astype(np.float32)

novelty_dist = _min_dist_to_train(comp_eval, X_train_sub, batch=1024)
print(
    f"[NOVELTY CHECK] cand novelty min/mean/max = "
    f"{float(novelty_dist.min()):.4f} / {float(novelty_dist.mean()):.4f} / {float(novelty_dist.max()):.4f}"
)

def _greedy_diverse_select(score: np.ndarray, X: np.ndarray, k: int, min_sep: float) -> np.ndarray:
    """
    Greedy selection: pick best score, then enforce a minimum separation in composition space.
    Returns indices into X.
    """
    score = np.asarray(score, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    order = np.argsort(-score)  # descending
    chosen = []
    for idx in order:
        if len(chosen) >= int(k):
            break
        if len(chosen) == 0:
            chosen.append(int(idx))
            continue
        d = np.sqrt(np.sum((X[chosen] - X[idx])**2, axis=1))
        if float(np.min(d)) >= float(min_sep):
            chosen.append(int(idx))
    return np.asarray(chosen, dtype=int)

SIGMA_MAX = float(2.5 * SIGMA_REF_Q75)
sigma_ok = (sigma_cand_hv <= SIGMA_MAX)
logp_ok = (cand_logp >= float(logp_thr))
nov_ok = (novelty_dist >= float(novel_thr))
accept_mask = (logp_ok & nov_ok & sigma_ok)

sigma_pen = np.maximum(0.0, (sigma_cand_hv - SIGMA_REF_Q75)).astype(np.float32)
soft_score = (
    lcb_cand_hv
    + 250.0 * novelty_dist.astype(np.float32)
    + 0.03 * cand_logp.astype(np.float32)
    - 0.20 * (sigma_pen ** 2)
).astype(np.float32)

n_all = int(accept_mask.shape[0])
n_logp = int(np.sum(logp_ok))
n_nov  = int(np.sum(nov_ok))
n_sig  = int(np.sum(sigma_ok))
n_acc  = int(np.sum(accept_mask))
print(f"[GATES] N={n_all} | logp_ok={n_logp} | nov_ok={n_nov} | sigma_ok={n_sig} | accepted(all three)={n_acc}")

TOPK_EXPORT = int(200)
TOPK_DIVERSE = int(min(40, TOPK_EXPORT))

MIN_SEP_BASE = float(0.50 * novel_thr)
if n_acc == 0:
    MIN_SEP = float(0.25 * novel_thr)
else:
    MIN_SEP = MIN_SEP_BASE

df = pd.DataFrame(comp_eval, columns=composition_cols)
df["design_load_raw"] = float(DESIGN_LOAD_VALUE)

df["HV_mu"] = mu_cand_hv
df["HV_sigma"] = sigma_cand_hv
df["HV_q10"] = q10_cand_hv
df["HV_q80"] = q80_cand_hv
df["HV_LCB"] = lcb_cand_hv

df["logp_gmm"] = cand_logp.astype(np.float64)
df["logp_thr"] = float(logp_thr)

df["novelty_dist_to_train_sub"] = novelty_dist.astype(np.float32)
df["novelty_thr"] = float(novel_thr)
df["novelty_q_used"] = float(NOVELTY_Q)

df["LCB_TARGET"] = float(LCB_TARGET)
df["z_conf_seed"] = float(z_conf_seed)
df["MU_TARGET_DIAG"] = float(MU_TARGET_DIAG)
df["MU_TARGET_Q_DIAG"] = float(MU_TARGET_Q_DIAG)

df["gmm_component_seed"] = latent_component_ids_eval.astype(int)
df["accepted"] = accept_mask.astype(int)
df["sigma_ref_q75_train"] = float(SIGMA_REF_Q75)
df["sigma_max_gate"] = float(SIGMA_MAX)
df["soft_score"] = soft_score

df["dedup_pool_index"] = keep_idx_full.astype(int)
df["orig_mcmc_index"] = uniq_idx[keep_idx_full].astype(int)

df = df.sort_values(
    ["accepted", "soft_score", "HV_LCB", "logp_gmm"],
    ascending=[False, False, False, False]
).reset_index(drop=True)

df["diverse_pick"] = 0

if int(np.sum(df["accepted"].values)) > 0:
    acc_df = df[df["accepted"] == 1].copy()
    X_acc = acc_df[composition_cols].values.astype(np.float32)
    score_acc = acc_df["soft_score"].values.astype(np.float32)
    pick_local = _greedy_diverse_select(score_acc, X_acc, k=TOPK_DIVERSE, min_sep=MIN_SEP)
    diverse_global_idx = acc_df.index.values[pick_local]
    df.loc[diverse_global_idx, "diverse_pick"] = 1
else:
    top_df = df.head(TOPK_EXPORT).copy()
    X_top = top_df[composition_cols].values.astype(np.float32)
    score_top = top_df["soft_score"].values.astype(np.float32)
    pick_local = _greedy_diverse_select(score_top, X_top, k=TOPK_DIVERSE, min_sep=MIN_SEP)
    diverse_global_idx = top_df.index.values[pick_local]
    df.loc[diverse_global_idx, "diverse_pick"] = 1
    print("[DIVERSE] No accepted candidates; diverse_pick computed on top soft_score as fallback (inspect gates).")

csv_filename = "inverse_design_candidates_FINAL.csv"
df.head(TOPK_EXPORT).to_csv(csv_filename, index=False)

y_train_hv = np.asarray(y_train_fit, dtype=np.float32).reshape(-1)

print("TRAIN LABEL HV: min/mean/max =", float(y_train_hv.min()), float(y_train_hv.mean()), float(y_train_hv.max()))
print("TRAIN MC mu@1N: min/mean/max =", float(mu_train_mc_hv.min()), float(mu_train_mc_hv.mean()), float(mu_train_mc_hv.max()))
print("TRAIN MC LCB@1N: min/mean/max =", float(lcb_train_hv.min()), float(lcb_train_hv.mean()), float(lcb_train_hv.max()))
print("LCB_TARGET (near-peak@1N) =", float(LCB_TARGET))

print("CAND MC mu (HV): min/mean/max =", float(mu_cand_hv.min()), float(mu_cand_hv.mean()), float(mu_cand_hv.max()))
print("CAND MC sigma (HV): min/mean/max =", float(sigma_cand_hv.min()), float(sigma_cand_hv.mean()), float(sigma_cand_hv.max()))
print("CAND LCB (HV): min/mean/max =", float(lcb_cand_hv.min()), float(lcb_cand_hv.mean()), float(lcb_cand_hv.max()))
print("CAND novelty_dist: min/mean/max =", float(novelty_dist.min()), float(novelty_dist.mean()), float(novelty_dist.max()))
print("SIGMA_REF_Q75 (train@1N) =", float(SIGMA_REF_Q75), "| SIGMA_MAX gate =", float(SIGMA_MAX))
print("GATE COUNTS: logp_ok / nov_ok / sigma_ok / accepted =", n_logp, n_nov, n_sig, n_acc)

print(f"Saved: {csv_filename} (top {TOPK_EXPORT} rows)")
print(f"Accepted (logp + novelty + sigma): {n_acc} / {n_all}")
print(
    f"[INV-DES ACCEPT] logp>=weighted q{LOGP_Q:.2f}, "
    f"novelty>=q{NOVELTY_Q:.2f}(train LOO NN spacing), "
    f"sigma<=2.5*sigma_ref_q75"
)


# =====================================
# Gradient-Based Optimization
# =====================================
best_loss = float("inf")
no_improvement_counter = 0

min_hardness = 2300.0
max_hardness = 2600.0
target_hardness = 0.5 * (min_hardness + max_hardness)

num_new_alloys = 5
learning_rate = 3e-4
num_iterations = 100000

ACQ_MODE = "lcb"
z_conf = float(ZCONF_FINAL)

lambda_norm   = 0.01
lambda_band   = 15.0
lambda_center = 0.5
lambda_prior  = 0.10

lambda_sigma_soft = 1.00
lambda_seed_pull  = 0.10

lambda_repulse_latent = 0.10
lambda_repulse_comp   = 0.10

lambda_band_max = float(lambda_band)
lambda_band_warmup = 500

T_mc = 24
SIGMA_EPS = 1e-6
K_SIGMA_ALLOW = 2.50
sigma_ref = float(SIGMA_REF_Q75)

USE_SOFT_LOGP_BARRIER = True
USE_SOFT_LCB_TARGET_BARRIER = True
lambda_logp_barrier = 2.0
lambda_lcb_target_barrier = 1.5

y_train_hv = np.asarray(y_train_fit, dtype=np.float32).reshape(-1)
HV_train_max = float(np.max(y_train_hv))
HV_TRUST_MARGIN = 250.0
HV_CAP = float(HV_train_max + HV_TRUST_MARGIN)

_required = [
    "decoder_model", "final_model", "sampled_latent_vectors", "scaler_load_final",
    "y_mu_final", "y_sd_final", "DESIGN_LOAD_VALUE", "set_mc_dropout_inv",
    "project_comp_to_simplex_np", "scale_load_np_inv", "ensure_2d_col_inv",
    "invert_y_to_hv", "composition_cols", "mu_k_tf", "cov_k_inv_tf",
    "gmm_latent_train", "_log_gmm_pdf", "logp_thr", "seed",
    "LCB_TARGET", "SIGMA_REF_Q75",
]
for _n in _required:
    if _n not in globals():
        raise RuntimeError(f"Missing required object/function: {_n}")

_test_out = decoder_model(tf.zeros((1, sampled_latent_vectors.shape[1]), dtype=tf.float32), training=False)
if isinstance(_test_out, (list, tuple)):
    raise RuntimeError("decoder_model must be comp-only (single tensor Comp_Recon).")

mean_tf  = tf.constant(np.asarray(scaler_load_final.mean_,  dtype=np.float32).reshape(1, 1), dtype=tf.float32)
scale_tf = tf.constant(np.asarray(scaler_load_final.scale_, dtype=np.float32).reshape(1, 1), dtype=tf.float32)

def scale_load_tf(load_raw: tf.Tensor) -> tf.Tensor:
    load_raw = tf.cast(load_raw, tf.float32)
    return (load_raw - mean_tf) / scale_tf

y_mu_tf = tf.constant(float(y_mu_final), dtype=tf.float32)
y_sd_tf = tf.constant(float(y_sd_final), dtype=tf.float32)

def yscaled_to_hv_tf(y_scaled: tf.Tensor) -> tf.Tensor:
    y_scaled = tf.cast(y_scaled, tf.float32)
    return y_scaled * y_sd_tf + y_mu_tf

def mahalanobis_penalty(z_batch: tf.Tensor, mu_tf: tf.Tensor, cov_inv_tf: tf.Tensor) -> tf.Tensor:
    z_batch = tf.cast(z_batch, tf.float32)
    dz = z_batch - tf.cast(mu_tf, tf.float32)
    cov_inv_tf = tf.cast(cov_inv_tf, tf.float32)
    maha = tf.einsum("bi,ij,bj->b", dz, cov_inv_tf, dz)
    return tf.reduce_mean(maha)

def _seed_stream(base_seed: int, n: int) -> np.ndarray:
    ss = np.random.SeedSequence(int(base_seed))
    return ss.generate_state(int(n), dtype=np.uint32)

def mc_dropout_eval_pool_hv_np(
    model: tf.keras.Model,
    Xc: np.ndarray,
    Xl_scaled: np.ndarray,
    S: int,
    seed: int,
    batch_size: int = 512,
):
    """
    Cheap BN-safe MC ranking stats: returns (mu_hv, sigma_hv, lcb_hv).
    Uses a seed stream for reproducibility.
    """
    Xc = np.asarray(Xc, dtype=np.float32)
    Xl_scaled = ensure_2d_col_inv(np.asarray(Xl_scaled, dtype=np.float32)).astype(np.float32)

    n = Xc.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((Xc, Xl_scaled)).batch(int(batch_size))
    samples_hv = np.empty((int(S), n), dtype=np.float32)

    mc_seeds = _seed_stream(int(seed) + 50000, int(S))

    set_mc_dropout_inv(model, True)
    try:
        for s in range(int(S)):
            tf.keras.utils.set_random_seed(int(mc_seeds[s]))
            preds = []
            for xb, xl in ds:
                out = model([xb, xl], training=False)
                if isinstance(out, dict):
                    yb = out["Output_Layer"]
                elif isinstance(out, (list, tuple)):
                    yb = out[0]
                else:
                    yb = out
                yb = tf.reshape(tf.cast(yb, tf.float32), (-1,)).numpy().astype(np.float32)
                preds.append(yb)
            y_scaled = np.concatenate(preds, axis=0).astype(np.float32)
            y_hv = invert_y_to_hv(y_scaled, y_mu_final, y_sd_final).astype(np.float32)
            samples_hv[s, :] = y_hv
    finally:
        set_mc_dropout_inv(model, False)

    mu = samples_hv.mean(axis=0).astype(np.float32)
    sigma = np.maximum(samples_hv.std(axis=0).astype(np.float32), float(SIGMA_EPS)).astype(np.float32)
    lcb = (mu - float(z_conf) * sigma).astype(np.float32)
    return mu, sigma, lcb


Z_pool = np.asarray(sampled_latent_vectors, dtype=np.float32)

comp_pool = decoder_model.predict(Z_pool, verbose=0)
comp_pool = project_comp_to_simplex_np(np.asarray(comp_pool, dtype=np.float32))

DEDUP_DECIMALS = 4
comp_round = np.round(comp_pool, DEDUP_DECIMALS)
_, uniq_idx = np.unique(comp_round, axis=0, return_index=True)
uniq_idx = np.sort(uniq_idx)

Z_pool_u = Z_pool[uniq_idx]
comp_pool_u = comp_pool[uniq_idx]

X_load_pool_raw = np.full((comp_pool_u.shape[0], 1), float(DESIGN_LOAD_VALUE), dtype=np.float32)
X_load_pool_scaled = ensure_2d_col_inv(scale_load_np_inv(X_load_pool_raw, scaler_load_final)).astype(np.float32)

pool_logp = _log_gmm_pdf(gmm_latent_train, Z_pool_u.astype(np.float64)).astype(np.float64)
feasible_mask = (pool_logp >= float(logp_thr))

mu_seed, sig_seed, lcb_seed = mc_dropout_eval_pool_hv_np(
    final_model, comp_pool_u, X_load_pool_scaled, S=64, seed=int(seed) + 99, batch_size=512
)

seed_mask = feasible_mask & (lcb_seed >= float(LCB_TARGET))
if int(seed_mask.sum()) < int(num_new_alloys):
    seed_mask = feasible_mask.copy()
if int(seed_mask.sum()) < int(num_new_alloys):
    seed_mask = np.ones_like(seed_mask, dtype=bool)

seed_score = (lcb_seed + 0.03 * pool_logp.astype(np.float32)).astype(np.float32)
seed_score_rank = seed_score.copy()
seed_score_rank[~seed_mask] = -np.inf

def _greedy_diverse_select(score: np.ndarray, X: np.ndarray, k: int, min_sep: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    order = np.argsort(-score, kind="mergesort")
    chosen = []
    for idx in order:
        if len(chosen) >= int(k):
            break
        if len(chosen) == 0:
            chosen.append(int(idx))
            continue
        d = np.sqrt(np.sum((X[chosen] - X[idx])**2, axis=1))
        if float(np.min(d)) >= float(min_sep):
            chosen.append(int(idx))
    return np.asarray(chosen, dtype=int)

try:
    MIN_SEP_SEED = float(0.50 * float(novel_thr))
except Exception:
    MIN_SEP_SEED = 0.25

cand_idx = np.where(np.isfinite(seed_score_rank))[0]
if cand_idx.size == 0:
    cand_idx = np.arange(seed_score_rank.shape[0])

pick_local = _greedy_diverse_select(seed_score_rank[cand_idx], comp_pool_u[cand_idx], k=int(num_new_alloys), min_sep=MIN_SEP_SEED)
top_idx_local = cand_idx[pick_local]

if top_idx_local.size < int(num_new_alloys):
    order = np.argsort(-seed_score_rank, kind="mergesort")
    topup = []
    top_set = set(top_idx_local.tolist())
    for idx in order:
        if len(topup) + int(top_idx_local.size) >= int(num_new_alloys):
            break
        if int(idx) in top_set:
            continue
        topup.append(int(idx))
    top_idx_local = np.concatenate([top_idx_local, np.asarray(topup, dtype=int)], axis=0)[:int(num_new_alloys)]

initial_latent_vectors = Z_pool_u[top_idx_local, :].astype(np.float32)
seed_latents_tf = tf.constant(initial_latent_vectors, dtype=tf.float32)

print(
    f"[GRAD INIT] seeds={int(num_new_alloys)} | "
    f"LCB(mean)={float(lcb_seed[top_idx_local].mean()):.2f} | "
    f"mu(mean)={float(mu_seed[top_idx_local].mean()):.2f} | "
    f"sigma(mean)={float(sig_seed[top_idx_local].mean()):.2f} | "
    f"logp(mean)={float(pool_logp[top_idx_local].mean()):.3f} | "
    f"HV_CAP={HV_CAP:.1f} (train_max + margin)"
)


latent_vectors_tf = tf.Variable(initial_latent_vectors, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
best_latent_vectors = None

def pairwise_sq_dists(X: tf.Tensor) -> tf.Tensor:
    X = tf.cast(X, tf.float32)
    x2 = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)
    d2 = x2 - 2.0 * tf.matmul(X, X, transpose_b=True) + tf.transpose(x2)
    return tf.maximum(d2, 0.0)

LATENT_MIN_SEP = 0.75
COMP_MIN_SEP   = float(MIN_SEP_SEED)

def repulsion_penalty(X: tf.Tensor, min_sep: float) -> tf.Tensor:
    d2 = pairwise_sq_dists(X)
    b = tf.shape(d2)[0]
    mask = tf.ones_like(d2) - tf.eye(b, dtype=tf.float32)
    d = tf.sqrt(d2 + 1e-12) * mask
    viol = tf.nn.relu(tf.cast(min_sep, tf.float32) - d)
    denom = tf.maximum(1.0, tf.reduce_sum(mask))
    return tf.reduce_sum(viol) / denom

def manifold_barrier_proxy(z_batch: tf.Tensor, mu_tf: tf.Tensor, cov_inv_tf: tf.Tensor) -> tf.Tensor:
    return mahalanobis_penalty(z_batch, mu_tf, cov_inv_tf)

GRAD_CLIP_NORM = 5.0

# Optimization loop
for iteration in range(int(num_iterations)):
    mc_seeds_iter = _seed_stream(int(seed) + 3000000 + iteration, int(T_mc))

    with tf.GradientTape() as tape:
        comp_logits = decoder_model(latent_vectors_tf, training=False)
        comp_logits = tf.cast(comp_logits, tf.float32)
        comp_pos = tf.nn.softplus(comp_logits)
        comp_sum = tf.reduce_sum(comp_pos, axis=1, keepdims=True)
        comp_pred_tf = comp_pos / tf.maximum(comp_sum, tf.constant(1e-8, tf.float32))

        bsz = tf.shape(comp_pred_tf)[0]
        load_raw = tf.fill([bsz, 1], tf.constant(float(DESIGN_LOAD_VALUE), dtype=tf.float32))
        load_scaled = scale_load_tf(load_raw)

        set_mc_dropout_inv(final_model, True)
        y_mc_hv = []
        for s in range(int(T_mc)):
            tf.keras.utils.set_random_seed(int(mc_seeds_iter[s]))
            out = final_model([comp_pred_tf, load_scaled], training=False)
            if isinstance(out, dict):
                y_s = out["Output_Layer"]
            elif isinstance(out, (list, tuple)):
                y_s = out[0]
            else:
                y_s = out
            y_s = tf.reshape(tf.cast(y_s, tf.float32), (-1,))
            y_mc_hv.append(yscaled_to_hv_tf(y_s))
        y_mc_hv = tf.stack(y_mc_hv, axis=0)  # (T_mc, batch)
        set_mc_dropout_inv(final_model, False)

        hv_mu = tf.reduce_mean(y_mc_hv, axis=0)
        hv_sigma = tf.maximum(tf.math.reduce_std(y_mc_hv, axis=0), tf.constant(SIGMA_EPS, dtype=tf.float32))

        hv_lcb = hv_mu - float(z_conf) * hv_sigma
        if ACQ_MODE != "lcb":
            raise ValueError("Configured for LCB optimization only.")
        utility_loss = -tf.reduce_mean(hv_lcb)

        lower_violation = tf.nn.relu(tf.constant(min_hardness, tf.float32) - hv_lcb)
        upper_violation = tf.nn.relu(hv_lcb - tf.constant(max_hardness, tf.float32))
        band_penalty = tf.reduce_mean(lower_violation + upper_violation)

        band_scale = tf.minimum(
            1.0,
            tf.cast(iteration, tf.float32) / tf.constant(float(lambda_band_warmup), tf.float32)
        )

        center_loss = tf.reduce_mean(tf.square(hv_lcb - tf.constant(target_hardness, tf.float32)))
        hv_cap_pen = tf.reduce_mean(tf.nn.relu(hv_mu - tf.constant(HV_CAP, tf.float32)))
        latent_norm = tf.reduce_mean(tf.square(latent_vectors_tf))
        prior_pen = mahalanobis_penalty(latent_vectors_tf, mu_k_tf, cov_k_inv_tf)
        seed_pull = tf.reduce_mean(tf.square(latent_vectors_tf - seed_latents_tf))

        sigma_soft_pen = tf.reduce_mean(
            tf.nn.relu(hv_sigma - tf.constant(float(K_SIGMA_ALLOW * sigma_ref), tf.float32))
        )

        repulse_z = repulsion_penalty(latent_vectors_tf, min_sep=LATENT_MIN_SEP)
        repulse_x = repulsion_penalty(comp_pred_tf,     min_sep=COMP_MIN_SEP)

        logp_barrier = tf.constant(0.0, tf.float32)
        if USE_SOFT_LOGP_BARRIER:
            logp_barrier = manifold_barrier_proxy(latent_vectors_tf, mu_k_tf, cov_k_inv_tf)

        lcb_target_barrier = tf.constant(0.0, tf.float32)
        if USE_SOFT_LCB_TARGET_BARRIER:
            lcb_target_barrier = tf.reduce_mean(tf.nn.relu(tf.constant(float(LCB_TARGET), tf.float32) - hv_lcb))

        loss = (
            utility_loss
            + (lambda_band_max * band_scale) * band_penalty
            + float(lambda_center) * center_loss
            + float(lambda_norm) * latent_norm
            + float(lambda_prior) * prior_pen
            + float(lambda_seed_pull) * seed_pull
            + float(lambda_sigma_soft) * sigma_soft_pen
            + 1.0 * hv_cap_pen
            + float(lambda_repulse_latent) * repulse_z
            + float(lambda_repulse_comp)   * repulse_x
            + float(lambda_logp_barrier) * tf.cast(logp_barrier, tf.float32)
            + float(lambda_lcb_target_barrier) * tf.cast(lcb_target_barrier, tf.float32)
        )

    grads = tape.gradient(loss, [latent_vectors_tf])[0]
    if grads is None:
        raise RuntimeError("Gradients are None. Check that final_model depends on latent via decoder->comp->model.")
    grads = tf.clip_by_norm(grads, GRAD_CLIP_NORM)
    optimizer.apply_gradients([(grads, latent_vectors_tf)])

    current_loss = float(loss.numpy())
    if np.isfinite(current_loss) and current_loss < best_loss:
        best_loss = current_loss
        best_latent_vectors = latent_vectors_tf.numpy().copy()
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1

    if iteration % 200 == 0 or iteration == int(num_iterations) - 1:
        print(
            f"Iter {iteration:5d}: "
            f"Loss={float(loss.numpy()):.4f}, "
            f"U_loss={float(utility_loss.numpy()):.4f}, "
            f"BandPen={float(band_penalty.numpy()):.4f}, "
            f"BandScale={float(band_scale.numpy()):.3f}, "
            f"PriorPen={float(prior_pen.numpy()):.4f}, "
            f"SeedPull={float(seed_pull.numpy()):.4f}, "
            f"SigmaSoft={float(sigma_soft_pen.numpy()):.4f}, "
            f"RepZ={float(repulse_z.numpy()):.4f}, "
            f"RepX={float(repulse_x.numpy()):.4f}, "
            f"CapPen={float(hv_cap_pen.numpy()):.4f}, "
            f"LCB(mean)={float(tf.reduce_mean(hv_lcb).numpy()):.2f}, "
            f"mu(mean)={float(tf.reduce_mean(hv_mu).numpy()):.2f}, "
            f"sigma(mean)={float(tf.reduce_mean(hv_sigma).numpy()):.2f}, "
            f"LCB_target_bar={float(lcb_target_barrier.numpy()):.4f}, "
            f"logp_proxy={float(logp_barrier.numpy()):.4f}"
        )

if best_latent_vectors is None:
    raise RuntimeError("Optimization produced no valid improvement; best_latent_vectors is None.")


optimized_latent_vectors = np.asarray(best_latent_vectors, dtype=np.float32)
opt_comp = decoder_model.predict(optimized_latent_vectors, verbose=0).astype(np.float32)
opt_comp = project_comp_to_simplex_np(opt_comp)

X_load_opt_raw = np.full((opt_comp.shape[0], 1), float(DESIGN_LOAD_VALUE), dtype=np.float32)
X_load_opt_scaled = scale_load_np_inv(X_load_opt_raw, scaler_load_final)

def mc_dropout_eval_np_hv(
    model: tf.keras.Model,
    Xc: np.ndarray,
    Xl_scaled: np.ndarray,
    S: int = 500,
    seed: int = 0,
    batch_size: int = 512,
):
    Xc = np.asarray(Xc, dtype=np.float32)
    Xl_scaled = ensure_2d_col_inv(np.asarray(Xl_scaled, dtype=np.float32))
    n = Xc.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((Xc, Xl_scaled)).batch(int(batch_size))
    samples = np.empty((int(S), n), dtype=np.float32)

    mc_seeds = _seed_stream(int(seed) + 20000, int(S))

    set_mc_dropout_inv(model, True)
    try:
        for s in range(int(S)):
            tf.keras.utils.set_random_seed(int(mc_seeds[s]))
            preds = []
            for xb, xl in ds:
                out = model([xb, xl], training=False)
                if isinstance(out, dict):
                    yb = out["Output_Layer"]
                elif isinstance(out, (list, tuple)):
                    yb = out[0]
                else:
                    yb = out
                yb = tf.reshape(tf.cast(yb, tf.float32), (-1,))
                yb_hv = invert_y_to_hv(yb.numpy().astype(np.float32), y_mu_final, y_sd_final)
                preds.append(yb_hv.astype(np.float32))
            samples[s, :] = np.concatenate(preds, axis=0)
    finally:
        set_mc_dropout_inv(model, False)

    mu = samples.mean(axis=0).astype(np.float32)
    sigma = samples.std(axis=0).astype(np.float32)
    q10 = np.quantile(samples, 0.10, axis=0).astype(np.float32)
    q85 = np.quantile(samples, 0.85, axis=0).astype(np.float32)
    return samples, mu, sigma, q10, q85

S_opt, mu_opt, sigma_opt, q10_opt, q85_opt = mc_dropout_eval_np_hv(
    final_model, opt_comp, X_load_opt_scaled, S=500, seed=int(seed) + 777, batch_size=512
)

lcb_opt = mu_opt - float(z_conf) * sigma_opt
gmm_logp_opt = _log_gmm_pdf(gmm_latent_train, optimized_latent_vectors.astype(np.float64)).astype(np.float64)

valid_indices = (
    (gmm_logp_opt >= float(logp_thr)) &
    (sigma_opt <= float(K_SIGMA_ALLOW * sigma_ref)) &
    (lcb_opt >= float(min_hardness)) &
    (lcb_opt <= float(max_hardness)) &
    (mu_opt <= float(HV_CAP))
)

df_opt = pd.DataFrame(opt_comp, columns=composition_cols)
df_opt["design_load_raw"] = float(DESIGN_LOAD_VALUE)
df_opt["HV_mu"] = mu_opt
df_opt["HV_sigma"] = sigma_opt
df_opt["HV_q10"] = q10_opt
df_opt["HV_q85"] = q85_opt
df_opt["HV_lcb"] = lcb_opt
df_opt["accepted_LCB_in_band"] = valid_indices.astype(int)
df_opt["logp_gmm"] = gmm_logp_opt
df_opt["logp_thr"] = float(logp_thr)
df_opt["z_conf"] = float(z_conf)

df_opt["LCB_TARGET_train_near_peak_1N"] = float(LCB_TARGET)
df_opt["SIGMA_REF_Q75_train_1N"] = float(sigma_ref)
df_opt["SIGMA_ALLOW_cap"] = float(K_SIGMA_ALLOW * sigma_ref)
df_opt["TRAIN_HV_max_label_global"] = float(HV_train_max)
df_opt["HV_CAP_train_max_plus_margin"] = float(HV_CAP)

df_opt = df_opt.sort_values(
    ["accepted_LCB_in_band", "HV_lcb", "logp_gmm"],
    ascending=[False, False, False]
).reset_index(drop=True)

df_opt.to_csv("optimized_candidates_with_uncertainty.csv", index=False)
print("Saved: optimized_candidates_with_uncertainty.csv")
print(
    f"Accepted (LCB-in-band + logp>=thr + sigma<= {K_SIGMA_ALLOW:.2f}*SIGMA_REF_Q75@1N + mu<=HV_CAP): "
    f"{int(valid_indices.sum())} / {len(valid_indices)}"
)
print(
    f"[OPT SUMMARY] LCB(min/mean/max)={float(lcb_opt.min()):.2f}/{float(lcb_opt.mean()):.2f}/{float(lcb_opt.max()):.2f} | "
    f"mu(min/mean/max)={float(mu_opt.min()):.2f}/{float(mu_opt.mean()):.2f}/{float(mu_opt.max()):.2f} | "
    f"sigma(mean)={float(sigma_opt.mean()):.2f} | "
    f"logp(min/mean/max)={float(gmm_logp_opt.min()):.2f}/{float(gmm_logp_opt.mean()):.2f}/{float(gmm_logp_opt.max()):.2f}"
)


# =============================================================================
# Latent-space visualization AFTER inverse-design + gradient optimization
# =============================================================================
required = [
    "latent_vectors_tr",
    "sampled_latent_vectors",
    "optimized_latent_vectors",
    "decoder_model",
    "final_model",
    "scaler_load_final",
    "composition_cols",
    "seed",
    "y_mu_final", "y_sd_final",
    "set_mc_dropout_inv",
]
for name in required:
    if name not in globals():
        raise RuntimeError(f"Missing required object: {name}")

HAS_Y_TRUE = ("y_tr_full" in globals())
DESIGN_LOAD_VALUE = 1.0

def yscaled_to_hv_np(y_scaled: np.ndarray) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    return (y_scaled * float(y_sd_final) + float(y_mu_final)).astype(np.float32)

def mc_mean_hv_for_compositions(
    model,
    Xc: np.ndarray,
    Xl_scaled: np.ndarray,
    S: int = 200,
    seed: int = 0,
    batch_size: int = 512,
):
    Xc = np.asarray(Xc, dtype=np.float32)
    Xl_scaled = np.asarray(Xl_scaled, dtype=np.float32)
    n = Xc.shape[0]

    ds = tf.data.Dataset.from_tensor_slices((Xc, Xl_scaled)).batch(batch_size)
    samples = np.empty((S, n), dtype=np.float32)

    set_mc_dropout_inv(model, True)
    try:
        for s in range(int(S)):
            tf.keras.utils.set_random_seed(int(seed) + 20200 + s)
            preds = []
            for xb, xl in ds:
                out = model([xb, xl], training=False)
                if isinstance(out, dict):
                    yb = out["Output_Layer"]
                elif isinstance(out, (list, tuple)):
                    yb = out[0]
                else:
                    yb = out
                
                yb = tf.reshape(tf.cast(yb, tf.float32), (-1,)).numpy().astype(np.float32)
                preds.append(yscaled_to_hv_np(yb))
            samples[s, :] = np.concatenate(preds, axis=0)
    finally:
        set_mc_dropout_inv(model, False)

    mu = samples.mean(axis=0).astype(np.float32)
    sigma = samples.std(axis=0).astype(np.float32)
    return mu, sigma

def decode_latents_to_comp(Z: np.ndarray) -> np.ndarray:
    """
    Decode latent vectors to compositions using the *comp-only* decoder_model
    built in the preceding blocks:
        decoder_model = Model(latent_in -> Comp_Recon)
    So decoder_model.predict(Z) returns a single array, not (comp, aux).
    """
    Z = np.asarray(Z, dtype=np.float32)
    comp = decoder_model.predict(Z, verbose=0)
    comp = np.asarray(comp, dtype=np.float32)
    comp = project_comp_to_simplex_np(comp).astype(np.float32)
    return comp

Z0 = np.asarray(latent_vectors_tr, dtype=np.float32)
Z1 = np.asarray(sampled_latent_vectors, dtype=np.float32) 
Z2 = np.asarray(optimized_latent_vectors, dtype=np.float32)

if not (Z0.ndim == Z1.ndim == Z2.ndim == 2):
    raise ValueError("All latent arrays must be 2D: (N, D).")

D = Z0.shape[1]
if (Z1.shape[1] != D) or (Z2.shape[1] != D):
    raise ValueError(f"Latent dim mismatch: Z0 D={D}, Z1 D={Z1.shape[1]}, Z2 D={Z2.shape[1]}.")

Xc1 = decode_latents_to_comp(Z1)
Xc2 = decode_latents_to_comp(Z2)

Xl1 = scale_load_np_inv(np.full((Xc1.shape[0], 1), DESIGN_LOAD_VALUE, dtype=np.float32), scaler_load_final)
Xl2 = scale_load_np_inv(np.full((Xc2.shape[0], 1), DESIGN_LOAD_VALUE, dtype=np.float32), scaler_load_final)

mu1_hv, sig1_hv = mc_mean_hv_for_compositions(final_model, Xc1, Xl1, S=300, seed=seed, batch_size=512)

if "mu_opt" in globals():
    y2 = np.asarray(mu_opt, dtype=np.float32).reshape(-1)
    if y2.shape[0] != Z2.shape[0]:
        raise ValueError(f"Mismatch: optimized Z2 N={Z2.shape[0]} vs mu_opt N={y2.shape[0]}.")
    mu2_hv = y2
    sig2_hv = None
else:
    mu2_hv, sig2_hv = mc_mean_hv_for_compositions(final_model, Xc2, Xl2, S=300, seed=seed + 7, batch_size=512)

ORIGINAL_Y_IS_SCALED = False

if HAS_Y_TRUE:
    y0_raw = np.asarray(y_tr_full, dtype=np.float32).reshape(-1)
    if y0_raw.shape[0] != Z0.shape[0]:
        raise ValueError(f"Mismatch: Z0 N={Z0.shape[0]} vs y_tr_full N={y0_raw.shape[0]}.")
    y0_hv = yscaled_to_hv_np(y0_raw) if ORIGINAL_Y_IS_SCALED else y0_raw
    sig0_hv = None
else:
    Xc0 = decode_latents_to_comp(Z0)
    Xl0 = scale_load_np_inv(np.full((Xc0.shape[0], 1), DESIGN_LOAD_VALUE, dtype=np.float32), scaler_load_final)
    y0_hv, sig0_hv = mc_mean_hv_for_compositions(final_model, Xc0, Xl0, S=200, seed=seed + 11, batch_size=512)

y0 = np.asarray(y0_hv, dtype=np.float32).reshape(-1)
y1 = np.asarray(mu1_hv, dtype=np.float32).reshape(-1)
y2 = np.asarray(mu2_hv, dtype=np.float32).reshape(-1)

if (Z0.shape[0] != y0.shape[0]) or (Z1.shape[0] != y1.shape[0]) or (Z2.shape[0] != y2.shape[0]):
    raise ValueError("Mismatch between latent counts and hardness vector lengths.")


all_latent_vectors = np.vstack([Z0, Z1, Z2]).astype(np.float32)
all_hardness_values = np.concatenate([y0, y1, y2]).astype(np.float32)

labels = np.array(
    (["Original"] * Z0.shape[0]) +
    (["Sampled"] * Z1.shape[0]) +
    (["Optimized"] * Z2.shape[0]),
    dtype=object
)

Z_std = StandardScaler(with_mean=True, with_std=True).fit_transform(all_latent_vectors).astype(np.float32)

N_total = Z_std.shape[0]
perplexity = min(30, max(5, (N_total - 1) // 10))
perplexity = float(min(perplexity, (N_total - 1) / 3.0 - 1e-6))

tsne = TSNE(
    n_components=2,
    random_state=int(seed),
    init="pca",
    learning_rate="auto",
    perplexity=perplexity,
    n_iter=2000,
)
latent_tsne = tsne.fit_transform(Z_std)

original_mask  = labels == "Original"
sampled_mask   = labels == "Sampled"
optimized_mask = labels == "Optimized"


fig, ax = plt.subplots(figsize=(12, 8))

norm = plt.Normalize(vmin=float(all_hardness_values.min()), vmax=float(all_hardness_values.max()))

sc0 = ax.scatter(
    latent_tsne[original_mask, 0], latent_tsne[original_mask, 1],
    c=all_hardness_values[original_mask],
    cmap="coolwarm", norm=norm,
    s=120, marker="o", edgecolor="k", linewidths=0.6, alpha=0.55,
    label="Original"
)

sc1 = ax.scatter(
    latent_tsne[sampled_mask, 0], latent_tsne[sampled_mask, 1],
    c=all_hardness_values[sampled_mask],
    cmap="coolwarm", norm=norm,
    s=70, marker=".", edgecolor="none", alpha=0.95,
    label="Sampled (GMM)"
)

sc2 = ax.scatter(
    latent_tsne[optimized_mask, 0], latent_tsne[optimized_mask, 1],
    c=all_hardness_values[optimized_mask],
    cmap="coolwarm", norm=norm,
    s=220, marker="*", edgecolor="k", linewidths=0.8, alpha=1.0,
    label="Optimized (grad)"
)

cbar = plt.colorbar(sc2, ax=ax)
cbar.set_label("Hardness (HV)", fontsize=16)

ax.set_xlabel("t-SNE 1", fontsize=16)
ax.set_ylabel("t-SNE 2", fontsize=16)
ax.legend(fontsize=12, frameon=True)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig("tSNE_latent_original_sampled_optimized.jpeg", dpi=600, format="jpeg")
plt.show()

tsne_df = pd.DataFrame({
    "tSNE1": latent_tsne[:, 0].astype(np.float32),
    "tSNE2": latent_tsne[:, 1].astype(np.float32),
    "label": labels,
    "HV_mu": all_hardness_values.astype(np.float32),
})

sigma_all = np.full((N_total,), np.nan, dtype=np.float32)
sigma_all[original_mask] = (sig0_hv if sig0_hv is not None else np.nan)
sigma_all[sampled_mask] = sig1_hv.astype(np.float32)
if sig2_hv is not None:
    sigma_all[optimized_mask] = sig2_hv.astype(np.float32)
tsne_df["HV_sigma"] = sigma_all

tsne_df.to_csv("tSNE_latent_original_sampled_optimized.csv", index=False)
print("Saved: tSNE_latent_original_sampled_optimized.jpeg")
print("Saved: tSNE_latent_original_sampled_optimized.csv")


# ======================================
# Export sampled + optimized subsets
# ======================================
required = ["tsne_df"]
for name in required:
    if name not in globals():
        raise RuntimeError(f"Missing required object: {name}. Run the t-SNE block first (it creates tsne_df).")

sampled_df = tsne_df.loc[tsne_df["label"] == "Sampled", ["tSNE1", "tSNE2", "HV_mu", "HV_sigma"]].copy()
sampled_df.to_csv("sampled_latent_tsne_with_hardness.csv", index=False)
optimized_df = tsne_df.loc[tsne_df["label"] == "Optimized", ["tSNE1", "tSNE2", "HV_mu", "HV_sigma"]].copy()
optimized_df.to_csv("optimized_latent_tsne_with_hardness.csv", index=False)

print("Saved: sampled_latent_tsne_with_hardness.csv")
print("Saved: optimized_latent_tsne_with_hardness.csv")


# ============================
# Novelty diagnostics
# ============================
def _require_any(names):
    for n in names:
        if n in globals():
            return n
    return None

def _ensure_2d_float32(X, name):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape={X.shape}")
    return X

def _simplex_report(X, name, tol_sum=1e-3, tol_neg=1e-6):
    X = np.asarray(X, dtype=np.float32)
    row_sum = X.sum(axis=1)
    min_val = float(X.min()) if X.size else np.nan
    max_val = float(X.max()) if X.size else np.nan
    frac_bad_sum = float(np.mean(np.abs(row_sum - 1.0) > tol_sum)) if X.size else np.nan
    frac_neg = float(np.mean(X < -tol_neg)) if X.size else np.nan
    print(f"\n[{name}] simplex check:")
    print(f"  min={min_val:.4e}, max={max_val:.4e}")
    print(f"  row_sum: min={float(row_sum.min()):.6f}, max={float(row_sum.max()):.6f}, mean={float(row_sum.mean()):.6f}")
    print(f"  frac(|sum-1|>{tol_sum})={frac_bad_sum:.4f}, frac(neg<{ -tol_neg})={frac_neg:.4f}")

def _min_l2_to_reference(ref, query, batch=2048):
    ref = np.asarray(ref, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)

    ref_norm2 = np.sum(ref * ref, axis=1, keepdims=True).T
    out = np.empty((query.shape[0],), dtype=np.float32)

    for i in range(0, query.shape[0], batch):
        q = query[i:i+batch]
        q_norm2 = np.sum(q * q, axis=1, keepdims=True)
        dist2 = q_norm2 + ref_norm2 - 2.0 * (q @ ref.T)
        dist2 = np.maximum(dist2, 0.0)
        out[i:i+batch] = np.sqrt(np.min(dist2, axis=1))
    return out

def _summarize(name, d):
    d = np.asarray(d, dtype=np.float32)
    print(f"\n{name}:")
    print(f"  n     = {int(d.size)}")
    print(f"  min   = {float(d.min()):.6f}")
    print(f"  p50   = {float(np.quantile(d, 0.50)):.6f}")
    print(f"  p85   = {float(np.quantile(d, 0.85)):.6f}")
    print(f"  max   = {float(d.max()):.6f}")
    print(f"  mean  = {float(d.mean()):.6f}")

def _summarize_logp(name, a):
    a = np.asarray(a, dtype=np.float32)
    print(f"\n{name} (GMM logp on TRAIN latent):")
    print(f"  n     = {int(a.size)}")
    print(f"  min   = {float(a.min()):.4f}")
    print(f"  p10   = {float(np.quantile(a, 0.10)):.4f}")
    print(f"  p50   = {float(np.quantile(a, 0.50)):.4f}")
    print(f"  p85   = {float(np.quantile(a, 0.85)):.4f}")
    print(f"  max   = {float(a.max()):.4f}")
    print(f"  mean  = {float(a.mean()):.4f}")

def _align_by_mask_or_truncate(X, Z, nameX, nameZ):
    nx, nz = X.shape[0], Z.shape[0]
    if nx == nz:
        return X, Z, "paired"

    if "accept_mask" in globals():
        m = np.asarray(accept_mask).astype(bool).reshape(-1)
        if m.size == nx and nz == nx:
            return X[m], Z[m], "accepted(mask)"
        if m.size == nz and nx == nz:
            return X[m], Z[m], "accepted(mask)"

        if m.size == nx and nz != nx:
            X2 = X[m]
            nx2 = X2.shape[0]
            if nx2 == nz:
                return X2, Z, "accepted(mask)+paired"
        if m.size == nz and nx != nz:
            Z2 = Z[m]
            nz2 = Z2.shape[0]
            if nz2 == nx:
                return X, Z2, "accepted(mask)+paired"

    n = int(min(nx, nz))
    print(
        f"\n[WARN] Length mismatch between {nameX} (n={nx}) and {nameZ} (n={nz}). "
        f"Truncating both to n={n} to keep 1:1 pairing. "
        f"Upstream fix: ensure sampled compositions come from the same sampled_latent_vectors."
    )
    return X[:n], Z[:n], "truncated"


if "composition_cols" not in globals():
    raise RuntimeError("Missing required object: composition_cols")
if "gmm_latent_train" not in globals():
    raise RuntimeError("Missing required object: gmm_latent_train")
if "sampled_latent_vectors" not in globals():
    raise RuntimeError("Missing required object: sampled_latent_vectors")
if "optimized_latent_vectors" not in globals():
    raise RuntimeError("Missing required object: optimized_latent_vectors")
if "opt_comp" not in globals():
    raise RuntimeError("Missing required object: opt_comp")

ref_comp_name = _require_any(["X_comp_all_f", "X_comp_all"])
if ref_comp_name is None:
    raise RuntimeError("Missing reference composition matrix: expected X_comp_all_f or X_comp_all.")
X_ref_comp = _ensure_2d_float32(globals()[ref_comp_name], ref_comp_name)

ref_lat_name = _require_any(["latent_vectors_tr", "z_train"])
if ref_lat_name is None:
    raise RuntimeError("Missing reference latent matrix: expected latent_vectors_tr or z_train.")
Z_ref = _ensure_2d_float32(globals()[ref_lat_name], ref_lat_name)

Z_samp = _ensure_2d_float32(sampled_latent_vectors, "sampled_latent_vectors")
Z_opt  = _ensure_2d_float32(optimized_latent_vectors, "optimized_latent_vectors")
X_opt_comp = _ensure_2d_float32(opt_comp, "opt_comp")

X_samp_comp = None
src = None

if "comp_pred" in globals():
    Xcand = _ensure_2d_float32(comp_pred, "comp_pred")
    if Xcand.shape[0] == Z_samp.shape[0] and Xcand.shape[1] == X_ref_comp.shape[1]:
        X_samp_comp = Xcand
        src = "comp_pred"

if X_samp_comp is None and "df" in globals():
    try:
        Xcand = _ensure_2d_float32(df[composition_cols].values, "df[composition_cols]")
        if Xcand.shape[0] == Z_samp.shape[0] and Xcand.shape[1] == X_ref_comp.shape[1]:
            X_samp_comp = Xcand
            src = "df[composition_cols]"
    except Exception:
        pass

if X_samp_comp is None and "predicted_compositions" in globals():
    Xcand = _ensure_2d_float32(predicted_compositions, "predicted_compositions")
    if Xcand.shape[0] == Z_samp.shape[0] and Xcand.shape[1] == X_ref_comp.shape[1]:
        X_samp_comp = Xcand
        src = "predicted_compositions"

if X_samp_comp is None:
    raise RuntimeError(
        "Could not find sampled compositions paired with sampled_latent_vectors.\n"
        "Expected one of:\n"
        "  - comp_pred (decoder output for sampled_latent_vectors)\n"
        "  - df[composition_cols] with same number of rows as sampled_latent_vectors\n"
        "  - predicted_compositions with same number of rows as sampled_latent_vectors\n"
        "Upstream fix: set `comp_pred = ... decoder_model.predict(sampled_latent_vectors) ...` and reuse it."
    )

print(f"\nUsing sampled compositions from: {src}")

X_samp_comp, Z_samp, samp_tag = _align_by_mask_or_truncate(X_samp_comp, Z_samp, "X_samp_comp", "Z_samp")
X_opt_comp,  Z_opt,  opt_tag  = _align_by_mask_or_truncate(X_opt_comp,  Z_opt,  "X_opt_comp",  "Z_opt")

if X_ref_comp.shape[1] != X_samp_comp.shape[1] or X_ref_comp.shape[1] != X_opt_comp.shape[1]:
    raise ValueError(
        f"Composition feature dimension mismatch: "
        f"ref={X_ref_comp.shape[1]}, samp={X_samp_comp.shape[1]}, opt={X_opt_comp.shape[1]}"
    )
if Z_ref.shape[1] != Z_samp.shape[1] or Z_ref.shape[1] != Z_opt.shape[1]:
    raise ValueError(
        f"Latent dimension mismatch: ref D={Z_ref.shape[1]}, samp D={Z_samp.shape[1]}, opt D={Z_opt.shape[1]}"
    )
if X_samp_comp.shape[0] != Z_samp.shape[0]:
    raise ValueError(f"Sampled pairing mismatch after alignment: X={X_samp_comp.shape[0]} vs Z={Z_samp.shape[0]}")
if X_opt_comp.shape[0] != Z_opt.shape[0]:
    raise ValueError(f"Optimized pairing mismatch after alignment: X={X_opt_comp.shape[0]} vs Z={Z_opt.shape[0]}")

_simplex_report(X_samp_comp, f"Sampled compositions ({samp_tag})")
_simplex_report(X_opt_comp,  f"Optimized compositions ({opt_tag})")

# Distances to reference
sampled_comp_dists   = _min_l2_to_reference(X_ref_comp, X_samp_comp)
optimized_comp_dists = _min_l2_to_reference(X_ref_comp, X_opt_comp)

_summarize("Composition min-L2 to reference (Sampled)", sampled_comp_dists)
_summarize("Composition min-L2 to reference (Optimized)", optimized_comp_dists)

sampled_latent_dists   = _min_l2_to_reference(Z_ref, Z_samp)
optimized_latent_dists = _min_l2_to_reference(Z_ref, Z_opt)

_summarize("Latent min-L2 to reference (Sampled)", sampled_latent_dists)
_summarize("Latent min-L2 to reference (Optimized)", optimized_latent_dists)

gmm_logp_samp = gmm_latent_train.score_samples(Z_samp.astype(np.float64)).astype(np.float32)
gmm_logp_opt  = gmm_latent_train.score_samples(Z_opt.astype(np.float64)).astype(np.float32)

_summarize_logp("Sampled", gmm_logp_samp)
_summarize_logp("Optimized", gmm_logp_opt)

df_samp = pd.DataFrame({
    "set": np.repeat("Sampled", sampled_comp_dists.size),
    "idx": np.arange(sampled_comp_dists.size, dtype=int),
    "minL2_comp_to_ref": sampled_comp_dists.astype(np.float32),
    "minL2_latent_to_ref": sampled_latent_dists.astype(np.float32),
    "gmm_logp_trainlatent": gmm_logp_samp.astype(np.float32),
})
df_opt = pd.DataFrame({
    "set": np.repeat("Optimized", optimized_comp_dists.size),
    "idx": np.arange(optimized_comp_dists.size, dtype=int),
    "minL2_comp_to_ref": optimized_comp_dists.astype(np.float32),
    "minL2_latent_to_ref": optimized_latent_dists.astype(np.float32),
    "gmm_logp_trainlatent": gmm_logp_opt.astype(np.float32),
})

df_novelty = pd.concat([df_samp, df_opt], axis=0, ignore_index=True)
df_novelty.to_csv("novelty_distances_sampled_vs_optimized.csv", index=False)
print("\nSaved: novelty_distances_sampled_vs_optimized.csv")

df_samp.to_csv("novelty_sampled_only.csv", index=False)
df_opt.to_csv("novelty_optimized_only.csv", index=False)
print("Saved: novelty_sampled_only.csv")
print("Saved: novelty_optimized_only.csv")
