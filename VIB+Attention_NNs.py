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
from scipy.stats import boxcox, skew, norm, pearsonr, spearmanr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tqdm import trange
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import random
import os
import shap
from sklearn.utils import shuffle
import csv


# Introduce fixed seed
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# Data Ingestion
data = pd.read_csv('H_v_dataset.csv')
print("\nBasic Statistics:")
print(data.describe())

composition_cols = [col for col in data.columns if col not in ['Load', 'HV']]
load_col = 'Load'
target_col = 'HV'

# Hardness Distribution Plot
plt.style.use('default')
sns.set_context("talk")

plt.figure(figsize=(10, 8))
sns.histplot(data['HV'], kde=True, color='slateblue', bins=15, edgecolor='black')
plt.xlabel('Hardness (HV)', fontsize=22, weight='bold')
plt.ylabel('Frequency', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Distribution of Hardness (HV)", dpi=600, format='jpeg')
plt.show()

# Hardness-Load Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['Load'], y=data['HV'], color='darkcyan', s=180, edgecolor='black')
plt.xlabel('Load (N)', fontsize=22, weight='bold')
plt.ylabel('Hardness (HV)', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Relationship Between Load and Hardness (HV)", dpi=600, format='jpeg')
plt.show()


# Data Preparation
X_comp_all = data[composition_cols].values
X_load = data[[load_col]]
y = data['HV']

scaler_load = StandardScaler()
X_load = scaler_load.fit_transform(X_load)

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_comp_all)

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in strat_split.split(X_comp_all, cluster_labels):
    train_df = data.iloc[train_idx].reset_index(drop=True)
    test_df = data.iloc[test_idx].reset_index(drop=True)

X_comp_train = train_df[composition_cols].values
X_load_train = train_df[['Load']].values
y_train = train_df['HV'].values

X_comp_test = test_df[composition_cols].values
X_load_test = test_df[['Load']].values
y_test = test_df['HV'].values

train_mean = train_df[composition_cols].mean()
test_mean = test_df[composition_cols].mean()

compare_df = pd.DataFrame({'Train': train_mean, 'Test': test_mean})
compare_df.plot(kind='bar', figsize=(14, 6))
plt.ylabel('Mean Fraction')
plt.title('Elemental Fraction Distribution: Train vs Test')
plt.tight_layout()
plt.show()


# Skewness Correction (if needed)
skewness = skew(y)
print(f"Skewness of hardness (HV): {skewness:.2f}")

# Apply Box-Cox transformation
if skewness > 0.5: 
    y_transformed, _ = boxcox(y + 1)
    y = y_transformed

    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=30, color='purple', edgecolor='black')
    plt.title("Distribution of Transformed Hardness (HV)")
    plt.xlabel("Transformed Hardness (HV)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
else:
    print("No significant skewness detected; proceeding without transformation.")


## The VIBANN Architechture
# VIB Layer
class VIBLayer(layers.Layer):
    def __init__(self, latent_dim=16, **kwargs):
        super(VIBLayer, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = tf.Variable(1e-3, trainable=False, dtype=tf.float32)
        self.mean_dense = layers.Dense(self.latent_dim)
        self.log_var_dense = layers.Dense(self.latent_dim)
        self.kl_loss = tf.Variable(0.0, trainable=False) 

    def call(self, inputs):
        mean = self.mean_dense(inputs)
        log_var = self.log_var_dense(inputs)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        latent = mean + tf.exp(0.5 * log_var) * epsilon
        self.kl_loss.assign(-0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var)))
        self.add_loss(self.beta * self.kl_loss)
        return latent


# Feature-Wise Attention Layer
class FeatureWiseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(FeatureWiseAttention, self).__init__(**kwargs)
        self.attention_dense = layers.Dense(1, activation="softmax")  

    def call(self, inputs):
        attention_scores = tf.nn.softmax(inputs, axis=1)
        weighted_inputs = attention_scores * inputs
        return weighted_inputs, attention_scores

class AttentionScoreLogger(tf.keras.callbacks.Callback):
    def __init__(self, comp_feature_count, output_file='attention_scores.csv'):
        super().__init__()
        self.comp_feature_count = comp_feature_count
        self.output_file = output_file
        self.attention_scores = []

        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f"Comp_Feature_{i+1}" for i in range(comp_feature_count)] + ["Load_Feature"]
            writer.writerow(["Epoch"] + header)

    def on_epoch_end(self, epoch, logs=None):
        comp_attention_scores, load_attention_scores = self.model.predict(
            [X_comp_train, X_load_train], verbose=0
        )[1:3]

        avg_comp_attention = np.mean(comp_attention_scores, axis=0).flatten()
        avg_load_attention = np.mean(load_attention_scores, axis=0).flatten()

        all_attention_scores = np.concatenate((avg_comp_attention, avg_load_attention), axis=0)

        self.attention_scores.append(all_attention_scores)

        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1] + all_attention_scores.tolist())


# VIBANN Model Definition
def build_vib_attention_model(input_shape_comp, input_shape_load, latent_dim=16, dropout_rate=0.3, attention_heads=4):
    comp_input = layers.Input(shape=(input_shape_comp,), name="Composition_Input")
    x_comp = layers.Dense(64, activation="relu")(comp_input)
    x_comp = layers.BatchNormalization()(x_comp)
    x_comp = layers.Dropout(dropout_rate)(x_comp)
    x_comp_attention, comp_attention_scores = FeatureWiseAttention(name="Comp_Attention")(x_comp)

    load_input = layers.Input(shape=(input_shape_load,), name="Load_Input")
    x_load = layers.Dense(32, activation="relu")(load_input)
    x_load = layers.BatchNormalization()(x_load)
    x_load = layers.Dropout(dropout_rate)(x_load)
    x_load_attention, load_attention_scores = FeatureWiseAttention(name="Load_Attention")(x_load)

    combined = layers.Concatenate()([x_comp_attention, x_load_attention])
    vib_output = VIBLayer(latent_dim=latent_dim, name="vib_layer")(combined)

    reshaped_vib = layers.Reshape((1, latent_dim))(vib_output)
    attention_output = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=latent_dim)(reshaped_vib, reshaped_vib)
    attention_output = layers.Flatten()(attention_output)

    x = layers.Dense(64, activation="relu")(attention_output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(1, name="Output_Layer")(x)
    model = models.Model(inputs=[comp_input, load_input],
                     outputs=[output, comp_attention_scores, load_attention_scores],
                     name="VIB_Attention_Model")

    return model

vib_attention_model = build_vib_attention_model(X_comp_train.shape[1], X_load_train.shape[1])


# Adaptive Beta Callback
class AdaptiveBetaCallback(tf.keras.callbacks.Callback):
    def __init__(self, vib_layer, target_kl=0.1, beta_adjustment_factor=1.05):
        super().__init__()
        self.vib_layer = vib_layer
        self.target_kl = target_kl
        self.beta_adjustment_factor = beta_adjustment_factor

    def on_epoch_end(self, epoch, logs=None):
        kl_loss = self.vib_layer.kl_loss.numpy()
        if kl_loss < self.target_kl:
            new_beta = self.vib_layer.beta * self.beta_adjustment_factor
        else:
            new_beta = self.vib_layer.beta / self.beta_adjustment_factor
        self.vib_layer.beta.assign(new_beta)
        print(f"Epoch {epoch + 1}: KL Loss = {kl_loss:.4f}, Updated Beta = {self.vib_layer.beta.numpy():.6f}")


def adjust_beta(vib_layer, kl_loss, target_kl=0.1, beta_adjustment_factor=1.05):
    if kl_loss < target_kl:
        new_beta = vib_layer.beta * beta_adjustment_factor
    else:
        new_beta = vib_layer.beta / beta_adjustment_factor
    vib_layer.beta.assign(new_beta)

vib_layer = None
for layer in vib_attention_model.layers:
    if isinstance(layer, VIBLayer):
        vib_layer = layer
        break
if vib_layer is None:
    raise ValueError("VIB layer not found in the model.")


# Bayesian Optimization for Latent Dimension
def objective(trial):
    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    attention_heads = trial.suggest_categorical("attention_heads", [2, 4, 8])
    
    model = build_vib_attention_model(
        input_shape_comp=X_comp_train.shape[1],
        input_shape_load=X_load_train.shape[1],
        latent_dim=latent_dim,
        attention_heads=4
    )

    vib_layer = next(layer for layer in model.layers if isinstance(layer, VIBLayer))

    model.compile(
        optimizer="adam",
        loss={
            "Output_Layer": tf.keras.losses.MeanSquaredError(),
            "Comp_Attention": lambda y_true, y_pred: 0.0,
            "Load_Attention": lambda y_true, y_pred: 0.0
        },
        loss_weights={
            "Output_Layer": 1.0,
            "Comp_Attention": 0.0,
            "Load_Attention": 0.0
        }
    )

    dummy_comp = np.zeros_like(X_comp_train)
    dummy_load = np.zeros_like(X_load_train)

    history = model.fit(
        [X_comp_train, X_load_train],
        {
            "Output_Layer": y_train,
            "Comp_Attention": dummy_comp,
            "Load_Attention": dummy_load
        },
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )

    val_loss = history.history["val_loss"][-1]
    return val_loss

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=25)

all_info = sorted(study.trials, key=lambda x: x.value)[:25]
all_info_df = pd.DataFrame(
    [(t.number,
      t.params['latent_dim'],
      t.params['dropout_rate'],
      t.params['attention_heads'],
      t.value) for t in all_info],
    columns=["Trial", "Latent_Dim", "Dropout", "Attention_Heads", "Val_Loss"]
)
all_info_df.to_csv("Bayesian_Optimization_VIBANN_parameters.csv", index=False)
print("All Latent dimensions saved to 'Bayesian_Optimization_Results.csv'")

best_latent_dim = study.best_trial.params['latent_dim']
best_dropout_rate = study.best_trial.params['dropout_rate']
best_attention_heads = study.best_trial.params['attention_heads']
print(f"Best latent dimension selected via BO: {best_latent_dim}")
print(f"Best drouput rate selected via BO: {best_dropout_rate}")
print(f"Best number of attention heads selected via BO: {best_attention_heads}")


# Model Training
train_losses = []
val_losses = []
beta_values = []

epochs = 1000
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_results = {
    "train_losses": [],
    "val_losses": [],
    "r2_scores": [],
    "mse_scores": [],
    "beta_traces": [],
    "attention_scores": []
}

for fold, (train_index, val_index) in enumerate(skf.split(X_comp_all, cluster_labels)):
    print(f"\nFold {fold + 1}/{k}")

    X_comp_train = X_comp_all[train_index]
    X_load_train = X_load[train_index]
    y_train = y.iloc[train_index]

    X_comp_val = X_comp_all[val_index]
    X_load_val = X_load[val_index]
    y_val = y.iloc[val_index]

    dummy_comp_train = np.zeros((X_comp_train.shape[0], X_comp_train.shape[1]))
    dummy_load_train = np.zeros((X_comp_train.shape[0], 1))
    dummy_comp_val = np.zeros((X_comp_val.shape[0], X_comp_val.shape[1]))
    dummy_load_val = np.zeros((X_comp_val.shape[0], 1))

    vib_attention_model = build_vib_attention_model(input_shape_comp=X_comp_train.shape[1], input_shape_load=X_load_train.shape[1], latent_dim=best_latent_dim, dropout_rate=best_dropout_rate, attention_heads=best_attention_heads)
    vib_layer = next(layer for layer in vib_attention_model.layers if isinstance(layer, VIBLayer))

    vib_attention_model.compile(
        optimizer="adam",
        loss={
            "Output_Layer": tf.keras.losses.MeanSquaredError(),
            "Comp_Attention": lambda y_true, y_pred: 0.0,
            "Load_Attention": lambda y_true, y_pred: 0.0,
        },
        loss_weights={
            "Output_Layer": 1.0,
            "Comp_Attention": 0.0,
            "Load_Attention": 0.0,
        }
    )

    attention_logger = AttentionScoreLogger(comp_feature_count=X_comp_train.shape[1])
    adaptive_beta_callback = AdaptiveBetaCallback(vib_layer)

    train_losses, val_losses, beta_values = [], [], []

    for epoch in trange(epochs, desc=f"Fold {fold + 1} Training"):
        history = vib_attention_model.fit(
            [X_comp_train, X_load_train],
            {
                "Output_Layer": y_train,
                "Comp_Attention": dummy_comp_train,
                "Load_Attention": dummy_load_train,
            },
            validation_data=(
                [X_comp_val, X_load_val],
                {
                    "Output_Layer": y_val,
                    "Comp_Attention": dummy_comp_val,
                    "Load_Attention": dummy_load_val,
                }
            ),
            epochs=1,
            callbacks=[attention_logger, adaptive_beta_callback],
            verbose=0
        )
        train_losses.append(history.history['loss'][0])
        val_losses.append(history.history['val_loss'][0])
        kl_loss = vib_layer.kl_loss.numpy()
        adjust_beta(vib_layer, kl_loss)
        beta_values.append(vib_layer.beta.numpy())

    y_pred = vib_attention_model.predict([X_comp_val, X_load_val], verbose=0)[0]
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)

    fold_results["train_losses"].append(train_losses)
    fold_results["val_losses"].append(val_losses)
    fold_results["beta_traces"].append(beta_values)
    fold_results["r2_scores"].append(r2)
    fold_results["mse_scores"].append(mse)
    fold_results["attention_scores"].append(attention_logger.attention_scores.copy())

    vib_attention_model.save(f"vib_model_fold_{fold+1}.h5")

    print(f"Fold {fold+1} R²: {r2:.4f}, MSE: {mse:.2f}")


# Plot learning curves for all folds
plt.figure(figsize=(10, 8))
for i in range(k):
    plt.plot(fold_results["train_losses"][i], label=f'Train Fold {i+1}')
    plt.plot(fold_results["val_losses"][i], label=f'Val Fold {i+1}', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot Training and Validation Loss for best fold
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(train_losses, label='Training Loss', color='blue', linewidth=3, marker='.', markersize=2, markerfacecolor='blue')
ax.plot(val_losses, label='Validation Loss', color='red', linewidth=3, marker='.', markersize=2, markerfacecolor='red')
ax.set_xlabel('Epochs', fontsize=22, weight='bold', color='black')
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=22, weight='bold', color='black')
legend = ax.legend(fontsize=14, loc='upper right', fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')
ax.tick_params(axis='both', which='major', labelsize=15, color='black')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig("Model Training and Validation Loss Over Epochs", dpi=600, format='jpeg')
plt.show()


# Plot loss vs. folds
avg_train_losses = [np.mean(losses) for losses in fold_results["train_losses"]]
std_train_losses = [np.std(losses) for losses in fold_results["train_losses"]]
avg_val_losses = [np.mean(losses) for losses in fold_results["val_losses"]]
std_val_losses = [np.std(losses) for losses in fold_results["val_losses"]]

fold_ids = np.arange(1, k + 1)

plt.figure(figsize=(10, 8))
plt.plot(fold_ids, avg_train_losses, 'o--', color='crimson', label='Train', linewidth=1.8)
plt.fill_between(fold_ids,
                 np.array(avg_train_losses) - np.array(std_train_losses),
                 np.array(avg_train_losses) + np.array(std_train_losses),
                 alpha=0.3, color='crimson')

plt.plot(fold_ids, avg_val_losses, 'o--', color='teal', label='Validation', linewidth=1.8)
plt.fill_between(fold_ids,
                 np.array(avg_val_losses) - np.array(std_val_losses),
                 np.array(avg_val_losses) + np.array(std_val_losses),
                 alpha=0.3, color='teal')

plt.xlabel("Training set size", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.xticks(fold_ids)
plt.ylim(80, 110)
plt.legend(frameon=True, facecolor='lightgrey', fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig("Learning_Curve.jpeg", dpi=600, format='jpeg')
plt.show()


# Plot beta value over epochs
plt.figure(figsize=(10, 8))
plt.plot(beta_values, label=r'$\beta$ (Adaptive)', color='darkviolet', linewidth=3, marker='.', markersize=3, markerfacecolor='indigo')
plt.xlabel('Epochs', fontsize=22, labelpad=10, weight='bold', color='black')
plt.ylabel(r'$\beta$ Value', fontsize=22, labelpad=10, weight='bold', color='black')
plt.grid(False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig("beta Value Dynamics During Training", dpi=600, format='jpeg')
plt.show()


# Save model weights and full model
vib_attention_model.save_weights("vib_attention_model.weights.h5")
vib_attention_model.save("vib_attention_full_model.keras")
print("Model weights and full model saved.")


# Monte Carlo Dropout for Uncertaininty Quantification
def bootstrap_rmse_ci(y_true, y_pred_samples, n_bootstrap=1000, ci=0.95):
    n = len(y_true)
    rmse_list = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[idx]
        y_pred_sample = y_pred_samples[:, idx].mean(axis=0)
        rmse_sample = np.sqrt(mean_squared_error(y_true_sample, y_pred_sample))
        rmse_list.append(rmse_sample)

    rmse_array = np.array(rmse_list)
    lower = np.percentile(rmse_array, ((1 - ci) / 2) * 100)
    upper = np.percentile(rmse_array, (1 - (1 - ci) / 2) * 100)
    rmse_mean = np.mean(rmse_array)

    return rmse_mean, lower, upper, rmse_list


def mc_dropout_predictions(model, X_comp, X_load, num_samples=100):
    predictions = []

    @tf.function
    def predict_with_dropout(comp_input, load_input):
        return model([comp_input, load_input], training=True)

    for _ in range(num_samples):
        outputs = predict_with_dropout(X_comp, X_load)
        y_pred = outputs[0]
        predictions.append(y_pred.numpy().flatten())

    predictions = np.array(predictions)
    y_pred_mean = predictions.mean(axis=0)
    y_pred_std = predictions.std(axis=0) 
    return y_pred_mean, y_pred_std


predictions = []
for _ in range(100):
    y_pred_sample, _ = mc_dropout_predictions(vib_attention_model, X_comp_test, X_load_test, num_samples=1)
    predictions.append(y_pred_sample)

predictions = np.array(predictions)
y_pred_mean = predictions.mean(axis=0)
y_pred_std = predictions.std(axis=0)

if len(y_test) != len(y_pred_mean):
    min_len = min(len(y_test), len(y_pred_mean))
    y_test = y_test[:min_len]
    y_pred_mean = y_pred_mean[:min_len]
    y_pred_std = y_pred_std[:min_len]

rmse_mean, rmse_lower, rmse_upper, rmse_list = bootstrap_rmse_ci(
    y_true=np.array(y_test), 
    y_pred_samples=predictions,
    n_bootstrap=1000,
    ci=0.95
)

print(f"Test RMSE: {rmse_mean:.4f} (95% CI: [{rmse_lower:.4f}, {rmse_upper:.4f}])")


# Plot Predicted vs Actual Hardness with Uncertainty Intervals
fig, ax = plt.subplots(figsize=(10, 8))
ax.errorbar(y_test, y_pred_mean, yerr=y_pred_std/2.5, fmt='o', ecolor='red', alpha=0.8, capsize=5, 
            markerfacecolor='blue', markeredgewidth=1, markersize=10, label='Monte-Carlo Dropout Predictions')
ax.plot([0, 2000], [0, 2000], 'k--', lw=2)
ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)
ax.set_xlabel('Actual Hardness (HV)', fontsize=22, weight='bold')
ax.set_ylabel('Predicted Hardness (HV)', fontsize=22, weight='bold')
ax.grid(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(['Ideal Prediction Line', 'Monte-Carlo Dropout Predictions'], loc='upper left', fontsize=16)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Predicted vs. Actual Hardness with Uncertainty Intervals", dpi=600, format='jpeg')
plt.show()


# Distribution of Predicted Uncertainty
fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(y_pred_std/2, bins=30, color='red', kde=True, alpha=0.7, ax=ax)
ax.set_xlabel('Prediction Uncertainty (Std Dev)', fontsize=26, weight='bold')
ax.set_ylabel('Frequency', fontsize=26, weight='bold')
ax.grid(False)
ax.tick_params(axis='both', which='major', labelsize=24)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Distribution of Prediction Uncertainty (Standard Deviation)", dpi=600, format='jpeg')
plt.show()


# RMSE Frequency Distribution
plt.figure(figsize=(10,6))
sns.histplot(rmse_list, kde=True, bins=30, color='skyblue')
plt.axvline(rmse_mean, color='blue', linestyle='--', label='Mean RMSE')
plt.axvline(rmse_lower, color='red', linestyle='--', label='Lower CI')
plt.axvline(rmse_upper, color='green', linestyle='--', label='Upper CI')
plt.xlabel("RMSE")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of RMSE")
plt.legend()
plt.tight_layout()
plt.show()


# Calculate additional metrics for evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
mae = mean_absolute_error(y_test, y_pred_mean)
mape = mean_absolute_percentage_error(y_test, y_pred_mean)
r2 = r2_score(y_test, y_pred_mean)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MAPE: {mape:.4f}")
print(f"Test R² Score: {r2:.4f}")


# Define the attention extractor model
comp_input_layer, load_input_layer = vib_attention_model.input

comp_attention_layer_output = vib_attention_model.get_layer("Comp_Attention").output[1]
load_attention_layer_output = vib_attention_model.get_layer("Load_Attention").output[1]

attention_extractor_model = tf.keras.Model(
    inputs=[comp_input_layer, load_input_layer],
    outputs=[comp_attention_layer_output, load_attention_layer_output],
    name="Attention_Extractor"
)

# Obtain Attention Scores
comp_attention_scores_final, load_attention_scores_final = attention_extractor_model.predict([X_comp_test, X_load_test])
comp_attention_scores_final = comp_attention_scores_final[:, :56]
avg_comp_attention_scores = np.mean(comp_attention_scores_final, axis=0)

feature_names = composition_cols

plt.figure(figsize=(14, 6))
sns.barplot(x=feature_names, y=avg_comp_attention_scores, palette="viridis")
plt.xlabel("Composition Feature", fontsize=22, weight="bold", labelpad=10, color="black")
plt.ylabel("Average Attention Score", fontsize=22, weight="bold", labelpad=10, color="black")
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("Average Attention Weights for Composition Features", dpi=600, format='jpeg')
plt.show()


# Attriation Analysis for composition features
def integrated_gradients(model, baseline, inputs, target_output_idx=None, steps=200):
    """
    Compute Integrated Gradients for a model and inputs.

    Parameters:
    - model: The model to interpret.
    - baseline: The baseline input values (same shape as inputs).
    - inputs: The input values for which gradients are calculated.
    - target_output_idx: The index of the target output if there are multiple outputs.
    - steps: The number of steps to approximate the integral.

    Returns:
    - Integrated gradients for each input tensor.
    """
    scaled_inputs_comp = [baseline[0] + (float(i) / steps) * (inputs[0] - baseline[0]) for i in range(steps + 1)]
    scaled_inputs_load = [baseline[1] + (float(i) / steps) * (inputs[1] - baseline[1]) for i in range(steps + 1)]
    
    gradients = []
    for comp_inp, load_inp in zip(scaled_inputs_comp, scaled_inputs_load):
        with tf.GradientTape() as tape:
            tape.watch([comp_inp, load_inp])
            outputs = model([comp_inp, load_inp])
            if target_output_idx is not None:
                outputs = outputs[target_output_idx]
        grads = tape.gradient(outputs, [comp_inp, load_inp])
        gradients.append(grads)

    avg_gradients_comp = tf.reduce_mean([grad[0] for grad in gradients], axis=0)
    avg_gradients_load = tf.reduce_mean([grad[1] for grad in gradients], axis=0)

    integrated_grads_comp = (inputs[0] - baseline[0]) * avg_gradients_comp
    integrated_grads_load = (inputs[1] - baseline[1]) * avg_gradients_load

    return integrated_grads_comp, integrated_grads_load

comp_baseline_np = np.mean(X_comp_train, axis=0, keepdims=True)
load_baseline_np = np.mean(X_load_train, axis=0, keepdims=True)

comp_baseline = tf.constant(comp_baseline_np, dtype=tf.float32)
load_baseline = tf.constant(load_baseline_np, dtype=tf.float32)

sample_idx = 0

comp_row = X_comp_test[sample_idx]
comp_input = tf.expand_dims(comp_row, axis=0)

load_row = X_load_test[sample_idx]
load_row = load_row.reshape(1, -1)
load_input = tf.constant(load_row, dtype=tf.float32)

baseline_inputs = [comp_baseline, load_baseline]
sample_inputs = [comp_input, load_input]

comp_input_tf = tf.cast(comp_input, tf.float32)
load_input_tf = tf.cast(load_input, tf.float32)
comp_baseline_tf = tf.cast(comp_baseline, tf.float32)
load_baseline_tf = tf.cast(load_baseline, tf.float32)

ig_comp = integrated_gradients(vib_attention_model, [comp_baseline_tf, load_baseline_tf],
                               [comp_input_tf, load_input_tf])

ig_comp_values = ig_comp[0].numpy().flatten()
feature_names = composition_cols

colors = sns.color_palette("mako", len(ig_comp_values))

plt.figure(figsize=(16, 8))
bars = plt.bar(feature_names, ig_comp_values, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Compositional Constituent', fontsize=28, weight='bold', labelpad=10, color='black')
plt.ylabel('Attribution Value', fontsize=28, weight='bold', labelpad=10, color='black')
plt.xticks(range(len(ig_comp_values)), feature_names, rotation=90, fontsize=26, color='black')
plt.yticks(fontsize=26)
plt.grid(False)
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("Integrated Gradients for Composition Features", dpi=600, format='jpeg')
plt.show()


# Reliability Diagram for VIBANN Model
confidence_levels = np.linspace(0.05, 0.95, 10)
coverage_probabilities = []

predictions_sorted = np.sort(predictions, axis=0)

for alpha in confidence_levels:
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    lower_bound = np.percentile(predictions, lower_q * 100, axis=0)
    upper_bound = np.percentile(predictions, upper_q * 100, axis=0)
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
    coverage_probabilities.append(coverage)

plt.figure(figsize=(10, 8))
plt.plot(confidence_levels, coverage_probabilities, 'o-', label="Model Coverage", color="blue")
plt.plot([0, 1.0], [0, 1.0], 'k--', label="Perfect Calibration")
plt.xlabel("Confidence Level (Predicted)", fontsize=22, weight='bold')
plt.ylabel("Coverage Probability (Empirical)", fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.savefig("Reliability Diagram for Prediction Calibration", dpi=600, format='jpeg')
plt.show()


# Calibration Curve for VIBANN Model
pred_bins = np.linspace(y_pred_mean.min(), y_pred_mean.max(), num=20)
bin_indices = np.digitize(y_pred_mean, pred_bins)
bin_centers = (pred_bins[:-1] + pred_bins[1:]) / 2
actual_means = [np.mean(y_test[bin_indices == i]) for i in range(1, len(pred_bins))]
predicted_means = [np.mean(y_pred_mean[bin_indices == i]) for i in range(1, len(pred_bins))]

plt.figure(figsize=(10, 8))
plt.plot(predicted_means, actual_means, 'o-', color='green', label="Calibration Curve")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Perfect Calibration")
plt.xlabel("Mean Predicted Hardness", fontsize=22, weight='bold')
plt.ylabel("Mean Observed Hardness", fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.savefig("Calibration Curve for Predicted Hardness", dpi=600, format='jpeg')
plt.show()


## Latent Sampling and Inverse Design
# VIB Latent Space Visualization
latent_model = tf.keras.Model(inputs=vib_attention_model.input, 
                              outputs=vib_attention_model.get_layer('vib_layer').output)

latent_vectors = latent_model.predict([X_composition, X_load_scaled])

cluster_range = range(1, 11)
wcss = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(latent_vectors)
    wcss.append(kmeans.inertia_)

# Plot WCSS for the Elbow Method
plt.figure(figsize=(10, 8))
plt.plot(cluster_range, wcss, marker='o', color='g', markersize=15, linestyle='--', linewidth=2)
plt.xlabel('Number of Clusters (k)', fontsize=22, weight='bold')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Elbow Method for Optimal Number of Clusters", dpi=600, format='jpeg')
plt.show()

# Dimensionality Reduction with t-SNE
tsne = TSNE(n_components=2, random_state=42)
latent_tsne = tsne.fit_transform(latent_vectors)

# K-Means Clustering in Latent Space
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(latent_vectors)
cluster_labels = kmeans.labels_

# t-SNE plot with Cluster Assignments
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=clusters, cmap='plasma', s=200, edgecolor='k', alpha=0.7)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_xlabel('Dimension 1', fontsize=22, weight='bold')
ax.set_ylabel('Dimension 2', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("t-SNE Visualization of Latent Space with K-Means Clustering", dpi=600, format='jpeg')
plt.show()

# Identify clusters
unique_clusters = np.unique(cluster_labels)
print(f"Found {len(unique_clusters)} clusters:", unique_clusters)
data['Cluster_Label'] = clusters

# t-SNE plot with Color-Coding by Hardness
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=y, cmap='coolwarm', s=200, edgecolor='k', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('Hardness (HV)', fontsize=22, weight='bold')
ax.set_xlabel('Dimension 1', fontsize=22, weight='bold')
ax.set_ylabel('Dimension 2', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("t-SNE Visualization of Latent Space (Color-Coded by Hardness)", dpi=600, format='jpeg')
plt.show()

# Plot hardness distributions across clusters
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster_Label', y='HV', data=data, palette='plasma')
plt.xlabel('Cluster Label', fontsize=22, weight='bold')
plt.ylabel('Hardness (HV)', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig("Hardness Distribution Across Clusters", dpi=600)
plt.show()

# Fit a GMM on the latent vectors
top_percentile = 95
high_idx = np.where(y >= np.percentile(y, top_percentile))[0]
high_latent_vectors = latent_vectors[high_idx]
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=1)
gmm.fit(high_latent_vectors)

# Fit a separate GMM on the 2D latent space (for visualization only)
gmm_temp = GaussianMixture(n_components=3, covariance_type='full', random_state=1)
gmm_temp.fit(latent_tsne)

x_temp = np.linspace(latent_tsne[:, 0].min() - 1, latent_tsne[:, 0].max() + 1, 100)
y_temp = np.linspace(latent_tsne[:, 1].min() - 1, latent_tsne[:, 1].max() + 1, 100)
X_temp, Y_temp = np.meshgrid(x_temp, y_temp)
grid_points_temp = np.c_[X_temp.ravel(), Y_temp.ravel()]

log_likelihood_temp = (gmm_temp.score_samples(grid_points_temp))
Z_temp = log_likelihood_temp.reshape(X_temp.shape)

if len(y) != len(latent_tsne):
    y_temp = np.resize(y_temp, latent_tsne.shape[0])

x_temp = np.linspace(X_temp.min(), X_temp.max(), 300)
y_temp = np.linspace(Y_temp.min(), Y_temp.max(), 300)
X_smooth_temp, Y_smooth_temp = np.meshgrid(x_temp, y_temp)

Z_smooth_temp = griddata(
    (X_temp.ravel(), Y_temp.ravel()), Z_temp.ravel(), (X_smooth_temp, Y_smooth_temp), method='cubic'
)

plt.figure(figsize=(12, 8))
contour = plt.contourf(X_smooth_temp, Y_smooth_temp, Z_smooth_temp, levels=50, cmap='coolwarm', alpha=1)
contour_lines = plt.contour(X_smooth_temp, Y_smooth_temp, Z_smooth_temp, levels=50, colors='grey', linewidths=0.5, alpha=0.5)

plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c='black', s=10, alpha=0.3)
cbar = plt.colorbar(contour)
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Log Likelihood', fontsize=22, weight='bold')
plt.xlabel('Dimension 1', fontsize=22, weight='bold')
plt.ylabel('Dimension 2', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("gmm_probability_density_with_scatter_points.png", dpi=600, format='png')
plt.show()


# SHAP on all clusters
latent_all = latent_model.predict([X_comp_all, X_load])

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(latent_all)
cluster_labels = gmm.predict(latent_all)

# SHAP input for ALL DATA
X_full_all = np.concatenate([X_comp_all, X_load], axis=1)
y_all = y

def shap_predict(inputs):
    comp = inputs[:, :n_comp]
    load = inputs[:, n_comp:]
    pred = vib_attention_model.predict([comp, load], verbose=0)[0]
    return pred

n_comp = X_comp_all.shape[1]
X_full_all = np.concatenate([X_comp_all, X_load], axis=1)
feature_names = composition_cols + ['Load']

explainer = shap.Explainer(shap_predict, X_full_all, feature_names=feature_names)

cluster_means = {}

for cid in np.unique(cluster_labels):
    idx = np.where(cluster_labels == cid)[0]
    X_cluster = X_full_all[idx]
    y_cluster = y.values[idx] 

    shap_values = explainer(X_cluster)

    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title(f"SHAP Bar Plot - Cluster {cid}")
    plt.tight_layout()
    plt.savefig(f"Cluster_{cid}_SHAP_Bar.png")
    plt.close()
  
    mean_feat = pd.DataFrame(X_cluster, columns=feature_names).mean()
    cluster_means[f"Cluster_{cid}"] = mean_feat

means_df = pd.DataFrame(cluster_means).T
means_df.to_csv("Cluster_Physical_Feature_Means.csv")
print(means_df)


# Latent Traversal Analysis
vib_idx = next(i for i, L in enumerate(vib_attention_model.layers) if L.name.startswith("vib_layer"))
latent_dim = vib_attention_model.layers[vib_idx].latent_dim

latent_in = tf.keras.Input(shape=(latent_dim,), name="latent_input")
x = tf.keras.layers.Reshape((1, latent_dim))(latent_in)

for L in vib_attention_model.layers[vib_idx+1:]:
    if isinstance(L, tf.keras.layers.MultiHeadAttention):
        x = L(x, x, x)
    else:
        x = L(x)

x = tf.keras.layers.Flatten()(x)

comp_dim = X_comp_train.shape[1]
comp_out = tf.keras.layers.Dense(comp_dim, activation='softmax', name="comp_out")(x)
hard_out = tf.keras.layers.Dense(1, activation='relu', name="hardness_out")(x)
decoder_16 = tf.keras.Model(latent_in, [comp_out, hard_out], name="decoder_16")

center_lv = latent_vectors.mean(axis=0)
top95 = np.percentile(y, 95)
high_idx = np.where(y >= top95)[0]
high_lv = latent_vectors[np.random.choice(high_idx)]

candidates = {"Center": center_lv, "High": high_lv}

latent_min = latent_vectors.min(axis=0)
latent_max = latent_vectors.max(axis=0)

plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

colors = {'Center': 'tab:green', 'High': 'tab:red'}

for name, base_lv in candidates.items():
    fig, axes = plt.subplots(latent_dim // 4, 4, figsize=(24, 16), facecolor='white')
    axes = axes.flatten()

    for dim in range(latent_dim):
        traj = []
        deltas = np.linspace(latent_min[dim], latent_max[dim], 20)
        for val in deltas:
            lv = base_lv.copy()
            lv[dim] = val
            _, h_pred = decoder_16.predict(lv[None, :], verbose=0)
            traj.append(float(h_pred))

        ax = axes[dim]
        ax.plot(
            deltas,
            traj,
            marker='o',
            markersize=8,
            linewidth=2,
            alpha=1,
            color=colors[name],
            label=name
        )
        ax.set_title(f"Dim {dim+1}", fontsize=11, pad=6)
        ax.set_xlabel("Latent Value", fontsize=9)
        ax.set_ylabel("Predicted HV", fontsize=9)
        ax.grid(False)

    fig.legend(
        loc='upper right',
        labels=[name],
        markerscale=1.5,
        frameon=True,
        framealpha=0
    )
    plt.suptitle(f"Latent Traversal — {name}", fontsize=20, y=1.02)
    plt.tight_layout()

    fig.savefig(
        f"latent_traversal_{name.lower()}.jpg",
        format='jpg',
        dpi=600,
        bbox_inches='tight'
    )
    plt.show()


# Latent–Hardness Correlations
vib_layer = vib_attention_model.get_layer("vib_layer")
latent_dim = vib_layer.latent_dim
print("Detected latent_dim =", latent_dim)

layer_names = [L.name for L in vib_attention_model.layers]
vib_idx     = layer_names.index(vib_layer.name)

post_layers = vib_attention_model.layers[vib_idx+1:]

latent_input = tf.keras.Input(shape=(latent_dim,), name="latent_in")

x = tf.keras.layers.Reshape((1, latent_dim))(latent_input)

for L in post_layers:
    if isinstance(L, tf.keras.layers.MultiHeadAttention):
        x = L(query=x, value=x, key=x)
    else:
        x = L(x)

decoder_model = tf.keras.Model(inputs=latent_input, outputs=x, name="decoder_proper")
decoder_model.summary()

latent_extractor = tf.keras.Model(
    inputs=vib_attention_model.input,
    outputs=vib_attention_model.get_layer(vib_layer.name).output
)
all_latents = latent_extractor.predict([X_composition, X_load_scaled], verbose=0)

N = 10000
means = all_latents.mean(axis=0)
stds  = all_latents.std(axis=0)
rand_latents = np.random.randn(N, latent_dim) * stds + means

hardness_preds = decoder_model.predict(rand_latents, verbose=0).flatten()

pearson_coeffs  = []
spearman_coeffs = []
for d in range(latent_dim):
    c = rand_latents[:, d]
    p, _ = pearsonr(c, hardness_preds)
    s, _ = spearmanr(c, hardness_preds)
    pearson_coeffs.append(p)
    spearman_coeffs.append(s)

dims = np.arange(1, latent_dim+1)
plt.figure(figsize=(12,8))
plt.plot(dims, pearson_coeffs,  marker='o', linestyle='-',  label='Pearson')
plt.plot(dims, spearman_coeffs, marker='s', linestyle='--', label='Spearman')
plt.xlabel('Latent Dimension', fontsize=14, weight='bold')
plt.ylabel('Correlation with Hardness', fontsize=14, weight='bold')
plt.title('Global Latent–Hardness Correlations', fontsize=16, weight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Pricipal Component Analysis on Latent Space
latent_vectors_test = latent_model.predict([X_comp_test, X_load_test])
hardness_preds = y_pred_mean

assert latent_vectors_test.shape[0] == hardness_preds.shape[0], \
       f"Mismatch: {latent_vectors_test.shape[0]} vs {hardness_preds.shape[0]}"

pca       = PCA(n_components=2)
pc_scores = pca.fit_transform(latent_vectors_test)

lr1     = LinearRegression().fit(pc_scores[:, [0]], hardness_preds)
r2_pc1  = r2_score(hardness_preds, lr1.predict(pc_scores[:, [0]]))
lr2     = LinearRegression().fit(pc_scores[:, [1]], hardness_preds)
r2_pc2  = r2_score(hardness_preds, lr2.predict(pc_scores[:, [1]]))

plt.figure(figsize=(8,6))
plt.scatter(pc_scores[:,0], hardness_preds, s=100, edgecolor='k', alpha=0.6)
x1 = np.array([pc_scores[:,0].min(), pc_scores[:,0].max()])[:,None]
y1 = lr1.predict(x1)
plt.plot(x1, y1, 'r--', lw=2, label=f'Fit R²={r2_pc1:.3f}')
plt.xlabel('PC 1', fontsize=14)
plt.ylabel('Predicted Hardness (HV)', fontsize=14)
plt.title('Principal Component 1 vs Hardness', fontsize=16)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(pc_scores[:,1], hardness_preds, s=100, edgecolor='k', alpha=0.6)
x2 = np.array([pc_scores[:,1].min(), pc_scores[:,1].max()])[:,None]
y2 = lr2.predict(x2)
plt.plot(x2, y2, 'r--', lw=2, label=f'Fit R²={r2_pc2:.3f}')
plt.xlabel('PC 2', fontsize=14)
plt.ylabel('Predicted Hardness (HV)', fontsize=14)
plt.title('Principal Component 2 vs Hardness', fontsize=16)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")


# Gradient‐Based Importance of Latent Dimensions
latent_var = tf.Variable(latent_vectors_test.astype(np.float32))  

with tf.GradientTape() as tape:
    tape.watch(latent_var)
    outputs = decoder_model(latent_var, training=False)
    hardness_pred = outputs[1]          
    hardness_pred = tf.squeeze(hardness_pred, axis=-1)  

grads = tape.gradient(hardness_pred, latent_var)  

sensitivities = tf.reduce_mean(tf.abs(grads), axis=0).numpy() 

plt.figure(figsize=(12,6))
sns.barplot(x=[f"z{i+1}" for i in range(sensitivities.shape[0])],
            y=sensitivities, palette="magma")
plt.xticks(rotation=90, fontsize=12)
plt.ylabel("Mean |∂Hardness / ∂z|", fontsize=14)
plt.xlabel("Latent Dimension", fontsize=14)
plt.title("Gradient‐Based Importance of Latent Dimensions", fontsize=16)
plt.tight_layout()
plt.show()


## Inverse Design of ultra-high hardness alloys
cluster_0_mean = gmm.means_[0]
cluster_0_cov = gmm.covariances_[0]

# Define the MCMC model for cluster 0
def gmm_cluster_0_model():
    numpyro.sample(
        "sampled_latent",
        dist.MultivariateNormal(loc=cluster_0_mean, covariance_matrix=cluster_0_cov)
    )

rng_key = jax.random.PRNGKey(0)
nuts_kernel = NUTS(gmm_cluster_0_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=200, num_chains=1)

# Run MCMC sampling
mcmc.run(rng_key)
sampled_latent_vectors = mcmc.get_samples()["sampled_latent"]
print("Sampled Latent Vectors Shape:", sampled_latent_vectors.shape)

# Decoder Model
vib_layer_name = 'vib_layer'
vib_layer_index = None
for idx, layer in enumerate(vib_attention_model.layers):
    if layer.name == vib_layer_name:
        vib_layer_index = idx
        break

if vib_layer_index is None:
    raise ValueError(f"VIB layer named '{vib_layer_name}' not found in the model.")

print(f"VIB Layer Index: {vib_layer_index}")

latent_dim_size = sampled_latent_vectors.shape[1]
latent_input = tf.keras.Input(shape=(latent_dim_size,), name='Latent_Input')

sequence_length = 1
projected_dim = best_latent_dim

x = tf.keras.layers.Reshape((sequence_length, latent_dim_size))(latent_input)
x = tf.keras.layers.Dense(projected_dim)(x)

layers_after_vib = vib_attention_model.layers[vib_layer_index + 2:]

for layer in layers_after_vib:
    if isinstance(layer, tf.keras.layers.MultiHeadAttention):
         x = layer(query=x, value=x, key=x)
    else:
        x = layer(x)

x = tf.keras.layers.Flatten()(x)
l1_weight = 1e-4
composition_output = tf.keras.layers.Dense(56, activation='softmax', name="Composition_Output", activity_regularizer=tf.keras.regularizers.L1(l1_weight))(x)
hardness_output = tf.keras.layers.Dense(1, activation='relu', name="Hardness_Output")(x)

decoder_model = tf.keras.Model(inputs=latent_input, outputs=[composition_output, hardness_output], name='Decoder_Model')
decoder_model.summary()

sampled_latent_vectors = np.array(sampled_latent_vectors, dtype=np.float32)

predicted_compositions, predicted_hardness = decoder_model.predict(sampled_latent_vectors)

predicted_hardness_flat = predicted_hardness.flatten()

predicted_hardness_flat = np.abs(predicted_hardness_flat)
print("Predicted hardness (converted to positive) shape:", predicted_hardness_flat.shape)

predicted_compositions_normalized = predicted_compositions

row_sums = np.sum(predicted_compositions_normalized, axis=1)
print("Row sums after normalization (should be close to 1.0 for each row):", row_sums)

compositions_df = pd.DataFrame(
    predicted_compositions_normalized,
    columns=[f'Element_{i+1}' for i in range(predicted_compositions_normalized.shape[1])]
)
compositions_df['Predicted_Hardness'] = predicted_hardness_flat

csv_filename = 'sampled_compositions_with_hardness.csv'
compositions_df.to_csv(csv_filename, index=False)
print(f"Compositions and hardness values saved to '{csv_filename}'")


# Gradient-Based Optimization for 5 Novel Alloys
def stable_normalization(compositions, min_threshold=1e-3):
    row_sums = np.sum(compositions, axis=1, keepdims=True)
    row_sums = np.where(row_sums < min_threshold, min_threshold, row_sums)
    return compositions / row_sums

patience = 1000
best_loss = float('inf')
no_improvement_counter = 0

# Desired hardness range
min_hardness = 2200
max_hardness = 2500

num_new_alloys = 5

latent_dim_size = decoder_model.input_shape[1]
indices = np.random.choice(sampled_latent_vectors.shape[0], size=num_new_alloys, replace=False)
initial_latent_vectors = sampled_latent_vectors[indices, :]

# Optimization parameters
learning_rate = 0.001
num_iterations = 10000

latent_vectors_tf = tf.Variable(initial_latent_vectors, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

best_latent_vectors = None

for iteration in range(num_iterations):
    with tf.GradientTape() as tape:
        predicted_compositions, predicted_hardness_opt = decoder_model(latent_vectors_tf)

        predicted_hardness_opt = tf.squeeze(predicted_hardness_opt, axis=-1)

        latent_numpy = latent_vectors_tf.numpy()
        gmm_log_likelihood = gmm.score_samples(latent_numpy)
        gmm_log_likelihood = tf.convert_to_tensor(gmm_log_likelihood, dtype=tf.float32)
        gmm_penalty = -tf.reduce_mean(gmm_log_likelihood)
      
        lambda_gmm = 0.05

        latent_norm = tf.reduce_mean(tf.square(latent_vectors_tf))
        lambda_norm = 0.01

        lower_bound_penalty = tf.maximum(min_hardness - predicted_hardness_opt, 0.0)
        upper_bound_penalty = tf.maximum(predicted_hardness_opt - max_hardness, 0.0)

        target_hardness = (min_hardness + max_hardness) / 2
        mse_loss = tf.reduce_mean(tf.square(predicted_hardness_opt - target_hardness))

        penalty_loss = tf.reduce_mean(lower_bound_penalty + upper_bound_penalty)

        comp_sum = tf.reduce_sum(predicted_compositions, axis=1)
        sum_deviation = tf.reduce_mean(tf.square(comp_sum - 1.0))
        
        neg_penalty = tf.reduce_mean(tf.nn.relu(-predicted_compositions))  # ReLU to penalize only negatives
        
        lambda_sum = tf.Variable(10.0, dtype=tf.float32, trainable=False)
        lambda_neg = tf.Variable(10.0, dtype=tf.float32, trainable=False)
        
        lambda_sum_max = 100.0
        lambda_neg_max = 100.0
        lambda_growth_rate = 1.1

        loss = (mse_loss + 10.0 * penalty_loss + lambda_gmm * gmm_penalty + lambda_norm * latent_norm + lambda_sum * sum_deviation + lambda_neg * neg_penalty)

    gradients = tape.gradient(loss, [latent_vectors_tf])
    optimizer.apply_gradients(zip(gradients, [latent_vectors_tf]))

    current_loss = loss.numpy()

    if current_loss < best_loss:
        best_loss = current_loss
        best_latent_vectors = latent_vectors_tf.numpy().copy()  
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1

    if sum_deviation.numpy() > 0.01:
        lambda_sum.assign(min(lambda_sum * lambda_growth_rate, lambda_sum_max))
    if neg_penalty.numpy() > 0.001:
        lambda_neg.assign(min(lambda_neg * lambda_growth_rate, lambda_neg_max))

    if iteration % 100 == 0 or iteration == num_iterations - 1:
        print(  f"Iter {iteration:4d}: "
                f"Loss={loss.numpy():.4f}, "
                f"MSE={mse_loss.numpy():.4f}, "
                f"SumDev={sum_deviation.numpy():.5f}, "
                f"NegPen={neg_penalty.numpy():.5f}, "
                f"λ_sum={lambda_sum.numpy():.2f}, "
                f"λ_neg={lambda_neg.numpy():.2f}"
            )

    if no_improvement_counter >= patience:
        print(f"Early stopping at iteration {iteration} with best_loss={best_loss:.4f}")
        break

optimized_latent_vectors = best_latent_vectors
optimized_compositions, optimized_hardness = decoder_model.predict(optimized_latent_vectors)
optimized_hardness = np.squeeze(optimized_hardness)

valid_indices = (optimized_hardness >= min_hardness) & (optimized_hardness <= max_hardness)
filtered_compositions = optimized_compositions[valid_indices]
filtered_hardness = optimized_hardness[valid_indices]
filtered_compositions_normalized = stable_normalization(filtered_compositions)

compositions_df = pd.DataFrame(filtered_compositions_normalized,
                               columns=[f'Element_{i+1}' for i in range(filtered_compositions_normalized.shape[1])])
compositions_df['Predicted_Hardness'] = filtered_hardness
compositions_df.to_csv('filtered_compositions_with_hardness.csv', index=False)
print(f"Filtered compositions and hardness values saved to 'filtered_compositions_with_hardness.csv'")


# Visualization of new alloys in latent space
y = data['HV']

all_latent_vectors = np.vstack((
    latent_vectors,           
    sampled_latent_vectors,
    optimized_latent_vectors
))

all_labels = np.array(
    ['Original'] * original_tsne.shape[0] +
    ['Sampled'] * sampled_latent_vectors.shape[0] +
    ['Optimized'] * optimized_latent_vectors.shape[0]
)

latent_tsne = TSNE(n_components=2, random_state=42).fit_transform(all_latent_vectors)

original_predicted_hardness = np.array(y)
sampled_predicted_hardness = predicted_hardness.flatten()
optimized_predicted_hardness = optimized_hardness.flatten()

print("Original latent vectors:", latent_vectors.shape)
print("Sampled latent vectors:", sampled_latent_vectors.shape)
print("Optimized latent vectors:", optimized_latent_vectors.shape)

print("Original hardness values:", original_predicted_hardness.shape)
print("Sampled hardness values:", sampled_predicted_hardness.shape)
print("Optimized hardness values:", optimized_predicted_hardness.shape)

all_hardness_values = np.concatenate((
    original_predicted_hardness[:latent_tsne.shape[0]],
    sampled_predicted_hardness[:sampled_latent_vectors.shape[0]],
    optimized_predicted_hardness[:optimized_latent_vectors.shape[0]]
))

print("All latent vectors shape:", all_latent_vectors.shape)
print("All hardness values shape:", all_hardness_values.shape)

tsne = TSNE(n_components=2, random_state=42)
latent_tsne = tsne.fit_transform(all_latent_vectors)

print("latent_tsne shape:", latent_tsne.shape)

original_mask = all_labels == 'Original'
sampled_mask = all_labels == 'Sampled'
optimized_mask = all_labels == 'Optimized'

print("original_mask sum:", np.sum(original_mask))
print("sampled_mask sum:", np.sum(sampled_mask))
print("optimized_mask sum:", np.sum(optimized_mask))

fig, ax = plt.subplots(figsize=(12, 8))

norm = plt.Normalize(vmin=all_hardness_values.min(), vmax=all_hardness_values.max())

scatter_original = ax.scatter(
    latent_tsne[original_mask, 0],
    latent_tsne[original_mask, 1],
    c=all_hardness_values[original_mask],
    cmap='coolwarm',
    norm=norm,  
    s=180,
    marker='o',
    edgecolor='k',
    alpha=0.6,
    label='Original Alloys'
)
scatter_sampled = ax.scatter(
    latent_tsne[sampled_mask, 0],
    latent_tsne[sampled_mask, 1],
    c=all_hardness_values[sampled_mask],
    cmap='coolwarm',
    norm=norm,  
    s=180,
    marker='.',
    edgecolor='k',
    alpha=1.0,
    label='Sampled Alloys'
)
scatter_optimized = ax.scatter(
    latent_tsne[optimized_mask, 0],
    latent_tsne[optimized_mask, 1],
    c=all_hardness_values[optimized_mask],
    cmap='coolwarm',
    norm=norm,  
    s=200,
    marker='*',
    edgecolor='k',
    alpha=1.0,
    label='Optimized Alloys'
)

cbar = plt.colorbar(scatter_optimized)
cbar.set_label('Predicted Hardness', fontsize=22, weight='bold')
ax.set_xlabel('t-SNE Component 1', fontsize=22, weight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=22, weight='bold')
ax.legend(fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig("t-SNE Visualization of Latent Space with Inverse Design Samples", dpi=600, format='jpeg')
