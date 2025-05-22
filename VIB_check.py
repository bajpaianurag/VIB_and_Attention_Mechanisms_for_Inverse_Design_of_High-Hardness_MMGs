import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from scipy.stats import wasserstein_distance

data = pd.read_csv('H_v_dataset.csv')
composition_cols = [col for col in data.columns if col not in ['Load', 'HV']]
load_col = 'Load'
target_col = 'HV'
latent_dim = 16  

X_composition = data[composition_cols]
X_load = data[[load_col]]
y = data['HV']

scaler_comp = StandardScaler()
X_composition_scaled = scaler_comp.fit_transform(X_composition)

scaler_load = StandardScaler()
X_load_scaled = scaler_load.fit_transform(X_load)

X_comp_train, X_comp_temp, X_load_train, X_load_temp, y_train, y_temp = train_test_split(
    X_composition_scaled, X_load_scaled, y, test_size=0.25, random_state=42)

X_comp_val, X_comp_test, X_load_val, X_load_test, y_val, y_test = train_test_split(
    X_comp_temp, X_load_temp, y_temp, test_size=0.5, random_state=42)

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
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        self.kl_loss.assign(kl_loss)
        self.add_loss(self.beta * self.kl_loss)
        self.add_metric(self.kl_loss, name="kl_divergence", aggregation="mean")
        return latent

class AdaptiveBetaCallback(tf.keras.callbacks.Callback):
    def __init__(self, vib_layer, target_kl=0.1, beta_adjustment_factor=1.05):
        super().__init__()
        self.vib_layer = vib_layer
        self.target_kl = target_kl
        self.beta_adjustment_factor = beta_adjustment_factor

    def on_epoch_end(self, epoch, logs=None):
        current_kl = self.vib_layer.kl_loss.numpy()
        if current_kl < self.target_kl:
            new_beta = self.vib_layer.beta * self.beta_adjustment_factor
        else:
            new_beta = self.vib_layer.beta / self.beta_adjustment_factor
        self.vib_layer.beta.assign(new_beta)
        print(f"[Epoch {epoch+1}] KL = {current_kl:.4f}, Updated β = {self.vib_layer.beta.numpy():.6f}")


comp_input = layers.Input(shape=(X_comp_train.shape[1],), name="Composition_Input")
load_input = layers.Input(shape=(X_load_train.shape[1],), name="Load_Input")
x_comp = layers.Dense(128, activation="relu")(comp_input)
x_comp = layers.BatchNormalization()(x_comp)
x_load = layers.Dense(64, activation="relu")(load_input)
x_load = layers.BatchNormalization()(x_load)
combined = layers.Concatenate()([x_comp, x_load])
vib_output = VIBLayer(latent_dim=latent_dim)(combined)

reconstructed_comp = layers.Dense(X_comp_train.shape[1], activation="linear", name="Reconstructed_Composition")(vib_output)
reconstructed_load = layers.Dense(X_load_train.shape[1], activation="linear", name="Reconstructed_Load")(vib_output)
predicted_hv = layers.Dense(1, activation="linear", name="Predicted_Hardness")(vib_output)

model = models.Model(inputs=[comp_input, load_input], outputs=[predicted_hv, reconstructed_comp, reconstructed_load])

vib_layer = None
for layer in model.layers:
    if isinstance(layer, VIBLayer):
        vib_layer = layer
        break

if vib_layer is None:
    raise ValueError("VIBLayer not found in the model.")

model.compile(optimizer="adam",
              loss={"Predicted_Hardness": "mse", "Reconstructed_Composition": "mse", "Reconstructed_Load": "mse"},
              loss_weights={"Predicted_Hardness": 1.0, "Reconstructed_Composition": 0.5, "Reconstructed_Load": 0.5})

adaptive_beta = AdaptiveBetaCallback(vib_layer)
history = model.fit([X_comp_train, X_load_train], [y_train, X_comp_train, X_load_train],
                    validation_data=([X_comp_val, X_load_val], [y_val, X_comp_val, X_load_val]),
                    epochs=1000, callbacks=[adaptive_beta], batch_size=32)

y_pred, recon_comp_scaled, recon_load_scaled = model.predict([X_comp_test, X_load_test])

# Inverse transform composition and load
recon_comp = scaler_comp.inverse_transform(recon_comp_scaled)
true_comp  = scaler_comp.inverse_transform(X_comp_test)

recon_load = scaler_load.inverse_transform(recon_load_scaled)
true_load  = scaler_load.inverse_transform(X_load_test)

comp_mae = mean_absolute_error(true_comp, recon_comp)
load_mae = mean_absolute_error(true_load, recon_load)

print(f"Reconstruction MAE (Composition): {comp_mae:.4f}")
print(f"Reconstruction MAE (Load): {load_mae:.4f}")

mae_per_element = np.mean(np.abs(true_comp - recon_comp), axis=0)
for name, err in zip(composition_cols, mae_per_element):
    print(f"{name}: MAE = {err:.4f}")

sns.set_style("whitegrid")
plt.figure(figsize=(10, 8), dpi=600)
colors = ["crimson", "slateblue"]

plt.plot(history.history['val_Reconstructed_Composition_loss'], 
         label="Validation Reconstruction Loss (Composition)", 
         color=colors[0], linestyle='-', linewidth=2.5)
plt.plot(history.history['val_Reconstructed_Load_loss'], 
         label="Validation Reconstruction Loss (Load)", 
         color=colors[1], linestyle='-', linewidth=2.5)
plt.xlabel("Epochs", fontsize=16, fontweight='bold', labelpad=10)
plt.ylabel("Loss", fontsize=16, fontweight='bold', labelpad=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14, frameon=True, loc="best")
plt.grid(False)
plt.savefig("Reconstruction_Loss.jpeg", dpi=600, bbox_inches='tight', transparent=True)
plt.show()


plt.plot(history.history['kl_divergence'], label="KL Divergence")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("KL Loss")
plt.title("KL Divergence over Epochs")
plt.show()

threshold_high = np.percentile(y_test, 75)
threshold_low  = np.percentile(y_test, 25)

high_mask = y_test >= threshold_high
low_mask  = y_test <= threshold_low

latent_high = latent_vectors[high_mask]
latent_low  = latent_vectors[low_mask]

# Calculate and plot Wasserstein distance
wd_per_dim = []
for i in range(latent_vectors.shape[1]):
    wd = wasserstein_distance(latent_high[:, i], latent_low[:, i])
    wd_per_dim.append(wd)

plt.figure(figsize=(12, 6))
plt.bar(np.arange(1, len(wd_per_dim)+1), wd_per_dim, color='coral')
plt.xlabel("Latent Dimension", fontsize=14, weight='bold')
plt.ylabel("Wasserstein Distance (High vs Low HV)", fontsize=14, weight='bold')
plt.title("Latent Space Separation across Dimensions", fontsize=16, weight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("wasserstein_distance_latent.png", dpi=600)
plt.show()
