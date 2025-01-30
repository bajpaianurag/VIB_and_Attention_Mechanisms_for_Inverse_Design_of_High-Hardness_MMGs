#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from scipy.stats import wasserstein_distance


# In[45]:


# Load the dataset
data = pd.read_csv('H_v_dataset.csv')


# In[46]:


# Preprocess the data
composition_cols = [col for col in data.columns if col not in ['Load', 'HV']]
load_col = 'Load'
target_col = 'HV'

X_composition = data[composition_cols]
X_load = data[[load_col]]
y = data['HV']


# In[47]:


scaler_comp = StandardScaler()
X_composition_scaled = scaler_comp.fit_transform(X_composition)

scaler_load = StandardScaler()
X_load_scaled = scaler_load.fit_transform(X_load)

X_comp_train, X_comp_temp, X_load_train, X_load_temp, y_train, y_temp = train_test_split(
    X_composition_scaled, X_load_scaled, y, test_size=0.25, random_state=42)

X_comp_val, X_comp_test, X_load_val, X_load_test, y_val, y_test = train_test_split(
    X_comp_temp, X_load_temp, y_temp, test_size=0.5, random_state=42)


# In[48]:


# Define VIBLayer with Reconstruction Loss
class VIBLayer(layers.Layer):
    def __init__(self, latent_dim=16, **kwargs):
        super(VIBLayer, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = tf.Variable(1e-3, trainable=False, dtype=tf.float32)
        self.mean_dense = layers.Dense(self.latent_dim)
        self.log_var_dense = layers.Dense(self.latent_dim)

    def call(self, inputs):
        mean = self.mean_dense(inputs)
        log_var = self.log_var_dense(inputs)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        latent = mean + tf.exp(0.5 * log_var) * epsilon
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        self.add_loss(self.beta * kl_loss)
        return latent


# In[49]:


# Define the model
comp_input = layers.Input(shape=(X_comp_train.shape[1],), name="Composition_Input")
load_input = layers.Input(shape=(X_load_train.shape[1],), name="Load_Input")

# Composition Branch
x_comp = layers.Dense(128, activation="relu")(comp_input)
x_comp = layers.BatchNormalization()(x_comp)

# Load Branch
x_load = layers.Dense(64, activation="relu")(load_input)
x_load = layers.BatchNormalization()(x_load)

# Concatenate and Pass through VIB Layer
combined = layers.Concatenate()([x_comp, x_load])
vib_output = VIBLayer(latent_dim=16)(combined)

# Decoder for Reconstruction
reconstructed_comp = layers.Dense(X_comp_train.shape[1], activation="linear", name="Reconstructed_Composition")(vib_output)
reconstructed_load = layers.Dense(X_load_train.shape[1], activation="linear", name="Reconstructed_Load")(vib_output)
predicted_hv = layers.Dense(1, activation="linear", name="Predicted_Hardness")(vib_output)

model = models.Model(inputs=[comp_input, load_input], outputs=[predicted_hv, reconstructed_comp, reconstructed_load])
model.compile(optimizer="adam",
              loss={"Predicted_Hardness": "mse", "Reconstructed_Composition": "mse", "Reconstructed_Load": "mse"},
              loss_weights={"Predicted_Hardness": 1.0, "Reconstructed_Composition": 0.5, "Reconstructed_Load": 0.5})


# In[ ]:


# Train the model
history = model.fit([X_comp_train, X_load_train], [y_train, X_comp_train, X_load_train],
                    validation_data=([X_comp_val, X_load_val], [y_val, X_comp_val, X_load_val]),
                    epochs=1000, batch_size=32)


# In[56]:


# Set seaborn style for high-quality plots
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8), dpi=600)

# Define colors for better distinction
colors = ["crimson", "slateblue"]

# Plot with improved visualization
plt.plot(history.history['val_Reconstructed_Composition_loss'], 
         label="Validation Reconstruction Loss (Composition)", 
         color=colors[0], linestyle='-', linewidth=2.5)

plt.plot(history.history['val_Reconstructed_Load_loss'], 
         label="Validation Reconstruction Loss (Load)", 
         color=colors[1], linestyle='-', linewidth=2.5)

# Labels and title
plt.xlabel("Epochs", fontsize=16, fontweight='bold', labelpad=10)
plt.ylabel("Loss", fontsize=16, fontweight='bold', labelpad=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14, frameon=True, loc="best")
plt.grid(False)

plt.savefig("Reconstruction_Loss.jpeg", dpi=600, bbox_inches='tight', transparent=True)
plt.show()

