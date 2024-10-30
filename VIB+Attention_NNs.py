import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.calibration import calibration_curve
from scipy.stats import boxcox, skew, norm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import Callback
import csv

# Load the dataset
data = pd.read_csv('H_v_dataset.csv')

# Display basic statistics for the dataset
print("\nBasic Statistics:")
print(data.describe())

# Identify composition, load, and target columns
composition_cols = [col for col in data.columns if col not in ['Load', 'HV']]
load_col = 'Load'
target_col = 'HV'
print(f"\nComposition Columns: {composition_cols}")
print(f"Load Column: {load_col}")
print(f"Target Column: {target_col}")

plt.style.use('default')
sns.set_context("talk")

# Target Distribution: Hardness (HV)
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

# Scatter Plot: Load vs. Hardness
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

# PCA for Composition Data
palette = sns.color_palette("coolwarm", as_cmap=True)
pca = PCA(n_components=2)
composition_pca = pca.fit_transform(data[composition_cols])
plt.figure(figsize=(10, 7))
plt.scatter(composition_pca[:, 0], composition_pca[:, 1], c=data['HV'], cmap=palette, edgecolor='black', s=180)
plt.colorbar(label='Hardness (HV)')
plt.xlabel('Principal Component 1', fontsize=22, weight='bold')
plt.ylabel('Principal Component 2', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("PCA of Composition Data", dpi=600, format='jpeg')
plt.show()

# Separate features and target
X_composition = data[composition_cols]
X_load = data[[load_col]]
y = data['HV']

# Standardize composition and load features
scaler_comp = StandardScaler()
X_composition_scaled = scaler_comp.fit_transform(X_composition)

scaler_load = StandardScaler()
X_load_scaled = scaler_load.fit_transform(X_load)

# Split the data into training, validation, and test sets
X_comp_train, X_comp_temp, X_load_train, X_load_temp, y_train, y_temp = train_test_split(
    X_composition_scaled, X_load_scaled, y, test_size=0.2, random_state=42)

X_comp_val, X_comp_test, X_load_val, X_load_test, y_val, y_test = train_test_split(
    X_comp_temp, X_load_temp, y_temp, test_size=0.5, random_state=42)


# Examine the skewness of the target variable
skewness = skew(y)
print(f"Skewness of hardness (HV): {skewness:.2f}")

# Apply Box-Cox transformation if needed
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


## Construction of VIB+Attention Neural Network

# Custom VIB Layer
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

class AttentionScoreLogger(Callback):
    def __init__(self, comp_feature_count, output_file='attention_scores.csv'):
        super(AttentionScoreLogger, self).__init__()
        self.comp_feature_count = comp_feature_count
        self.output_file = output_file
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f"Comp_Feature_{i+1}" for i in range(comp_feature_count)] + ["Load_Feature"]
            writer.writerow(["Epoch"] + header)
    
# Model Definition
def build_vib_attention_model(input_shape_comp, input_shape_load, latent_dim=16, attention_heads=4):
    comp_input = layers.Input(shape=(input_shape_comp,), name="Composition_Input")
    x_comp = layers.Dense(64, activation="relu")(comp_input)
    x_comp = layers.BatchNormalization()(x_comp)
    x_comp = layers.Dropout(0.3)(x_comp)
    x_comp_attention, comp_attention_scores = FeatureWiseAttention(name="Comp_Attention")(x_comp)

    load_input = layers.Input(shape=(input_shape_load,), name="Load_Input")
    x_load = layers.Dense(32, activation="relu")(load_input)
    x_load = layers.BatchNormalization()(x_load)
    x_load = layers.Dropout(0.3)(x_load)
    x_load_attention, load_attention_scores = FeatureWiseAttention(name="Load_Attention")(x_load)

    combined = layers.Concatenate()([x_comp_attention, x_load_attention])
    vib_output = VIBLayer(latent_dim=latent_dim)(combined)

    reshaped_vib = layers.Reshape((1, latent_dim))(vib_output)
    attention_output = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=latent_dim)(reshaped_vib, reshaped_vib)
    attention_output = layers.Flatten()(attention_output)

    x = layers.Dense(64, activation="relu")(attention_output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(1, name="Output_Layer")(x)
    model = models.Model(inputs=[comp_input, load_input], outputs=[output, comp_attention_scores, load_attention_scores],
                         name="VIB_Attention_Model")
    return model

# Instantiate the model
vib_attention_model = build_vib_attention_model(X_comp_train.shape[1], X_load_train.shape[1])

# Adaptive Beta Callback to adjust beta dynamically based on KL divergence
class AdaptiveBetaCallback(Callback):
    def __init__(self, vib_layer, target_kl=0.1, beta_adjustment_factor=1.05):
        super(AdaptiveBetaCallback, self).__init__()
        self.vib_layer = vib_layer
        self.target_kl = target_kl
        self.beta_adjustment_factor = beta_adjustment_factor

    def on_epoch_end(self, epoch, logs=None):
        kl_loss = self.vib_layer.losses[-1] if self.vib_layer.losses else 0
        if kl_loss < self.target_kl:
            new_beta = self.vib_layer.beta * self.beta_adjustment_factor
        else:
            new_beta = self.vib_layer.beta / self.beta_adjustment_factor
        self.vib_layer.beta.assign(new_beta)
        print(f"Epoch {epoch + 1}: KL Loss = {kl_loss:.4f}, Updated Beta = {self.vib_layer.beta.numpy():.6f}")
        
        comp_attention_scores = logs.get('comp_attention_scores')
        load_attention_scores = logs.get('load_attention_scores')

        if comp_attention_scores is None or load_attention_scores is None:
            print("Warning: Attention scores are missing in logs.")
            return
    
        avg_comp_attention = np.mean(comp_attention_scores, axis=0)
        avg_load_attention = np.mean(load_attention_scores, axis=0)
    
        avg_comp_attention = avg_comp_attention.flatten() if avg_comp_attention.ndim > 1 else avg_comp_attention
        avg_load_attention = avg_load_attention.flatten() if avg_load_attention.ndim > 1 else avg_load_attention
    
        all_attention_scores = np.concatenate((avg_comp_attention, avg_load_attention), axis=0)
    
        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1] + all_attention_scores.tolist())

# Function to adjust beta based on KL loss
def adjust_beta(vib_layer, kl_loss, target_kl=0.1, beta_adjustment_factor=1.05):
    if kl_loss < target_kl:
        new_beta = vib_layer.beta * beta_adjustment_factor
    else:
        new_beta = vib_layer.beta / beta_adjustment_factor
    vib_layer.beta.assign(new_beta)

# Find the correct VIB layer dynamically
vib_layer = None
for layer in vib_attention_model.layers:
    if isinstance(layer, VIBLayer):
        vib_layer = layer
        break

# Check if VIB layer was found
if vib_layer is None:
    raise ValueError("VIB layer not found in the model.")

# Compile the model with Mean Squared Error loss
vib_attention_model.compile(optimizer="adam", loss=MeanSquaredError())


## Model Training 

train_losses = []
val_losses = []
beta_values = []

attention_logger = AttentionScoreLogger(comp_feature_count=X_comp_train.shape[1])

epochs = 2000
for epoch in range(epochs):
    history = vib_attention_model.fit(
        [X_comp_train, X_load_train], y_train,
        validation_data=([X_comp_val, X_load_val], y_val),
        epochs=500,
        callbacks=[attention_logger],
        verbose=1
    )

    train_losses.append(history.history['loss'][0])
    val_losses.append(history.history['val_loss'][0])

    kl_loss = vib_layer.kl_loss.numpy()
    adjust_beta(vib_layer, kl_loss)
    beta_values.append(vib_layer.beta.numpy())

    print(f"Epoch {epoch + 1}/{epochs} - KL Loss: {kl_loss:.4f}, Updated Beta: {vib_layer.beta.numpy():.6f}")


# Plot Training and Validation Loss with Enhanced Styling
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(train_losses, label='Training Loss', color='blue', linewidth=2, marker='o', markersize=8, markerfacecolor='blue')
ax.plot(val_losses, label='Validation Loss', color='red', linewidth=2, marker='o', markersize=8, markerfacecolor='red')
ax.fill_between(range(len(train_losses)), train_losses, val_losses, color='gray', alpha=0.2, label='Difference Area')
ax.set_xlabel('Epochs', fontsize=2, weight='bold', color='black')
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

# Plot change in beta adaptive over epochs
plt.figure(figsize=(10, 8))
plt.plot(beta_values, label=r'$\beta$ (Adaptive)', color='darkviolet', linewidth=2.5, marker='o', markersize=5, markerfacecolor='indigo')
plt.fill_between(range(len(beta_values)), beta_values, color='violet', alpha=0.2)
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

# Perform Monte-Carlo dropout for Uncertainity Estimation
def mc_dropout_predictions(model, X_comp, X_load, num_samples=100):
    predictions = []
    comp_attention_scores_list = []
    load_attention_scores_list = []

    @tf.function
    def predict_with_dropout(comp_input, load_input):
        return model([comp_input, load_input], training=True)

    for _ in range(num_samples):
        outputs = predict_with_dropout(X_comp, X_load)
        y_pred = outputs[0]
        comp_attention_scores = outputs[1]
        load_attention_scores = outputs[2]

        predictions.append(y_pred.numpy().flatten())
        comp_attention_scores_list.append(comp_attention_scores.numpy())
        load_attention_scores_list.append(load_attention_scores.numpy())

    predictions = np.array(predictions)
    comp_attention_scores_array = np.array(comp_attention_scores_list)
    load_attention_scores_array = np.array(load_attention_scores_list)
    y_pred_mean = predictions.mean(axis=0)
    y_pred_std = predictions.std(axis=0)
    comp_attention_scores_mean = comp_attention_scores_array.mean(axis=0)
    load_attention_scores_mean = load_attention_scores_array.mean(axis=0)
    return y_pred_mean, y_pred_std, comp_attention_scores_mean, load_attention_scores_mean

# Make predictions on the test set using MC Dropout
y_pred_mean, y_pred_std, comp_attention_scores_test, load_attention_scores_test = mc_dropout_predictions(
    vib_attention_model, X_comp_test, X_load_test)

if len(y_test) != len(y_pred_mean):
    min_len = min(len(y_test), len(y_pred_mean))
    y_test = y_test[:min_len]
    y_pred_mean = y_pred_mean[:min_len]
    y_pred_std = y_pred_std[:min_len]

fig, ax = plt.subplots(figsize=(10, 8))
ax.errorbar(y_test, y_pred_mean, yerr=y_pred_std, fmt='o', ecolor='red', alpha=0.8, capsize=5, 
            markerfacecolor='blue', markeredgewidth=1, markersize=10, label='Monte-Carlo Dropout Predictions')
ax.plot([0, 1800], [0, 1800], 'k--', lw=2)
ax.set_xlim(0, 1800)
ax.set_ylim(0, 1800)
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
sns.histplot(y_pred_std, bins=30, color='red', kde=True, alpha=0.7, ax=ax)
ax.set_xlabel('Prediction Uncertainty (Std Dev)', fontsize=26, weight='bold')
ax.set_ylabel('Frequency', fontsize=26, weight='bold')
ax.grid(False)
ax.tick_params(axis='both', which='major', labelsize=24)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("Distribution of Prediction Uncertainty (Standard Deviation)", dpi=600, format='jpeg')
plt.show()

# Calculate metrics for evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
mae = mean_absolute_error(y_test, y_pred_mean)
mape = mean_absolute_percentage_error(y_test, y_pred_mean)
r2 = r2_score(y_test, y_pred_mean)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MAPE: {mape:.4f}")
print(f"Test R² Score: {r2:.4f}")


# Attention Weights
_, comp_attention_scores_final, load_attention_scores_final = vib_attention_model.predict([X_comp_test, X_load_test])
comp_attention_scores_final = comp_attention_scores_final[:, :49]

avg_comp_attention_scores = np.mean(comp_attention_scores_final, axis=0)
feature_names = ['Ag', 'Al', 'Au', 'B', 'Be', 'C', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Fe', 'Ga', 'Gd', 'Hf', 'Ho', 'In', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pd', 'Pr', 'Pt', 'Re', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Ti', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

plt.figure(figsize=(14, 6))
sns.barplot(x=feature_names, y=avg_comp_attention_scores, palette="plasma")
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

## Using VIB to geneerate latent space and Inverse Design
latent_model = tf.keras.Model(inputs=vib_attention_model.input, 
                              outputs=vib_attention_model.get_layer('vib_layer_1').output)

latent_vectors = latent_model.predict([X_composition_scaled, X_load_scaled])

# Range of possible cluster numbers for the Elbow Method
cluster_range = range(1, 11)
wcss = [] 

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(latent_vectors)
    wcss.append(kmeans.inertia_)  

plt.figure(figsize=(10, 8))
plt.plot(cluster_range, wcss, marker='o', color='b', markersize=15, linestyle='--', linewidth=2)
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

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=clusters, cmap='plasma', s=200, edgecolor='k', alpha=0.7)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_xlabel('t-SNE Component 1', fontsize=22, weight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("t-SNE Visualization of Latent Space with K-Means Clustering", dpi=600, format='jpeg')
plt.show()

# t-SNE with Color-Coding by Hardness
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=y, cmap='coolwarm', s=200, edgecolor='k', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('Hardness Value', fontsize=22, weight='bold')
ax.set_xlabel('t-SNE Component 1', fontsize=22, weight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=22, weight='bold')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
plt.savefig("t-SNE Visualization of Latent Space (Color-Coded by Hardness)", dpi=600, format='jpeg')
plt.show()

# Function to sample new latent vectors around cluster centers
def sample_latent_space(num_samples_per_cluster=100, std_dev=0.5):
    cluster_centers = kmeans.cluster_centers_
    sampled_latent_vectors = []
    for center in cluster_centers:
        samples = np.random.normal(loc=center, scale=std_dev, size=(num_samples_per_cluster, center.shape[0]))
        sampled_latent_vectors.append(samples)
    sampled_latent_vectors = np.vstack(sampled_latent_vectors)
    return sampled_latent_vectors

# Generate sampled latent vectors
num_samples_per_cluster = 100
sampled_latent_vectors = sample_latent_space(num_samples_per_cluster=num_samples_per_cluster, std_dev=0.5)

# Decoder model to reconstruct new compositions
vib_layer_name = 'vib_layer_1' 
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

x = latent_input
x = tf.keras.layers.Reshape((1, latent_dim_size))(x)

layers_after_vib = vib_attention_model.layers[vib_layer_index + 2:]

for layer in layers_after_vib:
    if isinstance(layer, tf.keras.layers.MultiHeadAttention):
        x = layer(query=x, value=x, key=x)
    else:
        x = layer(x)

decoder_model = tf.keras.Model(inputs=latent_input, outputs=x, name='Decoder_Model')

predicted_outputs = decoder_model.predict(sampled_latent_vectors)

if isinstance(predicted_outputs, list) or isinstance(predicted_outputs, tuple):
    predicted_compositions = predicted_outputs[0]
    predicted_hardness = predicted_outputs[1].flatten()
else:
    predicted_compositions = predicted_outputs
    predicted_hardness = predicted_compositions.flatten()

predicted_compositions_normalized = np.apply_along_axis(lambda x: x / np.sum(x), 1, predicted_compositions)

row_sums = np.sum(predicted_compositions_normalized, axis=1)
print("Row sums after normalization (should be 1.0 for each row):", row_sums)

compositions_df = pd.DataFrame(predicted_compositions_normalized, columns=[f'Element_{i+1}' for i in range(predicted_compositions_normalized.shape[1])])
compositions_df['Predicted_Hardness'] = predicted_hardness
compositions_df.to_csv('decoded_compositions_with_hardness.csv', index=False)
print(f"Decoded compositions and hardness values saved to 'decoded_compositions_with_hardness.csv'")

## Optimization of synthetic compositions
desired_hardness = 2000
num_new_alloys = 20
latent_dim_size = decoder_model.input_shape[1]  # Get latent dimension size from the decoder model
initial_latent_vectors = 0.1 * np.random.normal(size=(num_new_alloys, latent_dim_size))
learning_rate = 0.00001
num_iterations = 10000000

latent_vectors_tf = tf.Variable(initial_latent_vectors, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for iteration in range(num_iterations):
    with tf.GradientTape() as tape:
        predicted_outputs = decoder_model(latent_vectors_tf)
        if isinstance(predicted_outputs, list) or isinstance(predicted_outputs, tuple):
            predicted_hardness_opt = predicted_outputs[0]
        else:
            predicted_hardness_opt = predicted_outputs
        loss = tf.reduce_mean(tf.square(predicted_hardness_opt - desired_hardness))

    gradients = tape.gradient(loss, [latent_vectors_tf])
    clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(clipped_gradients, [latent_vectors_tf]))
    
    optimizer.apply_gradients(zip(gradients, [latent_vectors_tf]))

    if iteration % 100 == 0 or iteration == num_iterations - 1:
        current_loss = loss.numpy()
        print(f"Iteration {iteration}: Loss = {current_loss:.4f}")

optimized_latent_vectors = latent_vectors_tf.numpy()
optimized_predicted_outputs = decoder_model.predict(optimized_latent_vectors)
if isinstance(optimized_predicted_outputs, list) or isinstance(optimized_predicted_outputs, tuple):
    optimized_predicted_hardness = optimized_predicted_outputs[0].flatten()
else:
    optimized_predicted_hardness = optimized_predicted_outputs.flatten()

print(f"Optimized predicted hardness values:\n{optimized_predicted_hardness}")

# Combine all latent vectors for visualization
all_latent_vectors = np.vstack((latent_vectors, sampled_latent_vectors, optimized_latent_vectors))
all_labels = np.array(
    ['Original'] * latent_vectors.shape[0] +
    ['Sampled'] * sampled_latent_vectors.shape[0] +
    ['Optimized'] * optimized_latent_vectors.shape[0]
)

# Predict hardness values for all vectors
original_predicted_hardness = y.to_numpy()
sampled_predicted_hardness = predicted_hardness.flatten()
optimized_predicted_hardness = optimized_predicted_outputs.flatten()

# Combine all hardness values
all_hardness_values = np.concatenate((
    original_predicted_hardness,
    sampled_predicted_hardness,
    optimized_predicted_hardness
))

# Apply t-SNE to all latent vectors
tsne = TSNE(n_components=2, random_state=42)
latent_tsne = tsne.fit_transform(all_latent_vectors)
print("latent_tsne shape:", latent_tsne.shape)

original_mask = all_labels == 'Original'
sampled_mask = all_labels == 'Sampled'
optimized_mask = all_labels == 'Optimized'
print("original_mask sum:", np.sum(original_mask))
print("sampled_mask sum:", np.sum(sampled_mask))
print("optimized_mask sum:", np.sum(optimized_mask))

# Visualization
fig, ax = plt.subplots(figsize=(12, 9))

# Scatter plot for original data points
scatter_original = ax.scatter(
    latent_tsne[original_mask, 0],
    latent_tsne[original_mask, 1],
    c=all_hardness_values[original_mask],
    cmap='coolwarm',
    s=180,
    marker='o',
    edgecolor='k',
    alpha=0.8,
    label='Original Alloys'
)

# Scatter plot for sampled data points
scatter_sampled = ax.scatter(
    latent_tsne[sampled_mask, 0],
    latent_tsne[sampled_mask, 1],
    c=all_hardness_values[sampled_mask],
    cmap='coolwarm',
    s=150,
    marker='p',
    edgecolor='k',
    alpha=0.6,
    label='Sampled Alloys'
)

# Scatter plot for optimized latent vectors
scatter_optimized = ax.scatter(
    latent_tsne[optimized_mask, 0],
    latent_tsne[optimized_mask, 1],
    c=all_hardness_values[optimized_mask],
    cmap='coolwarm',
    s=150,
    marker='*',
    edgecolor='k',
    alpha=0.6,
    label='Optimized Alloys'
)

# Colorbar representing hardness values
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
plt.show()


##Integrated gradients for the VIB+Attention NNs
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

comp_baseline = tf.zeros(shape=(1, X_comp_test.shape[1]))
load_baseline = tf.zeros(shape=(1, X_load_test.shape[1]))
sample_idx = 0 
comp_input = tf.expand_dims(X_comp_test[sample_idx], axis=0)
load_input = tf.expand_dims(X_load_test[sample_idx], axis=0)
baseline_inputs = [comp_baseline, load_baseline]
sample_inputs = [comp_input, load_input]

comp_input_tf = tf.cast(comp_input, tf.float32)
load_input_tf = tf.cast(load_input, tf.float32)
comp_baseline_tf = tf.cast(comp_baseline, tf.float32)
load_baseline_tf = tf.cast(load_baseline, tf.float32)

ig_comp = integrated_gradients(vib_attention_model, [comp_baseline_tf, load_baseline_tf],
                               [comp_input_tf, load_input_tf])

ig_comp_values = ig_comp[0].numpy().flatten()
feature_names = ['Ag', 'Al', 'Au', 'B', 'Be', 'C', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Dy', 'Er', 'Fe', 'Ga', 'Gd', 'Hf', 'Ho', 'In', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pb', 'Pd', 'Pr', 'Pt', 'Re', 'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Ti', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

colors = sns.color_palette("mako", len(ig_comp_values))
plt.figure(figsize=(20, 14))
bars = plt.bar(feature_names, ig_comp_values, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Compositional Constituent', fontsize=28, weight='bold', labelpad=10, color='black')
plt.ylabel('Attribution Value', fontsize=28, weight='bold', labelpad=10, color='black')
plt.xticks(range(len(ig_comp_values)), feature_names, rotation=90, fontsize=26, color='black')
plt.yticks(fontsize=26)
plt.grid(False)
for spine in plt.gca().spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3.5)
plt.tight_layout()
plt.savefig("Integrated Gradients for Composition Features", dpi=600, format='jpeg')
plt.show()

## Reliability Diagram
confidence_levels = np.linspace(0, 1.0, 10)
coverage_probabilities = []

# Compute coverage probabilities for each confidence level
for conf_level in confidence_levels:
    lower_bound = y_pred_mean - conf_level * y_pred_std
    upper_bound = y_pred_mean + conf_level * y_pred_std
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

## Calibration Curve
pred_bins = np.linspace(y_pred_mean.min(), y_pred_mean.max(), num=10)
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
