The project aims to design and optimize new alloy compositions, specifically focusing on achieving high hardness values for multicomponent metallic glasses or alloys. Leveraging Variational Information Bottleneck (VIB) and attention-based neural networks, the approach combines composition and load data to predict hardness while also generating new, optimized compositions based on latent space exploration.

Model Architecture:

Variational Information Bottleneck (VIB) Layer: This layer encourages the model to capture only the most relevant information from the input by balancing reconstruction loss with a KL-divergence regularization term, controlled by the beta parameter. KL Annealing and Dynamic Beta Adjustment: These techniques were used to gradually increase the influence of KL divergence during training, improving the model's robustness in learning a compact and relevant latent representation.

Attention Mechanism: 
Feature-Wise Attention: A custom attention mechanism is applied separately to the composition and load features. This helps the model focus on the most influential features, improving interpretability by providing attention scores for each feature.
Multi-Head Attention: After the VIB layer, multi-head attention is used to capture dependencies within the latent space, further refining the information retained in the latent vectors.
Output Layer: Predicts the hardness (HV) based on the latent representations, with the entire network optimized using mean squared error (MSE) as the loss function.

Training Process:
Loss Tracking: Training and validation losses are tracked to monitor model convergence. Additionally, beta values and KL loss are recorded across epochs to observe the effect of KL divergence.
Attention Score Logging: During each epoch, attention scores for composition features are logged, allowing for post-training analysis of feature importance.
Early Stopping and Learning Rate Scheduling: These techniques were used to prevent overfitting and ensure efficient convergence.

Model Interpretability:
Integrated Gradients (IG): This method was used to understand feature attributions, providing a measure of how much each input feature contributes to the model's output.
Attention Visualization: Attention scores were visualized in heatmaps and bar plots, showing the importance of each feature across samples.

Inverse Design for New Alloy Compositions:

Latent Space Sampling:
The latent space of the VIB layer was explored by sampling around cluster centers derived from high-hardness samples.
New latent vectors were generated by sampling within these clusters, ensuring the generated compositions were likely to achieve high hardness values.

Optimization of Latent Vectors:
Starting from an initial set of latent vectors around high-hardness points, a gradient-based optimization loop was employed to minimize the mean squared error between predicted hardness and the desired hardness target (e.g., 2000 HV).
Latent Vector Constraints: Latent vectors were constrained to ensure that the decoded compositions summed to 1, maintaining physical validity.

Decoding Latent Vectors to Compositions:
The optimized latent vectors were decoded back to composition space using the decoder model.
The decoded compositions were checked to ensure they were physically valid (e.g., fractions summing to 1).
Export of New Alloy Compositions: The final, optimized compositions were exported to a CSV file for further experimental testing or analysis.

Visualization and Analysis:
Calibration Curves: Plotted to assess the reliability of probabilistic predictions and visualize model calibration.
Loss Curves: Training and validation loss curves were plotted with enhanced aesthetics to track model convergence and identify any signs of overfitting.
Attention Score Bar Charts: Showed the average attention weights for each composition feature, providing insights into which elements were most important for hardness prediction.
t-SNE and Latent Space Visualization: Used to visualize the distribution of both original and optimized samples in latent space, aiding in understanding the structure of the learned latent representations.
