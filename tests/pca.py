import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1️⃣ Create a 2D correlated dataset
np.random.seed(42)
n_samples = 200

# Generate x values (feature 1)
x = np.random.normal(0, 1, n_samples)
# Create y correlated with x (feature 2)
# Corellated with a noise
y = 2 * x + np.random.normal(0, 0.5, n_samples)

# Stack into a 2D dataset
X = np.column_stack((x, y))

# 2️⃣ Plot the original data
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Original data (correlated features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.show()

# 3️⃣ Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 4️⃣ Visualize principal components on original data
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)

# Mean of data (center)
mean = np.mean(X, axis=0)

# Principal components (directions)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([mean[0], mean[0] + v[0]], [mean[1], mean[1] + v[1]], linewidth=3)

plt.title("Principal Components on the original data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.show()

# 5️⃣ Compare variance explained
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 6️⃣ Reduce to 1D (keep only the first component)
pca_1d = PCA(n_components=1)
X_reduced = pca_1d.fit_transform(X)
X_reconstructed = pca_1d.inverse_transform(X_reduced)

# 7️⃣ Show the reconstruction
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="Original")
plt.scatter(
    X_reconstructed[:, 0],
    X_reconstructed[:, 1],
    alpha=0.7,
    label="Reconstructed (1D PCA)",
)
plt.legend()
plt.title("Projection onto first principal component")
plt.axis("equal")
plt.show()
