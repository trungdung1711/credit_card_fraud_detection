import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Step 1: Create correlated 2D data ---
np.random.seed(0)
height = np.linspace(150, 190, 30)
weight = height * 0.5 - 25 + np.random.normal(0, 2, 30)  # roughly correlated
X = np.column_stack((height, weight))

# --- Step 2: Apply PCA (reduce 2D → 1D) ---
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# --- Step 3: Reconstruct back to 2D to visualize the info loss ---
X_reconstructed = pca.inverse_transform(X_pca)

# --- Step 4: Plot original vs PCA line ---
plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label="Original data (Height, Weight)")
plt.scatter(
    X_reconstructed[:, 0],
    X_reconstructed[:, 1],
    color="red",
    label="Reconstructed (1D→2D)",
)
plt.plot(X_reconstructed[:, 0], X_reconstructed[:, 1], color="red", alpha=0.7)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.title("PCA: Reducing 2D (Height, Weight) → 1D Principal Component")
plt.axis("equal")
plt.show()

# --- Step 5: Check how much variance we kept ---
print("Variance retained:", pca.explained_variance_ratio_[0])
