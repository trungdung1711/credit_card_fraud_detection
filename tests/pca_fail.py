import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(0)
# Two independent features
x1 = np.random.normal(0, 1, 200)
x2 = np.random.normal(0, 1, 200)
X = np.column_stack((x1, x2))

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original data")
plt.scatter(
    pca.inverse_transform(X_pca)[:, 0],
    pca.inverse_transform(X_pca)[:, 1],
    color="red",
    label="After 1D PCA projection",
)
plt.legend()
plt.title("PCA on Independent Features")
plt.axis("equal")
plt.show()

print("Variance retained (1D):", pca.explained_variance_ratio_[0])
