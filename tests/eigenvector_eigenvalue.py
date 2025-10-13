import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define a transformation matrix A ---
A = np.array([[2, 0], [0, 3]])

# --- Step 2: Compute eigenvalues and eigenvectors ---
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# --- Step 3: Visualize the transformation ---
# Create some random 2D vectors
vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]])

# Apply the transformation A to each vector
transformed = vectors @ A.T

# --- Step 4: Plot ---
plt.figure(figsize=(7, 7))
plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)

# Plot original vectors (blue)
for v in vectors:
    plt.quiver(
        0,
        0,
        v[0],
        v[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.5,
    )

# Plot transformed vectors (red)
for t in transformed:
    plt.quiver(
        0, 0, t[0], t[1], angles="xy", scale_units="xy", scale=1, color="red", alpha=0.5
    )

# Plot eigenvectors (green)
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    plt.quiver(
        0,
        0,
        v[0] * eigenvalues[i],
        v[1] * eigenvalues[i],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.005,
        label=f"Eigenvector {i+1}",
    )

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect("equal")
plt.legend()
plt.title("Blue: Original  •  Red: Transformed  •  Green: Eigenvectors")
plt.show()
