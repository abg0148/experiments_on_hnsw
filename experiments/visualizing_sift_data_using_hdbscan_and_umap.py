import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap
from lib.vector_io import read_fvecs
from lib.paths import DATA_DIR

# Load SIFT descriptors
sift_vectors = read_fvecs(DATA_DIR / "siftsmall_base.fvecs")

# Cluster in high-dimensional space
clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
labels = clusterer.fit_predict(sift_vectors)

# Reduce to 2D using UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
sift_2d = reducer.fit_transform(sift_vectors)

# Plot
plt.figure(figsize=(6, 5))
plt.scatter(sift_2d[:, 0], sift_2d[:, 1], c=labels, cmap='Spectral', s=2, alpha=0.7)
plt.title("SIFT Descriptors (UMAP + HDBSCAN)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.show()
