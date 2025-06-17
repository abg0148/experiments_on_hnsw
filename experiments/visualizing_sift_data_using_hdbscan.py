import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan

from lib.vector_io import read_fvecs
from lib.paths import DATA_DIR, IMAGES_DIR

# Load the SIFT vectors
sift_vectors = read_fvecs(DATA_DIR / "siftsmall_base.fvecs")

# Run HDBSCAN clustering in high-dimensional space
clusterer = hdbscan.HDBSCAN(min_cluster_size=20)  # use higher value for meaningful structure
labels = clusterer.fit_predict(sift_vectors)

# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
sift_2d = pca.fit_transform(sift_vectors)

# Plot with cluster labels (noise labeled as -1)
plt.figure(figsize=(6, 5))
plt.scatter(sift_2d[:, 0], sift_2d[:, 1], c=labels, cmap='Spectral', s=2, alpha=0.7)
plt.title("SIFT Descriptors (PCA + HDBSCAN Clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

# Optional: save the figure
plt.savefig(IMAGES_DIR / "sift_hdbscan_clusters.png")

plt.show()
