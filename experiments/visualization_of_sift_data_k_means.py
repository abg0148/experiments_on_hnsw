from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from lib.vector_io import read_fvecs
from lib.paths import DATA_DIR

# Load the SIFT vectors
sift_vectors = read_fvecs(os.path.join(DATA_DIR, "siftsmall_base.fvecs"))

# Perform clustering in high-dimensional space
kmeans = KMeans(n_clusters=10, random_state=42).fit(sift_vectors)
labels = kmeans.labels_

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
sift_2d = pca.fit_transform(sift_vectors)

# Plot with cluster labels
plt.figure(figsize=(6,5))
plt.scatter(sift_2d[:, 0], sift_2d[:, 1], c=labels, s=1, cmap='tab10')
plt.title("SIFT (PCA + KMeans Clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
