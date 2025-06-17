import os
import numpy as np
import matplotlib.pyplot as plt
from lib.vector_io import read_fvecs
from lib.paths import DATA_DIR

sift_vectors = read_fvecs(os.path.join(DATA_DIR, "siftsmall_base.fvecs"))


# method 1 -- TSNE

# from sklearn.manifold import TSNE
#
# sift_2d_tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(sift_vectors)
#
# plt.figure(figsize=(6,5))
# plt.scatter(sift_2d_tsne[:, 0], sift_2d_tsne[:, 1], s=1, alpha=0.5)
# plt.title("SIFT1M (t-SNE Projection)")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.grid(True)
# plt.show()


# method 2 -- PCA

# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# sift_2d = pca.fit_transform(sift_vectors)
#
# plt.figure(figsize=(6,5))
# plt.scatter(sift_2d[:, 0], sift_2d[:, 1], s=1, alpha=0.5)
# plt.title("SIFT1M (PCA Projection)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.show()


