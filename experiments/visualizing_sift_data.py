import numpy as np

def read_fvecs(file_path):
    """Reads a .fvecs file into a NumPy array of shape (n, d)"""
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
        if data.size == 0:
            return np.array([], dtype=np.float32).reshape(0, 0)
        dim = data[0]
    data = np.fromfile(file_path, dtype=np.float32)
    return data.reshape(-1, dim + 1)[:, 1:]

sift_vectors = read_fvecs("../data/siftsmall_base.fvecs")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sift_2d_tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(sift_vectors)

plt.figure(figsize=(6,5))
plt.scatter(sift_2d_tsne[:, 0], sift_2d_tsne[:, 1], s=1, alpha=0.5)
plt.title("SIFT1M (t-SNE Projection)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()


# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# sift_2d = pca.fit_transform(sift_vectors)
#

#
# plt.figure(figsize=(6,5))
# plt.scatter(sift_2d[:, 0], sift_2d[:, 1], s=1, alpha=0.5)
# plt.title("SIFT1M (PCA Projection)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.show()


