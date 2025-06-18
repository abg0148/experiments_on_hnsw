# Visualizing SIFT Descriptors with t-SNE, PCA, and HDBSCAN

This project visualizes high-dimensional SIFT descriptors using dimensionality reduction techniques like **t-SNE**, **PCA**, and **UMAP**, combined with clustering using **HDBSCAN** and **K Means Clustering**. It helps reveal the structure and distribution of the dataset when projected onto 2D space, and explores how unsupervised clustering can expose dominant groupings within local image features.

## Dataset

* **File:** `siftsmall_base.fvecs`
* **Format:** Custom binary `.fvecs` format
* **Content:** Each vector is preceded by an int32 indicating its dimensionality (typically 128 for SIFT)

## Requirements

Install the required Python libraries using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## How It Works

### Step 1: Load SIFT Vectors

```python
def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
        if data.size == 0:
            return np.array([], dtype=np.float32).reshape(0, 0)
        dim = data[0]
    data = np.fromfile(file_path, dtype=np.float32)
    return data.reshape(-1, dim + 1)[:, 1:]
```

### Step 2: t-SNE Visualization

```python
from sklearn.manifold import TSNE
sift_2d_tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(sift_vectors)
```

### Step 3: PCA Visualization (Alternative)

```python
from sklearn.decomposition import PCA
sift_2d = PCA(n_components=2).fit_transform(sift_vectors)
```

### Step 4: Plotting

```python
plt.scatter(sift_2d_tsne[:, 0], sift_2d_tsne[:, 1], s=1, alpha=0.5)
plt.title("SIFT1M (t-SNE Projection)")
```

To visualize PCA results, just replace `sift_2d_tsne` with `sift_2d` in the plotting code.

## Output

The script generates a 2D scatter plot of the descriptors using either t-SNE or PCA. t-SNE typically shows local clusters more clearly, while PCA reflects variance direction.

## 📊 t-SNE Visualization

![t-SNE Projection](../images/tsne_projection.png)

## 🔍 PCA Visualization

![t-SNE Projection](../images/pca_projection.png)

---

## 🔎 PCA + KMeans Clustering

To better understand potential patterns in the high-dimensional SIFT descriptors, we applied **KMeans clustering** directly in the original 128D space and then projected the results using **PCA** for visualization.

KMeans is a well-known centroid-based clustering algorithm that works best when the data clusters are globular and well-separated. It is useful as a quick baseline for understanding separability and distribution.

### 🔹 Why PCA + KMeans?

* **Clustering in original space:** ensures true structural grouping is attempted before any transformation.
* **PCA for 2D plotting:** helps in visual inspection of KMeans behavior and cluster compactness.

### 🔍 Observations:

* The PCA + KMeans plot reveals clear separation for several clusters.
* Some cluster overlaps are visible, suggesting PCA’s linear projection can’t fully separate them.
* Cluster centroids seem aligned along major variance axes, reflecting KMeans’ assumptions.

This clustering serves as a helpful contrast to HDBSCAN which follows.

![PCA Projection](../images/pca_kmeans_clusters.png)

---

### 🧐 Clustering with HDBSCAN

Beyond visualization, we applied **HDBSCAN** (a density-based clustering algorithm) to group similar SIFT descriptors based on their structure in high-dimensional space. Clustering was performed in the original 128D feature space, and the results were visualized using both **PCA** and **UMAP** projections.

### 📍 Why HDBSCAN?

Unlike KMeans, HDBSCAN:

* Does **not require specifying the number of clusters**
* Can detect **clusters of varying density and shape**
* Automatically labels **outliers** as noise (`label = -1`)

This makes it highly suitable for unsupervised feature datasets like SIFT.

---

### 📉 PCA + HDBSCAN

![PCA + HDBSCAN](../images/sift_hdbscan_clusters.png)

When visualized with PCA, HDBSCAN identifies two major clusters with some noise points (in red). However, since PCA is linear, the separation is not perfect — some overlap exists between clusters.

---

### 🌐 UMAP + HDBSCAN

![UMAP + HDBSCAN](../images/umap_hdbscan_clusters.png)

Using UMAP, we observe **clearer cluster separation**. The two dominant groups are well isolated along the UMAP-1 axis. Outliers (red points) are neatly pushed to cluster borders or sparse regions. This result confirms that **UMAP preserves local structure** more effectively and helps HDBSCAN express clustering boundaries more naturally.

---

### 🧐 Interpretation

* **Two primary clusters** likely reflect dominant gradient orientations or patch-level structures inherent to the SIFT algorithm.
* **Red noise points** suggest some descriptors are outliers or ambiguously positioned between clusters.
* UMAP is clearly better at revealing the **true geometry** of the data, complementing HDBSCAN's structure-aware clustering.

## License

This project is released under the MIT License.

## 🔗 Related Notebooks

Explore the interactive workflows used to generate and interpret the visualizations:

* [SIFT Visualization: (TSNE & PCA)](../experiments/notebooks/visualize_sift_tsne_pca.ipynb)
* [SIFT Visualization: (KNN + PCA)](../experiments/notebooks/visualize_sift_knn_pca_clustering.ipynb)
* [SIFT Visualization: Clustering with HDBSCAN + PCA](../experiments/notebooks/hdbscan_pca_clustering.ipynb)
* [SIFT Visualization: Clustering with HDBSCAN + UMAP](../experiments/notebooks/hdbscan_umap_clustering.ipynb)

## Author

Abhinav Gupta

---
