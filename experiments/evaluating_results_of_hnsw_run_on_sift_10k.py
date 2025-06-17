from lib.paths import DATA_DIR, RESULTS_DIR
from lib.vector_io import read_ivecs
from lib.metrics import recall_at_k

# Load files using project-relative paths
gt = read_ivecs(DATA_DIR / "siftsmall_groundtruth.ivecs")
pred = read_ivecs(RESULTS_DIR / "sift_10k_hnsw_results.ivecs")

# Check shape
print("Prediction shape:", pred.shape)
print("Ground truth shape:", gt.shape)

# Evaluate recall at various k
for k in [1, 10, 100]:
    r = recall_at_k(pred, gt, k)
    print(f"Recall@{k}: {r:.4f}")
