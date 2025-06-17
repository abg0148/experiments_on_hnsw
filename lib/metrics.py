import faiss
import numpy as np

def recall_at_k(pred, gt, k):
    if pred.shape[1] < k or gt.shape[1] < k:
        raise ValueError(f"Not enough columns to compute recall@{k}")
    pred = pred[:, :k].astype('int32')
    gt = gt[:, :k].astype('int32')
    hits = faiss.eval_intersection(pred, gt)
    return hits / (pred.shape[0] * k)
