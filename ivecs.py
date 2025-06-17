import numpy as np

def read_faiss_ivecs(path):
    data = np.fromfile(path, dtype='int32')
    k = data[0]
    return data.reshape(-1, k + 1)[:, 1:]

ivecs = read_faiss_ivecs("siftsmall_groundtruth.ivecs")

for i, row in enumerate(ivecs):
    print(f"Query {i:4d}: {row.tolist()}")
