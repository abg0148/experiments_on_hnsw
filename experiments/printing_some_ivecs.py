from lib.vector_io import read_ivecs
from lib.paths import DATA_DIR

ivecs = read_ivecs(DATA_DIR / "siftsmall_groundtruth.ivecs")

for i, row in enumerate(ivecs):
    print(f"Query {i:4d}: {row.tolist()}")
