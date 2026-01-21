

import os
import numpy as np
import faiss

from src.config.settings import settings

# -----------------------------
# Configuration
# -----------------------------

DOCS_DIR = str(settings.DOCS_DIR)
INDEX_DIR = str(settings.INDEX_DIR)

# -----------------------------
# Load or create FAISS index
# -----------------------------
def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

### Use This, IndexIVFFlat support filter by ids
def faiss_index_setup(embedder, data):
    documents = [i[-1] for i in data]
    ids = np.array([i[0] for i in data], dtype=np.int64)

    if os.path.exists(INDEX_DIR) and os.path.exists(DOCS_DIR):
        print("Loading existing FAISS index...")
        faiss_index = faiss.read_index(INDEX_DIR)
        data = np.load(DOCS_DIR, allow_pickle=True).tolist()
        return faiss_index, data

    print("Building new FAISS index...")

    _len = len(documents)
    _window = 1000
    n = _len//_window
    for enum in range(n+1):
        embeddings_tmp = embedder.encode(documents[enum*_window:(enum+1)*_window], convert_to_numpy=True)
        if enum==0:
            embeddings = embeddings_tmp
        else:
            embeddings = np.concatenate((embeddings, embeddings_tmp), axis=0)

    # embeddings = embedder.encode(documents, convert_to_numpy=True)
    embeddings = l2_normalize(embeddings)
    dim = embeddings.shape[1]

    # 1. Quantizer
    quantizer = faiss.IndexFlatL2(dim)

    # 2. IVF index
    nlist = 100
    base_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    # 3. Train (MANDATORY)
    base_index.train(embeddings)

    # 4. Wrap with ID map
    faiss_index = faiss.IndexIDMap(base_index)

    # 5. Add vectors with IDs
    faiss_index.add_with_ids(embeddings, ids)

    faiss.write_index(faiss_index, INDEX_DIR)
    np.save(DOCS_DIR, np.array(data, dtype=object))

    return faiss_index, data


def faiss_index_loading(embedder):
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError("FAISS index not found")

    faiss_index = faiss.read_index(INDEX_DIR)
    data = np.load(DOCS_DIR, allow_pickle=True).tolist()
    # faiss.normalize_L2(faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal))
    return faiss_index,data

