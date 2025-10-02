import faiss
import numpy as np
import os

# --- Configuration ---
EMBEDDING_DIM = 768       # Dimension of the embeddings, must match the model and API
NUM_EMBEDDINGS = 100000   # Number of dummy embeddings to generate for the index
INDEX_TYPE = "HNSW"       # Type of index to build ("HNSW", "FLAT", "IVFPQ")
OUTPUT_DIR = "models"
INDEX_FILENAME = "index.faiss"

# HNSW Parameters (only used if INDEX_TYPE is "HNSW")
HNSW_M = 32              # Number of neighbors for each node
HNSW_EF_CONSTRUCTION = 200 # efConstruction parameter for HNSW
HNSW_EF_SEARCH = 128     # efSearch parameter for HNSW

def build_hnsw_index(embeddings: np.ndarray, d: int, m: int, ef_c: int, ef_s: int):
    """Builds and tunes an HNSW flat index."""
    print(f"Building HNSW index with M={m}, efConstruction={ef_c}...")
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_c
    index.hnsw.efSearch = ef_s # Set search-time parameter

    print("Normalizing embeddings (L2)...")
    faiss.normalize_L2(embeddings)

    print(f"Adding {embeddings.shape[0]} embeddings to the index...")
    index.add(embeddings)

    print(f"Index built successfully. Total vectors in index: {index.ntotal}")
    return index

def build_flat_index(embeddings: np.ndarray, d: int):
    """Builds a simple L2 flat index."""
    print("Building FLAT L2 index...")
    index = faiss.IndexFlatL2(d)

    print("Normalizing embeddings (L2)...")
    faiss.normalize_L2(embeddings)

    print(f"Adding {embeddings.shape[0]} embeddings to the index...")
    index.add(embeddings)

    print(f"Index built successfully. Total vectors in index: {index.ntotal}")
    return index


def main():
    """
    Main function to generate data, build the index, and save it.
    """
    print("--- FAISS Index Builder ---")

    # 1. Generate dummy data
    print(f"Generating {NUM_EMBEDDINGS} random embeddings of dimension {EMBEDDING_DIM}...")
    # Using float32 is important as it's the standard for FAISS
    xb = np.random.random((NUM_EMBEDDINGS, EMBEDDING_DIM)).astype('float32')

    # 2. Build the selected index
    if INDEX_TYPE.upper() == "HNSW":
        index = build_hnsw_index(xb, EMBEDDING_DIM, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH)
    elif INDEX_TYPE.upper() == "FLAT":
        index = build_flat_index(xb, EMBEDDING_DIM)
    else:
        raise ValueError(f"Unknown INDEX_TYPE: {INDEX_TYPE}. Supported: HNSW, FLAT.")

    # 3. Create output directory and save the index
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, INDEX_FILENAME)
    print(f"Saving index to {output_path}...")
    faiss.write_index(index, output_path)

    print("\n--- Verification ---")
    print(f"Index file created at: {output_path}")
    print(f"Index type: {type(index)}")
    print(f"Is index trained? {index.is_trained}")
    print(f"Number of vectors: {index.ntotal}")

    # Optional: Test a search
    print("\nPerforming a test search for 5 vectors...")
    xq = np.random.random((5, EMBEDDING_DIM)).astype('float32')
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k=5)
    print("Top 5 neighbors for the first test vector:")
    print("Distances:", D[0])
    print("Indices:", I[0])
    print("\nBuild process complete.")


if __name__ == "__main__":
    main()