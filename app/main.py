import json
import faiss
import numpy as np
import redis
import torch
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

# --- Configuration & Globals ---
# In a real app, use a config file (e.g., Dynaconf, Pydantic Settings)
REDIS_HOST = "redis"
REDIS_PORT = 6379
INDEX_PATH = "models/index.faiss"  # Path to the pre-built FAISS index
MODEL_PATH = "models/model.pth"    # Path to the trained encoder model
EMBEDDING_DIM = 768  # Example dimension, should match your model

# --- Initialization ---
app = FastAPI(
    title="Unified AI Platform Orchestrator",
    description="API for high-performance semantic search and retrieval.",
    version="1.0.0",
)

# Connect to Redis
try:
    cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_connect_timeout=2)
    cache.ping()
    print("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    cache = None

# Load FAISS index
try:
    index = faiss.read_index(INDEX_PATH)
    print(f"Successfully loaded FAISS index from {INDEX_PATH}.")
except Exception as e:
    print(f"Could not load FAISS index: {e}. Search endpoint will be disabled.")
    index = None

# Load the encoder model (mocked for now)
# In a real scenario, you'd load your PyTorch/TensorFlow/etc. model here.
# For this example, we'll use a mock function.
class MockEncoder:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, text_input):
        # Create a deterministic "embedding" based on the input hash
        # This is NOT a real embedding, just for placeholder purposes.
        np.random.seed(hash(text_input) & 0xFFFFFFFF)
        return np.random.rand(1, self.dim).astype('float32')

model = MockEncoder(dim=EMBEDDING_DIM)
print("Mock encoder model is ready.")


# --- Prometheus Metrics ---
# This exposes a /metrics endpoint
Instrumentator().instrument(app).expose(app)


# --- API Endpoints ---
@app.get('/health', summary="Health Check", tags=["System"])
def health_check():
    """
    Performs a health check on the API and its dependencies.
    """
    return {
        "status": "ok",
        "redis_connected": cache is not None and cache.ping(),
        "faiss_index_loaded": index is not None,
    }

@app.get('/search', summary="Perform Semantic Search", tags=["Search"])
def search(q: str, k: int = 10):
    """
    Performs a semantic search for a given query `q` and returns the top `k` results.

    - **q**: The search query string.
    - **k**: The number of results to return.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty.")
    if not index:
        raise HTTPException(status_code=503, detail="FAISS index is not available.")
    if not cache:
        print("Warning: Redis cache is not available. Proceeding without caching.")

    cache_key = f"search:q:{q}:k:{k}"

    # 1. Check cache first
    if cache:
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                print(f"Cache hit for query: '{q}'")
                return json.loads(cached_result)
        except redis.exceptions.RedisError as e:
            print(f"Redis error while getting cache: {e}")


    print(f"Cache miss for query: '{q}'")

    # 2. Compute embedding for the query
    query_embedding = model(q)
    faiss.normalize_L2(query_embedding)

    # 3. Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # 4. Format results (in a real app, you'd fetch metadata for the indices)
    results = {
        "query": q,
        "results": [
            {"id": int(i), "distance": float(d)}
            for i, d in zip(indices[0], distances[0])
        ],
    }

    # 5. Store in cache
    if cache:
        try:
            # Cache with a TTL (e.g., 1 hour)
            cache.setex(cache_key, 3600, json.dumps(results))
        except redis.exceptions.RedisError as e:
            print(f"Redis error while setting cache: {e}")


    return results