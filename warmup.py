import requests
import time
import json

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT = 10 # seconds

# A list of common or important queries to pre-warm the cache with.
# In a real-world scenario, this list would be populated from analytics,
# logs of frequent searches, or a curated list of high-value queries.
PREWARM_QUERIES = [
    "what is the capital of France",
    "latest advancements in AI",
    "how to train a dual-encoder model",
    "python fastapi tutorial",
    "benefits of momentum encoders",
    "what is InfoNCE loss",
    "deploying machine learning models with kubernetes",
    "using redis for caching",
    "introduction to FAISS",
    "canary deployment strategy",
]

def check_api_health():
    """Checks if the API is ready to accept requests."""
    print(f"Checking API health at {HEALTH_ENDPOINT}...")
    for i in range(10): # Try for 100 seconds
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                print("API is healthy and ready.")
                return True
        except requests.ConnectionError:
            pass
        print(f"API not ready, retrying in 10 seconds... (Attempt {i+1}/10)")
        time.sleep(10)
    print("Error: API did not become healthy in time.")
    return False

def run_warmup():
    """Sends the predefined queries to the search endpoint."""
    print("\n--- Starting Cache Warmup ---")
    if not check_api_health():
        return

    print(f"Found {len(PREWARM_QUERIES)} queries to pre-warm.")
    success_count = 0
    failure_count = 0

    for i, query in enumerate(PREWARM_QUERIES):
        params = {"q": query, "k": 10}
        try:
            response = requests.get(SEARCH_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                print(f"({i+1}/{len(PREWARM_QUERIES)}) SUCCESS: Sent query '{query}'")
                success_count += 1
            else:
                print(f"({i+1}/{len(PREWARM_QUERIES)}) FAILED: Query '{query}' returned status {response.status_code}")
                failure_count += 1
        except requests.exceptions.RequestException as e:
            print(f"({i+1}/{len(PREWARM_QUERIES)}) ERROR: Request for query '{query}' failed: {e}")
            failure_count += 1

        time.sleep(0.2) # Small delay to avoid overwhelming the server

    print("\n--- Warmup Complete ---")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {failure_count}")

if __name__ == "__main__":
    run_warmup()