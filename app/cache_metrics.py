import redis

# --- Configuration ---
# In a real app, Redis connection details should be managed via configuration.
REDIS_HOST = "redis"
REDIS_PORT = 6379

# --- Module-level Redis Connection ---
# It's generally better to manage connections explicitly, but for this
# self-contained metrics module, a global connection is acceptable.
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    r.ping()
except redis.exceptions.ConnectionError as e:
    print(f"Warning: Could not connect to Redis for cache metrics. Metrics will not be recorded. Error: {e}")
    r = None

# --- Metric Keys ---
# Using a clear, consistent naming convention for Redis keys is crucial.
TOTAL_REQUESTS = "cache:total_requests"
HITS = "cache:hits"
MISSES = "cache:misses"
# The 'evictions' key is a placeholder; true eviction tracking is more complex
# and often requires configuration on the Redis side (e.g., keyspace notifications).
EVICTIONS = "cache:evictions"


# --- Atomic Operations ---
def incr_total_requests(pipe=None):
    """Increments the total number of search requests handled."""
    if r:
        (pipe or r).incr(TOTAL_REQUESTS)

def incr_hits(pipe=None):
    """Increments the cache hit counter."""
    if r:
        # Use a pipeline to ensure both counters are incremented in one transaction.
        p = pipe or r.pipeline()
        p.incr(HITS)
        p.incr(TOTAL_REQUESTS) # A hit is also a request.
        if not pipe:
            p.execute()

def incr_misses(pipe=None):
    """Increments the cache miss counter."""
    if r:
        p = pipe or r.pipeline()
        p.incr(MISSES)
        p.incr(TOTAL_REQUESTS) # A miss is also a request.
        if not pipe:
            p.execute()

def get_snapshot():
    """
    Retrieves a snapshot of the current cache metrics.
    Returns a dictionary of the metric values.
    """
    if not r:
        return {k: 0 for k in [TOTAL_REQUESTS, HITS, MISSES, EVICTIONS]}

    keys = [TOTAL_REQUESTS, HITS, MISSES, EVICTIONS]
    values = r.mget(keys)
    # mget returns None for non-existent keys, so we default to 0.
    return {keys[i]: int(values[i] or 0) for i in range(len(keys))}