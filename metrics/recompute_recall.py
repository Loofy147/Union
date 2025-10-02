from typing import List, Set, Iterable
import json
import numpy as np

def recall_at_k(retrieved_indices: Iterable[Iterable[int]], ground_truth_sets: Iterable[Set[int]], k: int) -> float:
    """
    Compute recall@k where retrieved_indices and ground_truth_sets are aligned.
    Ensures monotonicity across k when used on same data.
    """
    retrieved_indices = list(retrieved_indices)
    ground_truth_sets = list(ground_truth_sets)
    assert len(retrieved_indices) == len(ground_truth_sets), "Mismatch queries / gts"
    hits = 0
    total = len(retrieved_indices)
    per_query = []
    for retrieved, gt in zip(retrieved_indices, ground_truth_sets):
        topk = retrieved[:k]
        hit = any(r in gt for r in topk)
        per_query.append(int(hit))
        hits += int(hit)
    # The original implementation was missing the return of per_query, adding it back
    return (hits / total if total > 0 else 0.0), per_query

if __name__ == "__main__":
    # example usage: load JSON dumps from logs: each entry {"query_id":..., "retrieved":[...], "gt":[...]}
    import sys
    # Create a dummy file for demonstration if no path is provided
    if len(sys.argv) < 2:
        print("Usage: python metrics/recompute_recall.py <path_to_jsonl>")
        print("Creating a dummy 'retrieval_results.jsonl' for demonstration.")
        with open("retrieval_results.jsonl", "w") as f:
            f.write('{"query_id": 1, "retrieved": [101, 102, 103, 104, 105], "gt": [103]}\n')
            f.write('{"query_id": 2, "retrieved": [201, 202, 203, 204, 205], "gt": [209]}\n')
            f.write('{"query_id": 3, "retrieved": [301, 302, 303, 304, 305], "gt": [301, 302]}\n')
        path = "retrieval_results.jsonl"
    else:
        path = sys.argv[1]  # e.g. logs/retrieval_results.jsonl

    records = []
    try:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                records.append(rec)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {path}")
        sys.exit(1)

    retrieved = [r['retrieved'] for r in records]
    gts = [set(r.get('gt', [])) for r in records]

    print(f"--- Calculating Recall for {len(records)} queries ---")
    for k in (1,5,10,20,50):
        val, perq = recall_at_k(retrieved, gts, k)
        print(f"recall@{k} = {val:.6f}")