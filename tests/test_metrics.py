import sys
import os
import pytest

# Add the project root to the Python path to allow importing from the 'metrics' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.recompute_recall import recall_at_k

def test_recall_monotonicity():
    """
    Ensures that recall@k is monotonically increasing as k increases.
    """
    # A sample list of retrieved document IDs for multiple queries
    retrieved = [
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Query 1
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],            # Query 2
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]     # Query 3
    ]
    # The corresponding ground truth sets of relevant document IDs
    gts = [
        {60},   # Hit for Query 1 is at position 6
        {99},   # Query 2 is always a miss
        {12}    # Hit for Query 3 is at position 2
    ]

    # Calculate recall at various k values
    r1, _ = recall_at_k(retrieved, gts, 1)
    r2, _ = recall_at_k(retrieved, gts, 2)
    r5, _ = recall_at_k(retrieved, gts, 5)
    r6, _ = recall_at_k(retrieved, gts, 6)
    r10, _ = recall_at_k(retrieved, gts, 10)

    # Assert that recall never decreases as k grows
    assert r1 <= r2, "Recall should not decrease from k=1 to k=2"
    assert r2 <= r5, "Recall should not decrease from k=2 to k=5"
    assert r5 <= r6, "Recall should not decrease from k=5 to k=6"
    assert r6 <= r10, "Recall should not decrease from k=6 to k=10"

def test_recall_edge_cases():
    """
    Tests recall calculation with edge cases like empty inputs or no hits.
    """
    # Test with no queries
    r_empty, _ = recall_at_k([], [], k=10)
    assert r_empty == 0.0, "Recall should be 0 for empty input"

    # Test with queries but no ground truth
    retrieved = [[1, 2, 3]]
    gts = [{}]
    r_no_gt, _ = recall_at_k(retrieved, gts, k=3)
    assert r_no_gt == 0.0, "Recall should be 0 if there are no ground truth labels"

    # Test with no retrieved items for a query
    retrieved = [[]]
    gts = [{1}]
    r_no_retrieved, _ = recall_at_k(retrieved, gts, k=1)
    assert r_no_retrieved == 0.0, "Recall should be 0 if no items were retrieved"

def test_recall_correct_values():
    """
    Tests that recall@k calculates the correct percentage.
    """
    retrieved = [
        [1, 2, 3],  # Hit
        [4, 5, 6],  # Miss
        [7, 8, 9],  # Hit
        [10, 11, 12] # Miss
    ]
    gts = [
        {3},
        {99},
        {7},
        {98}
    ]

    # With k=1, only the 3rd query is a hit.
    r1, _ = recall_at_k(retrieved, gts, k=1)
    assert r1 == pytest.approx(1/4), "Recall@1 should be 0.25"

    # With k=3, the 1st and 3rd queries are hits.
    r3, _ = recall_at_k(retrieved, gts, k=3)
    assert r3 == pytest.approx(2/4), "Recall@3 should be 0.5"

    # With k=10 (larger than result size), result should be same as k=3
    r10, _ = recall_at_k(retrieved, gts, k=10)
    assert r10 == pytest.approx(2/4), "Recall@10 should be 0.5"