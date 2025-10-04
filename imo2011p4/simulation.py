#!/usr/bin/env python3
"""
IMO 2011 Problem 4 Simulation
Given n weights of weight 2^0, 2^1, ..., 2^(n-1), place them on a balance
such that the right pan is never heavier than the left pan.
Count the number of valid ways to do this.
"""

from typing import List, Tuple, Set
from collections import defaultdict
import json

def simulate_placements(n: int, verbose: bool = False) -> Tuple[int, List[List[Tuple[int, str]]]]:
    """
    Simulate all possible ways to place n weights on a balance.
    Returns the count of valid placements and the list of valid sequences.

    Args:
        n: Number of weights (weights are 2^0, 2^1, ..., 2^(n-1))
        verbose: If True, print debug information

    Returns:
        Tuple of (count, list of valid placement sequences)
        Each sequence is a list of (weight_index, 'L' or 'R') tuples
    """
    weights = [2**i for i in range(n)]

    valid_sequences = []

    def backtrack(remaining_indices: Set[int], left_weight: int, right_weight: int,
                  sequence: List[Tuple[int, str]]):
        """
        Backtracking to explore all valid placements.

        Args:
            remaining_indices: Set of weight indices not yet placed
            left_weight: Current weight on left pan
            right_weight: Current weight on right pan
            sequence: Current sequence of placements
        """
        # Base case: all weights placed
        if not remaining_indices:
            valid_sequences.append(sequence[:])
            return

        # Try placing each remaining weight
        for idx in remaining_indices:
            weight = weights[idx]

            # Try placing on left pan
            new_left = left_weight + weight
            new_right = right_weight
            if new_right <= new_left:  # Constraint: right never heavier than left
                new_remaining = remaining_indices - {idx}
                new_sequence = sequence + [(idx, 'L')]
                backtrack(new_remaining, new_left, new_right, new_sequence)

            # Try placing on right pan
            new_left = left_weight
            new_right = right_weight + weight
            if new_right <= new_left:  # Constraint: right never heavier than left
                new_remaining = remaining_indices - {idx}
                new_sequence = sequence + [(idx, 'R')]
                backtrack(new_remaining, new_left, new_right, new_sequence)

    # Start backtracking
    backtrack(set(range(n)), 0, 0, [])

    if verbose:
        print(f"\nFor n={n}:")
        print(f"Total valid placements: {len(valid_sequences)}")
        if n <= 3:
            for i, seq in enumerate(valid_sequences, 1):
                print(f"\n  Sequence {i}:")
                left, right = 0, 0
                for idx, side in seq:
                    w = weights[idx]
                    if side == 'L':
                        left += w
                    else:
                        right += w
                    print(f"    Place weight 2^{idx} (={w}) on {side} pan -> L:{left}, R:{right}")

    return len(valid_sequences), valid_sequences


def analyze_patterns(max_n: int = 8) -> dict:
    """
    Analyze patterns for different values of n.

    Args:
        max_n: Maximum value of n to analyze

    Returns:
        Dictionary with analysis results
    """
    results = {}

    print("=" * 60)
    print("IMO 2011 Problem 4 - Simulation Results")
    print("=" * 60)

    for n in range(1, max_n + 1):
        count, sequences = simulate_placements(n, verbose=(n <= 3))
        results[n] = {
            'count': count,
            'sequences': sequences
        }
        print(f"\nn = {n}: {count} valid placements")

    # Look for pattern
    print("\n" + "=" * 60)
    print("Pattern Analysis:")
    print("=" * 60)
    counts = [results[n]['count'] for n in range(1, max_n + 1)]
    print(f"Counts: {counts}")

    # Check if it's a known sequence
    print("\nRatios (count[n] / count[n-1]):")
    for n in range(2, max_n + 1):
        ratio = counts[n-1] / counts[n-2]
        print(f"  n={n}: {ratio:.4f}")

    # Check for factorial pattern
    print("\nComparison with factorials:")
    import math
    for n in range(1, max_n + 1):
        fact = math.factorial(n)
        print(f"  n={n}: count={counts[n-1]}, n!={fact}, count/n!={counts[n-1]/fact:.4f}")

    # Check for powers of 2
    print("\nComparison with powers of 2:")
    for n in range(1, max_n + 1):
        power = 2**(n-1)
        print(f"  n={n}: count={counts[n-1]}, 2^(n-1)={power}, ratio={counts[n-1]/power:.4f}")

    # Check for Catalan numbers
    print("\nComparison with Catalan numbers:")
    def catalan(n):
        if n <= 1:
            return 1
        c = [0] * (n + 1)
        c[0], c[1] = 1, 1
        for i in range(2, n + 1):
            for j in range(i):
                c[i] += c[j] * c[i - 1 - j]
        return c[n]

    for n in range(1, max_n + 1):
        cat = catalan(n)
        print(f"  n={n}: count={counts[n-1]}, C_{n}={cat}, ratio={counts[n-1]/cat:.4f}")

    return results


def export_ground_truth(results: dict, filename: str = "ground_truth_summary.json"):
    """
    Export compressed summary of ground truth data to JSON file.
    Stores counts and up to 3 example sequences per n to keep file size small.

    Args:
        results: Results dictionary from analyze_patterns
        filename: Output filename (default: ground_truth_summary.json)
    """
    # Convert to summary format with limited examples
    export_data = {}
    for n, data in results.items():
        sequences = data['sequences']
        export_data[str(n)] = {
            'count': data['count'],
            'n': n,
            'example_sequences': [
                [(idx, side) for idx, side in seq]
                for seq in sequences[:min(3, len(sequences))]  # Keep only 3 examples
            ]
        }

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)

    file_size = len(json.dumps(export_data)) / 1024
    print(f"\n\nGround truth summary exported to {filename} ({file_size:.1f} KB)")
    print(f"Stored counts for n=1 to {max(results.keys())} with up to 3 example sequences each")


if __name__ == "__main__":
    # Run simulation for small instances
    results = analyze_patterns(max_n=8)

    # Export ground truth data
    export_ground_truth(results)

    # Summary
    print("\n" + "=" * 60)
    print("CONJECTURE:")
    print("=" * 60)
    print("Based on the pattern analysis, the number of valid placements")
    print("for n weights appears to be related to factorials or Catalan numbers.")
    print("Further mathematical analysis needed to prove the exact formula.")
