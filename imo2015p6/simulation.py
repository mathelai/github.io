#!/usr/bin/env python3
"""
IMO 2015 Problem 6 Simulation

This simulation explores sequences satisfying:
(i) 1 ≤ a_j ≤ 2015 for all j ≥ 1
(ii) k + a_k ≠ ℓ + a_ℓ for all 1 ≤ k < ℓ

The problem asks to prove that there exist b and N such that:
|Σ(j=m+1 to n) (a_j - b)| ≤ 1007² for all n > m ≥ N

Key insights:
- Condition (ii) means all values k + a_k must be distinct
- Since 1 ≤ a_j ≤ 2015, we have k+1 ≤ k + a_k ≤ k + 2015
- For large k, there are 2015 possible values but they must all be distinct
- This creates a "pigeonhole" constraint that forces the sequence to balance
"""

import json
import random
from typing import List, Tuple, Dict
from collections import defaultdict


class SequenceGenerator:
    """Generate valid sequences satisfying the IMO 2015 P6 constraints."""

    def __init__(self, max_value: int = 2015):
        self.max_value = max_value
        self.used_sums = set()  # Track k + a_k values

    def is_valid(self, k: int, value: int) -> bool:
        """Check if adding value at position k is valid."""
        if value < 1 or value > self.max_value:
            return False
        return (k + value) not in self.used_sums

    def add_value(self, k: int, value: int):
        """Add a value to the sequence."""
        self.used_sums.add(k + value)

    def generate_random_sequence(self, length: int, seed: int = None) -> List[int]:
        """Generate a random valid sequence of given length."""
        if seed is not None:
            random.seed(seed)

        sequence = []
        self.used_sums = set()

        for k in range(1, length + 1):
            # Find all valid values for position k
            valid_values = [v for v in range(1, self.max_value + 1)
                          if self.is_valid(k, v)]

            if not valid_values:
                raise ValueError(f"No valid value at position {k}")

            # Choose randomly from valid values
            value = random.choice(valid_values)
            sequence.append(value)
            self.add_value(k, value)

        return sequence

    def generate_greedy_sequence(self, length: int, strategy: str = "middle") -> List[int]:
        """
        Generate sequence using greedy strategies.

        Strategies:
        - "middle": prefer values near the middle (1007-1008)
        - "low": prefer lower values
        - "high": prefer higher values
        - "balanced": try to maintain cumulative sum near 0
        """
        sequence = []
        self.used_sums = set()
        cumsum = 0

        for k in range(1, length + 1):
            valid_values = [v for v in range(1, self.max_value + 1)
                          if self.is_valid(k, v)]

            if not valid_values:
                raise ValueError(f"No valid value at position {k}")

            if strategy == "middle":
                # Prefer values near middle
                target = self.max_value // 2
                value = min(valid_values, key=lambda v: abs(v - target))
            elif strategy == "low":
                value = min(valid_values)
            elif strategy == "high":
                value = max(valid_values)
            elif strategy == "balanced":
                # Try to balance around average value
                avg = self.max_value / 2
                target_value = avg - (cumsum / k if k > 0 else 0)
                target_value = max(1, min(self.max_value, target_value))
                value = min(valid_values, key=lambda v: abs(v - target_value))
            else:
                value = valid_values[0]

            sequence.append(value)
            self.add_value(k, value)
            cumsum += value

        return sequence


def analyze_sequence(sequence: List[int], max_value: int = 2015) -> Dict:
    """
    Analyze a sequence for convergence properties.

    Returns statistics about partial sums and the optimal b value.
    """
    n = len(sequence)

    # Compute all partial sums for different b values
    # The optimal b should minimize the maximum deviation

    # Try b values around the average
    avg = sum(sequence) / len(sequence)
    b_candidates = list(range(max(1, int(avg - 50)), min(max_value, int(avg + 50)) + 1))

    best_b = None
    best_max_deviation = float('inf')
    best_n_threshold = None

    for b in b_candidates:
        # For this b, compute max deviation efficiently
        # The maximum deviation is max(cumsum) - min(cumsum)
        cumsum = [0]
        for a in sequence:
            cumsum.append(cumsum[-1] + a - b)

        # Maximum deviation between any two positions
        max_cumsum = max(cumsum)
        min_cumsum = min(cumsum)
        max_dev = max_cumsum - min_cumsum

        # Check if any deviation exceeds bound
        n_threshold = 0
        for m in range(0, min(n, 100)):  # Only check first 100 positions
            for end in range(m + 1, min(n + 1, m + 101)):  # Check next 100
                deviation = abs(cumsum[end] - cumsum[m])
                if deviation > 1007**2:
                    n_threshold = max(n_threshold, m + 1)

        if max_dev < best_max_deviation:
            best_max_deviation = max_dev
            best_b = b
            best_n_threshold = n_threshold

    # Analyze the sequence values
    value_counts = defaultdict(int)
    for a in sequence:
        value_counts[a] += 1

    # Analyze k + a_k values
    sum_values = [k + sequence[k-1] for k in range(1, len(sequence) + 1)]

    return {
        "length": n,
        "average": avg,
        "min_value": min(sequence),
        "max_value": max(sequence),
        "optimal_b": best_b,
        "max_deviation": best_max_deviation,
        "n_threshold": best_n_threshold,
        "bound_satisfied": best_max_deviation <= 1007**2,
        "unique_values": len(set(sequence)),
        "sum_range": (min(sum_values), max(sum_values)),
        "first_20": sequence[:20],
        "last_20": sequence[-20:] if n > 20 else sequence,
    }


def find_constraints_on_sequence(length: int, max_value: int = 2015) -> Dict:
    """
    Analyze what constraints the distinctness condition imposes.

    For position k, we need k + a_k to be distinct from all previous.
    The range of k + a_k is [k+1, k+max_value].
    """
    results = {
        "max_value": max_value,
        "positions_analyzed": [],
    }

    # For various positions, show the constraints
    positions = [1, 10, 100, 1000, 2000, 3000, 5000, 10000]

    for k in positions[:min(len(positions), length // 100 + 3)]:
        if k > length:
            break

        # Range of possible k + a_k values
        min_sum = k + 1
        max_sum = k + max_value

        # How many previous values could conflict?
        # For position j < k, we have j + a_j in [j+1, j+max_value]
        # This overlaps with [k+1, k+max_value] when j+max_value >= k+1
        # i.e., when j >= k + 1 - max_value

        overlap_start = max(1, k + 1 - max_value)
        num_conflicts = k - overlap_start

        results["positions_analyzed"].append({
            "k": k,
            "range": [min_sum, max_sum],
            "range_size": max_value,
            "potential_conflicts": num_conflicts,
            "conflict_ratio": num_conflicts / max_value if max_value > 0 else 0,
        })

    return results


def run_simulation():
    """Run the full simulation and generate results."""
    print("IMO 2015 Problem 6 Simulation")
    print("=" * 60)

    max_value = 2015
    bound = 1007**2

    results = {
        "problem": {
            "max_value": max_value,
            "theoretical_bound": bound,
            "description": "Sequences with 1 ≤ a_j ≤ 2015 and all k+a_k distinct"
        },
        "theoretical_insights": find_constraints_on_sequence(10000, max_value),
        "sequences": []
    }

    # Test different sequence lengths
    lengths = [100, 300, 500]
    strategies = ["middle", "balanced", "random"]

    gen = SequenceGenerator(max_value)

    for length in lengths:
        print(f"\nTesting sequences of length {length}...")

        for strategy in strategies:
            try:
                if strategy == "random":
                    sequence = gen.generate_random_sequence(length, seed=42)
                    desc = "Random valid sequence"
                else:
                    sequence = gen.generate_greedy_sequence(length, strategy)
                    desc = f"Greedy ({strategy}) sequence"

                analysis = analyze_sequence(sequence, max_value)
                analysis["strategy"] = strategy
                analysis["description"] = desc

                results["sequences"].append(analysis)

                print(f"  {strategy:12s}: b={analysis['optimal_b']}, "
                      f"max_dev={analysis['max_deviation']}, "
                      f"bound_ok={analysis['bound_satisfied']}")

            except Exception as e:
                print(f"  {strategy:12s}: Failed - {e}")

    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)

    print("\n1. Constraint Analysis:")
    for pos_info in results["theoretical_insights"]["positions_analyzed"][:5]:
        k = pos_info["k"]
        ratio = pos_info["conflict_ratio"]
        print(f"   Position {k:5d}: {pos_info['potential_conflicts']:4d} conflicts "
              f"out of {pos_info['range_size']} slots (ratio: {ratio:.2%})")

    print("\n2. As k increases, the ratio approaches 1, meaning nearly all")
    print("   possible values conflict with earlier positions.")
    print("   This forces the sequence to 'fill in gaps' and balance out.")

    print("\n3. Sequence Convergence:")
    for seq in results["sequences"][-3:]:
        print(f"   {seq['description']:30s}: optimal b = {seq['optimal_b']:.1f}")
        print(f"   {'':30s}  max deviation = {seq['max_deviation']:.0f} "
              f"(bound: {bound})")

    # Save results
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_simulation()
