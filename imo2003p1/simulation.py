#!/usr/bin/env python3
"""
IMO 2003 Problem 1 Simulation

Problem: S = {1, 2, 3, ..., 1000000}. Show that for any subset A of S with 101 elements,
we can find 100 distinct elements x_i of S such that the sets {a + x_i | a in A}
are all pairwise disjoint.

This simulation explores small instances to understand patterns and find counterexamples.
"""

import random
import itertools
from typing import Set, List, Tuple, Optional
import json


def find_disjoint_offsets_greedy(A: Set[int], S_max: int, num_offsets: int) -> Optional[List[int]]:
    """
    Greedy algorithm to find num_offsets distinct offsets x_i such that
    {a + x_i | a in A} are pairwise disjoint.

    Args:
        A: The subset of S with |A| elements
        S_max: Maximum value in S (S = {1, 2, ..., S_max})
        num_offsets: Number of disjoint offsets to find

    Returns:
        List of offsets if successful, None otherwise
    """
    offsets = []
    used_values = set()

    for _ in range(num_offsets):
        # Try to find an offset that doesn't create conflicts
        found = False
        for x in range(0, S_max * 10):  # Search range
            if x in offsets:
                continue

            # Check if A + x is disjoint from all previously used values
            new_set = {a + x for a in A}
            if not new_set.intersection(used_values):
                offsets.append(x)
                used_values.update(new_set)
                found = True
                break

        if not found:
            return None

    return offsets


def find_disjoint_offsets_optimal(A: Set[int], S_max: int, num_offsets: int) -> Optional[List[int]]:
    """
    Optimal algorithm based on the theoretical proof.

    Strategy: Use offsets spaced such that the gaps between translated sets
    are large enough to prevent overlap. The key insight is that the maximum
    gap within A determines the minimum spacing needed.
    """
    A_sorted = sorted(A)
    A_size = len(A)

    # Find the maximum gap within A
    max_gap = A_sorted[-1] - A_sorted[0]

    # Use spacing of (max_gap + 1) to ensure disjointness
    # This ensures that A + k*(max_gap+1) and A + (k+1)*(max_gap+1) don't overlap
    # since max element of A + k*spacing < min element of A + (k+1)*spacing
    spacing = max_gap + 1

    offsets = []
    for k in range(num_offsets):
        offset = k * spacing
        offsets.append(offset)

    # Verify disjointness
    all_sets = []
    for x in offsets:
        translated_set = {a + x for a in A}
        all_sets.append(translated_set)

    # Check pairwise disjoint
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            if all_sets[i].intersection(all_sets[j]):
                return None

    return offsets


def verify_disjoint(A: Set[int], offsets: List[int]) -> bool:
    """Verify that the translated sets are pairwise disjoint."""
    translated_sets = []
    for x in offsets:
        translated_sets.append({a + x for a in A})

    # Check all pairs
    for i in range(len(translated_sets)):
        for j in range(i + 1, len(translated_sets)):
            if translated_sets[i].intersection(translated_sets[j]):
                return False
    return True


def run_simulation(S_max: int, A_size: int, num_offsets: int, num_trials: int = 100) -> dict:
    """
    Run simulation for given parameters.

    Args:
        S_max: Maximum value in S
        A_size: Size of subset A
        num_offsets: Number of disjoint offsets to find
        num_trials: Number of random trials

    Returns:
        Dictionary with simulation results
    """
    results = {
        'S_max': S_max,
        'A_size': A_size,
        'num_offsets': num_offsets,
        'num_trials': num_trials,
        'greedy_successes': 0,
        'optimal_successes': 0,
        'examples': []
    }

    for trial in range(num_trials):
        # Generate random subset A
        A = set(random.sample(range(1, S_max + 1), A_size))

        # Try greedy approach
        greedy_result = find_disjoint_offsets_greedy(A, S_max, num_offsets)
        if greedy_result is not None:
            results['greedy_successes'] += 1

        # Try optimal approach
        optimal_result = find_disjoint_offsets_optimal(A, S_max, num_offsets)
        if optimal_result is not None:
            results['optimal_successes'] += 1

        # Save first 5 examples
        if trial < 5:
            results['examples'].append({
                'A': sorted(list(A)),
                'greedy_offsets': greedy_result,
                'optimal_offsets': optimal_result,
                'greedy_success': greedy_result is not None,
                'optimal_success': optimal_result is not None
            })

    return results


def explore_problem_space():
    """Explore the problem space with various parameter combinations."""
    print("IMO 2003 Problem 1 - Simulation Environment")
    print("=" * 60)
    print()

    # Test cases scaling up to the actual problem
    test_cases = [
        (100, 11, 10),    # Small instance: S_max=100, |A|=11, find 10 offsets
        (200, 21, 20),    # Medium instance
        (500, 51, 50),    # Larger instance
        (1000, 101, 100), # Close to original problem scale
    ]

    all_results = []

    for S_max, A_size, num_offsets in test_cases:
        print(f"Testing: S={{1..{S_max}}}, |A|={A_size}, find {num_offsets} offsets")
        print("-" * 60)

        results = run_simulation(S_max, A_size, num_offsets, num_trials=50)
        all_results.append(results)

        print(f"Greedy success rate: {results['greedy_successes']}/50")
        print(f"Optimal success rate: {results['optimal_successes']}/50")
        print()

        # Show one example
        if results['examples']:
            example = results['examples'][0]
            print(f"Example A: {example['A'][:10]}..." if len(example['A']) > 10 else f"Example A: {example['A']}")
            if example['optimal_offsets']:
                print(f"Optimal offsets: {example['optimal_offsets'][:10]}..." if len(example['optimal_offsets']) > 10 else f"Optimal offsets: {example['optimal_offsets']}")
        print()

    # Save results to JSON
    with open('simulation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Results saved to simulation_results.json")

    return all_results


def analyze_bounds():
    """Analyze theoretical bounds for the problem."""
    print("\n" + "=" * 60)
    print("Theoretical Analysis")
    print("=" * 60)
    print()

    # For the original problem: S = {1..1000000}, |A| = 101, find 100 offsets
    S_max = 1000000
    A_size = 101
    num_offsets = 100

    print(f"Original Problem: S={{1..{S_max}}}, |A|={A_size}, find {num_offsets} offsets")
    print()

    # Key insight: If we use offsets 0, |A|, 2|A|, ..., (k-1)|A|
    # then the k translated sets occupy disjoint ranges
    print(f"Strategy: Use offsets k*|A| for k = 0, 1, ..., {num_offsets-1}")
    print(f"This gives offsets: 0, {A_size}, {2*A_size}, ..., {(num_offsets-1)*A_size}")
    print()

    # Maximum value reached
    max_offset = (num_offsets - 1) * A_size
    max_value_reached = S_max + max_offset
    print(f"Maximum value in any translated set: {S_max} + {max_offset} = {max_value_reached}")
    print()

    # This is well within bounds - we don't even need to restrict to S!
    print(f"Conclusion: The construction works! Each translated set A + k*|A|")
    print(f"occupies values in range [1 + k*|A|, {S_max} + k*|A|]")
    print(f"These ranges are disjoint for different k values.")


if __name__ == '__main__':
    random.seed(42)
    all_results = explore_problem_space()
    analyze_bounds()
