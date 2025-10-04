#!/usr/bin/env python3
"""
IMO 2002 Problem 1 Simulation
S is the set of all (h,k) with h,k non-negative integers such that h + k < n.
Each element of S is colored red or blue, with the constraint that if (h,k) is red,
then (h',k') is also red for all h' <= h, k' <= k (lower-left monotonicity).

Type 1 subset: n blue elements with different first member
Type 2 subset: n blue elements with different second member

Goal: Show that there are the same number of type 1 and type 2 subsets.
"""

import itertools
from typing import List, Tuple, Set
import json


def generate_all_points(n: int) -> Set[Tuple[int, int]]:
    """Generate all points (h,k) where h,k >= 0 and h + k < n."""
    return {(h, k) for h in range(n) for k in range(n - h)}


def is_valid_coloring(blue_points: Set[Tuple[int, int]], all_points: Set[Tuple[int, int]]) -> bool:
    """
    Check if a coloring is valid (monotonicity constraint).
    If (h,k) is red, then all (h',k') with h' <= h, k' <= k must be red.
    Equivalently: if (h',k') is blue, then all (h'',k'') with h'' >= h', k'' >= k' must be blue.
    """
    for h, k in blue_points:
        # Check all points that should be red if (h,k) is blue
        for h_prime in range(h + 1):
            for k_prime in range(k + 1):
                if (h_prime, k_prime) in all_points:
                    # If (h,k) is blue and (h',k') <= (h,k),
                    # we cannot have (h',k') red while (h,k) is blue
                    # Actually: if (h,k) is red, all smaller must be red
                    # So if (h',k') is blue and (h',k') <= (h,k), (h,k) must be blue too
                    pass

    # Check the constraint: if (h,k) is red, all (h',k') with h'<=h, k'<=k are red
    red_points = all_points - blue_points
    for h, k in red_points:
        for h_prime in range(h + 1):
            for k_prime in range(k + 1):
                if (h_prime, k_prime) in all_points and (h_prime, k_prime) in blue_points:
                    return False
    return True


def get_type1_subsets(blue_points: Set[Tuple[int, int]], n: int) -> List[Set[Tuple[int, int]]]:
    """
    Get all type 1 subsets: n blue elements with different first member.
    """
    # Group by first coordinate
    by_first = {}
    for h, k in blue_points:
        if h not in by_first:
            by_first[h] = []
        by_first[h].append((h, k))

    # We need exactly n different first coordinates
    if len(by_first) < n:
        return []

    # Choose n different first coordinates
    type1_subsets = []
    for first_coords in itertools.combinations(sorted(by_first.keys()), n):
        # For each chosen first coordinate, choose one point
        point_choices = [by_first[h] for h in first_coords]
        for combo in itertools.product(*point_choices):
            type1_subsets.append(set(combo))

    return type1_subsets


def get_type2_subsets(blue_points: Set[Tuple[int, int]], n: int) -> List[Set[Tuple[int, int]]]:
    """
    Get all type 2 subsets: n blue elements with different second member.
    """
    # Group by second coordinate
    by_second = {}
    for h, k in blue_points:
        if k not in by_second:
            by_second[k] = []
        by_second[k].append((h, k))

    # We need exactly n different second coordinates
    if len(by_second) < n:
        return []

    # Choose n different second coordinates
    type2_subsets = []
    for second_coords in itertools.combinations(sorted(by_second.keys()), n):
        # For each chosen second coordinate, choose one point
        point_choices = [by_second[k] for k in second_coords]
        for combo in itertools.product(*point_choices):
            type2_subsets.append(set(combo))

    return type2_subsets


def enumerate_valid_colorings(n: int) -> List[Set[Tuple[int, int]]]:
    """
    Enumerate all valid colorings (blue point sets) for a given n.
    """
    all_points = generate_all_points(n)
    valid_colorings = []

    # Try all possible subsets as blue points
    for r in range(len(all_points) + 1):
        for blue_combo in itertools.combinations(all_points, r):
            blue_set = set(blue_combo)
            if is_valid_coloring(blue_set, all_points):
                valid_colorings.append(blue_set)

    return valid_colorings


def simulate(n: int) -> dict:
    """
    Simulate the problem for a given n.
    Returns statistics about type 1 and type 2 subsets.
    """
    all_points = generate_all_points(n)
    valid_colorings = enumerate_valid_colorings(n)

    results = []
    counterexamples = []

    for coloring in valid_colorings:
        type1 = get_type1_subsets(coloring, n)
        type2 = get_type2_subsets(coloring, n)

        result = {
            'blue_points': sorted(list(coloring)),
            'num_type1': len(type1),
            'num_type2': len(type2),
            'equal': len(type1) == len(type2)
        }
        results.append(result)

        if len(type1) != len(type2):
            counterexamples.append(result)

    return {
        'n': n,
        'total_points': len(all_points),
        'valid_colorings': len(valid_colorings),
        'results': results,
        'counterexamples': counterexamples,
        'conjecture_holds': len(counterexamples) == 0
    }


def print_simulation_results(sim_results: dict):
    """Print simulation results in a readable format."""
    n = sim_results['n']
    print(f"\n{'='*60}")
    print(f"SIMULATION FOR n = {n}")
    print(f"{'='*60}")
    print(f"Total points in S: {sim_results['total_points']}")
    print(f"Valid colorings: {sim_results['valid_colorings']}")
    print(f"Conjecture holds: {sim_results['conjecture_holds']}")

    if sim_results['counterexamples']:
        print(f"\nCOUNTEREXAMPLES FOUND: {len(sim_results['counterexamples'])}")
        for i, ce in enumerate(sim_results['counterexamples'][:5]):  # Show first 5
            print(f"\nCounterexample {i+1}:")
            print(f"  Blue points: {ce['blue_points']}")
            print(f"  Type 1 subsets: {ce['num_type1']}")
            print(f"  Type 2 subsets: {ce['num_type2']}")
    else:
        print("\nNo counterexamples found!")
        print("\nSample colorings:")
        for i, result in enumerate(sim_results['results'][:5]):  # Show first 5
            print(f"\nColoring {i+1}:")
            print(f"  Blue points: {result['blue_points']}")
            print(f"  Type 1 subsets: {result['num_type1']}")
            print(f"  Type 2 subsets: {result['num_type2']}")


def main():
    """Run simulations for small values of n."""
    print("IMO 2002 Problem 1 - Simulation")
    print("="*60)

    all_results = {}

    for n in range(1, 6):  # Test n = 1, 2, 3, 4, 5
        sim_results = simulate(n)
        all_results[n] = sim_results
        print_simulation_results(sim_results)

    # Save results to JSON
    json_results = {
        str(k): {
            'n': v['n'],
            'total_points': v['total_points'],
            'valid_colorings': v['valid_colorings'],
            'conjecture_holds': v['conjecture_holds'],
            'num_counterexamples': len(v['counterexamples']),
            'counterexamples': v['counterexamples'][:10],  # Save first 10
            'sample_results': v['results'][:10]  # Save first 10
        }
        for k, v in all_results.items()
    }

    with open('simulation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print("\n" + "="*60)
    print("Results saved to simulation_results.json")
    print("="*60)


if __name__ == "__main__":
    main()
