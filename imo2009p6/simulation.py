"""
IMO 2009 Problem 6 - Grasshopper Jump Simulation

This simulation explores the grasshopper jumping problem:
- Given n distinct positive integers a_1, a_2, ..., a_n
- Given a set M of n-1 positive integers not containing s = sum(a_i)
- Find an ordering of jumps such that the grasshopper never lands on any point in M

The simulation:
1. Generates various test cases
2. Uses a greedy/backtracking algorithm to find valid orderings
3. Analyzes patterns in successful orderings
4. Generates ground truth data for proof construction
"""

import json
import itertools
from typing import List, Set, Tuple, Optional, Dict
import random


def compute_partial_sums(ordering: List[int]) -> List[int]:
    """
    Compute the sequence of positions the grasshopper lands on.

    Args:
        ordering: List of jump lengths in order

    Returns:
        List of cumulative positions
    """
    positions = []
    current = 0
    for jump in ordering:
        current += jump
        positions.append(current)
    return positions


def is_valid_ordering(jumps: List[int], forbidden: Set[int]) -> bool:
    """
    Check if an ordering avoids all forbidden positions.

    Args:
        jumps: Ordered list of jump lengths
        forbidden: Set of forbidden positions

    Returns:
        True if no partial sum lands on a forbidden position
    """
    partial_sums = compute_partial_sums(jumps)
    # Exclude the final position (which is the total sum)
    for i in range(len(partial_sums) - 1):
        if partial_sums[i] in forbidden:
            return False
    return True


def find_valid_ordering_bruteforce(jumps: List[int], forbidden: Set[int]) -> Optional[List[int]]:
    """
    Find a valid ordering by trying all permutations (brute force).
    Only practical for small n.

    Args:
        jumps: List of jump lengths
        forbidden: Set of forbidden positions

    Returns:
        A valid ordering if one exists, None otherwise
    """
    for perm in itertools.permutations(jumps):
        if is_valid_ordering(list(perm), forbidden):
            return list(perm)
    return None


def find_valid_ordering_backtrack(jumps: List[int], forbidden: Set[int]) -> Optional[List[int]]:
    """
    Find a valid ordering using backtracking.
    More efficient than brute force.

    Args:
        jumps: List of jump lengths
        forbidden: Set of forbidden positions

    Returns:
        A valid ordering if one exists, None otherwise
    """
    def backtrack(remaining: List[int], current_sum: int, path: List[int]) -> Optional[List[int]]:
        if not remaining:
            return path

        for i, jump in enumerate(remaining):
            next_sum = current_sum + jump
            # Check if this jump would land on a forbidden position
            # (but allow the final position which equals the total sum)
            if len(remaining) > 1 and next_sum in forbidden:
                continue

            new_remaining = remaining[:i] + remaining[i+1:]
            result = backtrack(new_remaining, next_sum, path + [jump])
            if result is not None:
                return result

        return None

    return backtrack(jumps, 0, [])


def greedy_ordering(jumps: List[int], forbidden: Set[int]) -> Optional[List[int]]:
    """
    Try a greedy approach: at each step, choose a jump that doesn't land on forbidden.
    This doesn't guarantee finding a solution even if one exists.

    Args:
        jumps: List of jump lengths
        forbidden: Set of forbidden positions

    Returns:
        An ordering (may not be valid)
    """
    remaining = jumps.copy()
    current_sum = 0
    ordering = []

    while remaining:
        # Find all valid next jumps
        valid_jumps = []
        for jump in remaining:
            next_sum = current_sum + jump
            if len(remaining) > 1 and next_sum in forbidden:
                continue
            valid_jumps.append(jump)

        if not valid_jumps:
            # Greedy failed, try any remaining jump
            if remaining:
                jump = remaining[0]
            else:
                break
        else:
            # Choose randomly from valid jumps (or deterministically)
            jump = valid_jumps[0]

        ordering.append(jump)
        remaining.remove(jump)
        current_sum += jump

    return ordering if is_valid_ordering(ordering, forbidden) else None


def generate_test_case(n: int, seed: Optional[int] = None) -> Tuple[List[int], Set[int], int]:
    """
    Generate a random test case.

    Args:
        n: Number of jumps
        seed: Random seed for reproducibility

    Returns:
        Tuple of (jumps, forbidden_set, total_sum)
    """
    if seed is not None:
        random.seed(seed)

    # Generate n distinct positive integers
    # Use a reasonable range to avoid huge numbers
    max_val = n * 10
    jumps = random.sample(range(1, max_val), n)

    total_sum = sum(jumps)

    # Generate n-1 forbidden positions (not including total_sum)
    possible_positions = list(range(1, total_sum))
    forbidden = set(random.sample(possible_positions, min(n-1, len(possible_positions))))

    return jumps, forbidden, total_sum


def analyze_ordering(jumps: List[int], ordering: List[int], forbidden: Set[int]) -> Dict:
    """
    Analyze a valid ordering to extract patterns.

    Args:
        jumps: Original jump lengths
        ordering: A valid ordering
        forbidden: Forbidden positions

    Returns:
        Dictionary with analysis results
    """
    positions = compute_partial_sums(ordering)
    total = positions[-1]

    # Compute distances to forbidden positions
    min_distances = []
    for pos in positions[:-1]:  # Exclude final position
        if forbidden:
            min_dist = min(abs(pos - f) for f in forbidden)
            min_distances.append(min_dist)

    # Find which positions were "close calls"
    close_calls = [pos for pos in positions[:-1] if any(abs(pos - f) <= 2 for f in forbidden)]

    return {
        "ordering": ordering,
        "positions": positions,
        "total_sum": total,
        "num_close_calls": len(close_calls),
        "close_call_positions": close_calls,
        "min_distances": min_distances,
        "avg_min_distance": sum(min_distances) / len(min_distances) if min_distances else 0
    }


def run_simulation(max_n: int = 7, num_random_per_n: int = 10) -> Dict:
    """
    Run comprehensive simulation across various test cases.

    Args:
        max_n: Maximum number of jumps to test
        num_random_per_n: Number of random test cases per value of n

    Returns:
        Dictionary containing all results
    """
    results = {
        "test_cases": [],
        "statistics": {
            "total_cases": 0,
            "solved_cases": 0,
            "unsolved_cases": 0,
            "by_n": {}
        },
        "patterns": [],
        "examples": []
    }

    for n in range(2, max_n + 1):
        n_stats = {
            "total": 0,
            "solved": 0,
            "unsolved": 0,
            "avg_solve_time": 0
        }

        # Test random cases
        for seed in range(num_random_per_n):
            jumps, forbidden, total_sum = generate_test_case(n, seed)

            # Try to find a valid ordering
            if n <= 6:
                ordering = find_valid_ordering_backtrack(jumps, forbidden)
            else:
                ordering = find_valid_ordering_backtrack(jumps, forbidden)

            test_case = {
                "n": n,
                "jumps": jumps,
                "forbidden": list(forbidden),
                "total_sum": total_sum,
                "solution_found": ordering is not None,
                "solution": ordering if ordering else None
            }

            if ordering:
                analysis = analyze_ordering(jumps, ordering, forbidden)
                test_case["analysis"] = analysis
                n_stats["solved"] += 1
            else:
                n_stats["unsolved"] += 1

            n_stats["total"] += 1
            results["test_cases"].append(test_case)

        results["statistics"]["by_n"][n] = n_stats
        results["statistics"]["total_cases"] += n_stats["total"]
        results["statistics"]["solved_cases"] += n_stats["solved"]
        results["statistics"]["unsolved_cases"] += n_stats["unsolved"]

    # Extract interesting examples
    for case in results["test_cases"]:
        if case["solution_found"] and case["n"] >= 4:
            results["examples"].append({
                "n": case["n"],
                "jumps": case["jumps"],
                "forbidden": case["forbidden"],
                "ordering": case["solution"],
                "positions": case["analysis"]["positions"]
            })
            if len(results["examples"]) >= 10:
                break

    # Identify patterns
    results["patterns"].append({
        "observation": "Success rate by n",
        "data": {n: results["statistics"]["by_n"][n]["solved"] / results["statistics"]["by_n"][n]["total"]
                 for n in results["statistics"]["by_n"]}
    })

    return results


def create_specific_examples() -> List[Dict]:
    """
    Create specific hand-crafted examples that illustrate key concepts.

    Returns:
        List of example test cases
    """
    examples = []

    # Example 1: Simple case
    jumps = [1, 2, 3]
    forbidden = {2, 4}  # n-1 = 2 elements
    total = 6  # not in M
    ordering = find_valid_ordering_backtrack(jumps, forbidden)
    examples.append({
        "name": "Simple example (n=3)",
        "jumps": jumps,
        "forbidden": list(forbidden),
        "total_sum": total,
        "ordering": ordering,
        "positions": compute_partial_sums(ordering) if ordering else None,
        "explanation": "With jumps [1,2,3] and forbidden {2,4}, we need to avoid positions 2 and 4."
    })

    # Example 2: Slightly larger
    jumps = [1, 2, 3, 4]
    total = 10
    forbidden = {2, 4, 7}  # n-1 = 3 elements
    ordering = find_valid_ordering_backtrack(jumps, forbidden)
    examples.append({
        "name": "Four jumps",
        "jumps": jumps,
        "forbidden": list(forbidden),
        "total_sum": total,
        "ordering": ordering,
        "positions": compute_partial_sums(ordering) if ordering else None,
        "explanation": "With jumps [1,2,3,4] and forbidden {2,4,7}, find a valid ordering."
    })

    # Example 3: Consecutive integers
    jumps = [1, 2, 3, 4, 5]
    total = 15
    # Make a challenging M
    forbidden = {3, 6, 9, 12}  # n-1 = 4 elements
    ordering = find_valid_ordering_backtrack(jumps, forbidden)
    examples.append({
        "name": "Consecutive jumps",
        "jumps": jumps,
        "forbidden": list(forbidden),
        "total_sum": total,
        "ordering": ordering,
        "positions": compute_partial_sums(ordering) if ordering else None,
        "explanation": "Consecutive integers as jumps with multiples of 3 forbidden."
    })

    return examples


def main():
    """
    Main function to run the simulation and save results.
    """
    print("=" * 60)
    print("IMO 2009 Problem 6 - Grasshopper Jump Simulation")
    print("=" * 60)
    print()

    # Run main simulation
    print("Running simulation...")
    results = run_simulation(max_n=7, num_random_per_n=10)

    print(f"Total test cases: {results['statistics']['total_cases']}")
    print(f"Solved: {results['statistics']['solved_cases']}")
    print(f"Unsolved: {results['statistics']['unsolved_cases']}")
    print()

    # Create specific examples
    print("Creating specific examples...")
    specific_examples = create_specific_examples()
    results["specific_examples"] = specific_examples

    for ex in specific_examples:
        print(f"\n{ex['name']}:")
        print(f"  Jumps: {ex['jumps']}")
        print(f"  Forbidden: {ex['forbidden']}")
        print(f"  Solution: {ex['ordering']}")
        if ex['ordering']:
            print(f"  Positions: {ex['positions']}")

    # Key insights for proof
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR PROOF:")
    print("=" * 60)

    insights = [
        "1. The problem always has a solution (the theorem to prove).",
        "2. There are at most n-1 forbidden positions but n! possible orderings.",
        "3. Each ordering creates n-1 intermediate positions (excluding start and final).",
        "4. The constraint |M| = n-1 is crucial - there are 'enough' orderings.",
        "5. A greedy/constructive approach often works: choose jumps to avoid forbidden positions.",
        "6. The final position (sum of all jumps) is guaranteed not to be in M.",
        "7. Different orderings create different sets of intermediate positions."
    ]

    for insight in insights:
        print(insight)

    results["proof_insights"] = insights

    # Save results
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
