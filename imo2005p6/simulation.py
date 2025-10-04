"""
IMO 2005 Problem 6 Simulation

This simulation explores the constraints of a mathematical competition where:
- 6 problems are posed
- Every pair of problems is solved by more than 2/5 of contestants
- No contestant solved all 6 problems

Goal: Show that at least 2 contestants solved exactly 5 problems.
"""

import json
import itertools
from typing import List, Set, Tuple, Dict
from collections import Counter


def count_contestants_solving_pair(contestants: List[Set[int]], p1: int, p2: int) -> int:
    """Count how many contestants solved both problems p1 and p2."""
    return sum(1 for solved in contestants if p1 in solved and p2 in solved)


def validates_pair_constraint(contestants: List[Set[int]], n_contestants: int) -> bool:
    """Check if every pair of problems is solved by more than 2/5 of contestants."""
    threshold = 2 * n_contestants / 5
    for p1, p2 in itertools.combinations(range(6), 2):
        count = count_contestants_solving_pair(contestants, p1, p2)
        if count <= threshold:
            return False
    return True


def validates_all_constraints(contestants: List[Set[int]]) -> bool:
    """Check if a configuration satisfies all problem constraints."""
    n_contestants = len(contestants)

    # No contestant solved all 6 problems
    if any(len(solved) == 6 for solved in contestants):
        return False

    # Every pair solved by more than 2/5
    return validates_pair_constraint(contestants, n_contestants)


def count_by_problems_solved(contestants: List[Set[int]]) -> Counter:
    """Count how many contestants solved exactly k problems for each k."""
    return Counter(len(solved) for solved in contestants)


def find_minimal_counterexample(max_contestants: int = 20) -> Dict:
    """
    Try to find a minimal counterexample where fewer than 2 contestants solved exactly 5 problems.
    This explores small configurations to understand the problem.
    """
    results = []

    for n in range(1, max_contestants + 1):
        print(f"Checking n={n} contestants...")

        # Try different distributions of problems solved
        # Each contestant solves 0-5 problems (not 6 by constraint)
        for distribution in generate_distributions(n):
            contestants = generate_contestants_from_distribution(distribution)

            if validates_all_constraints(contestants):
                count_dist = count_by_problems_solved(contestants)
                results.append({
                    'n_contestants': n,
                    'distribution': dict(count_dist),
                    'contestants_with_5': count_dist[5],
                    'satisfies_constraint': count_dist[5] >= 2,
                    'contestants': [sorted(list(s)) for s in contestants]
                })

                # If we found a counterexample (fewer than 2 with exactly 5)
                if count_dist[5] < 2:
                    return {
                        'counterexample_found': True,
                        'n_contestants': n,
                        'distribution': dict(count_dist),
                        'contestants': [sorted(list(s)) for s in contestants]
                    }

    return {
        'counterexample_found': False,
        'results': results[:10]  # Return first 10 valid configurations
    }


def generate_distributions(n: int, max_depth: int = 3) -> List[List[int]]:
    """Generate possible distributions of how many problems each contestant solved."""
    # For small n, generate all combinations
    if n <= max_depth:
        for combo in itertools.product(range(6), repeat=n):
            yield list(combo)
    else:
        # For larger n, sample strategically
        # Focus on distributions with many 4s and 5s
        for _ in range(100):  # Sample some distributions
            dist = [4] * (n // 2) + [5] * (n - n // 2)
            yield dist


def generate_contestants_from_distribution(distribution: List[int]) -> List[Set[int]]:
    """
    Given a distribution of how many problems each contestant solved,
    generate a specific assignment.
    """
    contestants = []
    for num_solved in distribution:
        # Assign a random subset of problems
        problems = set(itertools.combinations(range(6), num_solved)).__iter__().__next__()
        contestants.append(set(problems))
    return contestants


def analyze_minimum_contestants() -> Dict:
    """
    Analyze: What is the minimum number of contestants needed for the constraints to be satisfiable?

    For every pair of problems to be solved by more than 2n/5 contestants,
    we need at least ceil(2n/5) + 1 contestants solving each pair.
    """
    results = {}

    for n in range(1, 30):
        threshold = 2 * n / 5
        min_per_pair = int(threshold) + 1

        results[n] = {
            'threshold': threshold,
            'min_per_pair': min_per_pair,
            'feasible': None  # To be determined
        }

    return results


def systematic_search(n_contestants: int) -> Dict:
    """
    Systematic search for valid configurations with n contestants.
    Uses a greedy/strategic approach for larger n.
    """
    threshold = 2 * n_contestants / 5
    min_per_pair = int(threshold) + 1

    print(f"\nSearching for valid configurations with {n_contestants} contestants")
    print(f"Threshold: {threshold}, min per pair: {min_per_pair}")

    # Strategy: Try configurations where most contestants solve 5 problems
    # and vary which problem they skip

    valid_configs = []

    # Try: Each contestant solves exactly 5 problems
    # Contestant i skips problem (i mod 6)
    contestants = []
    for i in range(n_contestants):
        skipped = i % 6
        solved = set(range(6)) - {skipped}
        contestants.append(solved)

    if validates_all_constraints(contestants):
        count_dist = count_by_problems_solved(contestants)
        valid_configs.append({
            'strategy': 'each_skips_different',
            'distribution': dict(count_dist),
            'contestants_with_5': count_dist[5],
            'contestants': [sorted(list(s)) for s in contestants]
        })

    # Try: Mix of contestants with 4 and 5 problems
    if n_contestants >= 6:
        contestants = []
        # First n-6 solve problems {0,1,2,3,4}
        for i in range(max(0, n_contestants - 6)):
            contestants.append({0, 1, 2, 3, 4})
        # Last 6 each skip a different problem
        for i in range(min(6, n_contestants)):
            skipped = i
            solved = set(range(6)) - {skipped}
            contestants.append(solved)

        if validates_all_constraints(contestants):
            count_dist = count_by_problems_solved(contestants)
            valid_configs.append({
                'strategy': 'mixed_4_and_5',
                'distribution': dict(count_dist),
                'contestants_with_5': count_dist[5],
                'contestants': [sorted(list(s)) for s in contestants]
            })

    return {
        'n_contestants': n_contestants,
        'valid_configs': valid_configs
    }


def compute_pair_statistics(contestants: List[Set[int]], n_contestants: int) -> Dict:
    """Compute detailed statistics about each pair of problems."""
    pair_stats = {}
    threshold = 2 * n_contestants / 5

    for p1, p2 in itertools.combinations(range(6), 2):
        count = count_contestants_solving_pair(contestants, p1, p2)
        pair_stats[f"{p1},{p2}"] = {
            'count': count,
            'threshold': threshold,
            'satisfies': count > threshold,
            'ratio': count / n_contestants if n_contestants > 0 else 0
        }

    return pair_stats


def main():
    """Main simulation to generate ground truth data."""
    results = {
        'problem_statement': 'IMO 2005 Problem 6',
        'constraints': {
            'num_problems': 6,
            'pair_threshold': '> 2/5 of contestants',
            'max_solved_per_contestant': 5
        },
        'analysis': {}
    }

    print("=" * 70)
    print("IMO 2005 Problem 6 Simulation")
    print("=" * 70)

    # Test specific small cases
    test_cases = []

    # Test case 1: n=6, each contestant skips a different problem
    print("\nTest Case 1: n=6, each contestant skips a different problem")
    contestants = [
        {1, 2, 3, 4, 5},  # Skip 0
        {0, 2, 3, 4, 5},  # Skip 1
        {0, 1, 3, 4, 5},  # Skip 2
        {0, 1, 2, 4, 5},  # Skip 3
        {0, 1, 2, 3, 5},  # Skip 4
        {0, 1, 2, 3, 4},  # Skip 5
    ]
    n = len(contestants)
    valid = validates_all_constraints(contestants)
    count_dist = count_by_problems_solved(contestants)
    pair_stats = compute_pair_statistics(contestants, n)

    test_cases.append({
        'name': 'n=6, each skips different problem',
        'n_contestants': n,
        'valid': valid,
        'distribution': dict(count_dist),
        'contestants_with_5': count_dist[5],
        'pair_statistics': pair_stats,
        'contestants': [sorted(list(s)) for s in contestants]
    })

    print(f"Valid: {valid}")
    print(f"Distribution: {count_dist}")
    print(f"Contestants with exactly 5: {count_dist[5]}")

    # Test case 2: n=9, exploring if we can get away with just 1 contestant with 5
    print("\nTest Case 2: n=9, trying to minimize contestants with 5")
    # Try: 1 contestant with 5, others with 4
    contestants = [
        {0, 1, 2, 3, 4},  # 5 problems
    ]
    for i in range(8):
        contestants.append({0, 1, 2, 3})  # 4 problems

    n = len(contestants)
    valid = validates_all_constraints(contestants)
    count_dist = count_by_problems_solved(contestants)
    pair_stats = compute_pair_statistics(contestants, n)

    test_cases.append({
        'name': 'n=9, trying 1 with 5 and rest with 4',
        'n_contestants': n,
        'valid': valid,
        'distribution': dict(count_dist),
        'contestants_with_5': count_dist[5],
        'pair_statistics': pair_stats,
        'contestants': [sorted(list(s)) for s in contestants]
    })

    print(f"Valid: {valid}")
    print(f"Distribution: {count_dist}")
    print(f"Contestants with exactly 5: {count_dist[5]}")

    # Test case 3: n=10, balanced approach
    print("\nTest Case 3: n=10, each contestant skips different problem (cycling)")
    contestants = []
    for i in range(10):
        skipped = i % 6
        solved = set(range(6)) - {skipped}
        contestants.append(solved)

    n = len(contestants)
    valid = validates_all_constraints(contestants)
    count_dist = count_by_problems_solved(contestants)
    pair_stats = compute_pair_statistics(contestants, n)

    test_cases.append({
        'name': 'n=10, cycling which problem to skip',
        'n_contestants': n,
        'valid': valid,
        'distribution': dict(count_dist),
        'contestants_with_5': count_dist[5],
        'pair_statistics': pair_stats,
        'contestants': [sorted(list(s)) for s in contestants]
    })

    print(f"Valid: {valid}")
    print(f"Distribution: {count_dist}")
    print(f"Contestants with exactly 5: {count_dist[5]}")

    # Test case 4: Theoretical minimum n
    print("\nTest Case 4: Finding minimum n for feasibility")
    min_n_results = []
    for n in range(1, 15):
        threshold = 2 * n / 5
        min_per_pair = int(threshold) + 1
        # Check if n=6 pattern works
        contestants = []
        for i in range(n):
            skipped = i % 6
            solved = set(range(6)) - {skipped}
            contestants.append(solved)

        valid = validates_all_constraints(contestants)
        count_dist = count_by_problems_solved(contestants)

        min_n_results.append({
            'n': n,
            'threshold': threshold,
            'min_per_pair': min_per_pair,
            'valid': valid,
            'contestants_with_5': count_dist[5]
        })

        if valid:
            print(f"n={n}: Valid! Threshold={threshold:.2f}, contestants with 5={count_dist[5]}")

    results['analysis']['test_cases'] = test_cases
    results['analysis']['minimum_n_exploration'] = min_n_results

    # Generate summary insights
    print("\n" + "=" * 70)
    print("Summary Insights")
    print("=" * 70)

    insights = []

    # Find minimum n where constraint can be satisfied
    min_valid_n = next((r['n'] for r in min_n_results if r['valid']), None)
    if min_valid_n:
        insights.append(f"Minimum n for valid configuration: {min_valid_n}")

    # Count how many valid configs have >= 2 contestants with 5
    valid_with_enough_5 = sum(1 for tc in test_cases if tc['valid'] and tc['contestants_with_5'] >= 2)
    insights.append(f"Valid test cases with >= 2 contestants solving 5: {valid_with_enough_5}/{len([tc for tc in test_cases if tc['valid']])}")

    # Check if we ever found a valid counterexample
    counterexample = any(tc['valid'] and tc['contestants_with_5'] < 2 for tc in test_cases)
    insights.append(f"Found valid counterexample (< 2 with 5 problems): {counterexample}")

    results['analysis']['insights'] = insights

    for insight in insights:
        print(f"- {insight}")

    # Save results
    print("\nSaving results to results.json...")
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Done!")

    return results


if __name__ == '__main__':
    main()
