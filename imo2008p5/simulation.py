"""
IMO 2008 Problem 5 Simulation

This simulation explores the ratio N/M where:
- N = number of k-step sequences resulting in lamps 1..n ON and lamps (n+1)..2n OFF
- M = number of k-step sequences with same result but lamps (n+1)..2n never turned ON

Key insights:
- k >= n and k-n must be even
- We start with all lamps OFF
- Each step switches one lamp (ON->OFF or OFF->ON)

DISCOVERED PATTERN:
- N/M = 2^(k-n) for all valid (n, k) pairs
- Equivalently: N/M = 4^((k-n)/2)
- The answer is independent of n, depending only on the difference k-n!

Implementation:
- Uses dynamic programming to count sequences
- State space: lamp configurations at each step
- For N: counts all valid sequences
- For M: counts only sequences that never turn on lamps (n+1)..2n
"""

import json
from collections import defaultdict
from typing import List, Tuple, Set
import itertools


def lamp_state_to_tuple(state: List[bool]) -> Tuple[bool, ...]:
    """Convert lamp state list to hashable tuple."""
    return tuple(state)


def count_sequences_N(n: int, k: int) -> int:
    """
    Count sequences of k steps resulting in lamps 1..n ON and lamps (n+1)..2n OFF.
    Uses dynamic programming with memoization.

    State: (step, lamp_state)
    """
    total_lamps = 2 * n
    target_state = [True] * n + [False] * n

    # dp[step][state] = number of ways to reach state after 'step' steps
    dp = [defaultdict(int) for _ in range(k + 1)]

    # Initial state: all lamps OFF
    initial_state = lamp_state_to_tuple([False] * total_lamps)
    dp[0][initial_state] = 1

    # Fill DP table
    for step in range(k):
        for state_tuple, count in dp[step].items():
            state = list(state_tuple)
            # Try switching each lamp
            for lamp_idx in range(total_lamps):
                new_state = state[:]
                new_state[lamp_idx] = not new_state[lamp_idx]
                new_state_tuple = lamp_state_to_tuple(new_state)
                dp[step + 1][new_state_tuple] += count

    # Return count for target state
    target_tuple = lamp_state_to_tuple(target_state)
    return dp[k][target_tuple]


def count_sequences_M(n: int, k: int) -> int:
    """
    Count sequences of k steps resulting in lamps 1..n ON and lamps (n+1)..2n OFF,
    with the constraint that lamps (n+1)..2n are NEVER turned ON.

    Uses dynamic programming with memoization.
    """
    total_lamps = 2 * n
    target_state = [True] * n + [False] * n

    # dp[step][state] = number of ways to reach state after 'step' steps
    # Only track states where lamps (n+1)..2n are all OFF
    dp = [defaultdict(int) for _ in range(k + 1)]

    # Initial state: all lamps OFF
    initial_state = lamp_state_to_tuple([False] * total_lamps)
    dp[0][initial_state] = 1

    # Fill DP table
    for step in range(k):
        for state_tuple, count in dp[step].items():
            state = list(state_tuple)
            # Try switching each lamp
            for lamp_idx in range(total_lamps):
                new_state = state[:]
                new_state[lamp_idx] = not new_state[lamp_idx]

                # Check constraint: lamps (n+1)..2n must never be ON
                if any(new_state[n:]):
                    continue  # Skip this transition

                new_state_tuple = lamp_state_to_tuple(new_state)
                dp[step + 1][new_state_tuple] += count

    # Return count for target state
    target_tuple = lamp_state_to_tuple(target_state)
    return dp[k][target_tuple]


def compute_ratio(n: int, k: int) -> dict:
    """Compute N, M, and N/M for given n and k."""
    if k < n or (k - n) % 2 != 0:
        return {
            'n': n,
            'k': k,
            'valid': False,
            'error': 'Invalid parameters: k >= n and (k-n) must be even'
        }

    N = count_sequences_N(n, k)
    M = count_sequences_M(n, k)

    result = {
        'n': n,
        'k': k,
        'valid': True,
        'N': N,
        'M': M,
        'ratio': N / M if M > 0 else None,
        'ratio_simplified': f"{N}/{M}" if M > 0 else "undefined"
    }

    return result


def find_pattern():
    """
    Run simulation for small values to find pattern.
    Tests various (n, k) pairs where k >= n and k-n is even.
    """
    results = []

    # Test cases: small values of n and k
    test_cases = [
        (1, 1), (1, 3), (1, 5),
        (2, 2), (2, 4), (2, 6),
        (3, 3), (3, 5), (3, 7),
        (4, 4), (4, 6),
        (5, 5),
    ]

    for n, k in test_cases:
        print(f"Computing n={n}, k={k}...")
        result = compute_ratio(n, k)
        results.append(result)

        if result['valid']:
            print(f"  N={result['N']}, M={result['M']}, N/M={result['ratio']}")
        else:
            print(f"  {result['error']}")

    return results


def analyze_pattern(results: List[dict]) -> dict:
    """Analyze results to find patterns."""
    analysis = {
        'observations': [],
        'ratio_values': [],
        'conjectured_formula': None
    }

    # Extract valid results
    valid_results = [r for r in results if r['valid'] and r['M'] > 0]

    if not valid_results:
        return analysis

    # Check if ratio is constant or follows a pattern
    ratios = [r['ratio'] for r in valid_results]
    analysis['ratio_values'] = [
        {'n': r['n'], 'k': r['k'], 'ratio': r['ratio'], 'k_minus_n': r['k'] - r['n']}
        for r in valid_results
    ]

    # Check if all ratios are 2^(k-n)
    formula_holds = True
    for r in valid_results:
        k_minus_n = r['k'] - r['n']
        expected_ratio = 2 ** k_minus_n
        if abs(r['ratio'] - expected_ratio) < 0.0001:
            analysis['observations'].append(
                f"n={r['n']}, k={r['k']}: N/M = {r['ratio']} = 2^{k_minus_n} = 2^(k-n) ✓"
            )
        else:
            analysis['observations'].append(
                f"n={r['n']}, k={r['k']}: N/M = {r['ratio']} ≠ 2^{k_minus_n} = {expected_ratio} ✗"
            )
            formula_holds = False

    # Check if formula N/M = 2^(k-n) holds for all cases
    if formula_holds:
        analysis['conjectured_formula'] = "N/M = 2^(k-n)"
        analysis['observations'].append("\n" + "="*60)
        analysis['observations'].append("PATTERN FOUND: N/M = 2^(k-n) for all tested cases!")
        analysis['observations'].append("Equivalently: N/M = 4^((k-n)/2)")
        analysis['observations'].append("="*60)

    return analysis


def generate_sequence_examples(n: int, k: int, max_examples: int = 5) -> List[List[int]]:
    """
    Generate example sequences for visualization.
    Returns list of sequences (each sequence is a list of lamp indices that were switched).
    """
    if k < n or (k - n) % 2 != 0:
        return []

    total_lamps = 2 * n
    target_state = [True] * n + [False] * n
    examples_N = []
    examples_M = []

    # For small k, enumerate all possible sequences
    if k <= 8:
        # Generate all possible sequences of k lamp switches
        for sequence in itertools.product(range(total_lamps), repeat=k):
            state = [False] * total_lamps
            violates_M_constraint = False

            for lamp_idx in sequence:
                state[lamp_idx] = not state[lamp_idx]
                # Check if we violate M constraint
                if any(state[n:]):
                    violates_M_constraint = True

            # Check if we reach target state
            if state == target_state:
                seq_list = [i + 1 for i in sequence]  # Convert to 1-indexed
                examples_N.append(seq_list)
                if not violates_M_constraint:
                    examples_M.append(seq_list)

                if len(examples_N) >= max_examples and len(examples_M) >= max_examples:
                    break

    return {
        'examples_N': examples_N[:max_examples],
        'examples_M': examples_M[:max_examples]
    }


def main():
    """Main simulation function."""
    print("=" * 60)
    print("IMO 2008 Problem 5 Simulation")
    print("=" * 60)
    print()

    # Run pattern finding
    print("Finding patterns in N/M ratio...")
    print()
    results = find_pattern()

    print()
    print("=" * 60)
    print("Analyzing pattern...")
    print("=" * 60)
    analysis = analyze_pattern(results)

    for obs in analysis['observations']:
        print(obs)

    if analysis['conjectured_formula']:
        print()
        print(f"CONJECTURED FORMULA: {analysis['conjectured_formula']}")

    # Generate examples for visualization
    print()
    print("=" * 60)
    print("Generating example sequences for visualization...")
    print("=" * 60)

    example_cases = [(2, 2), (2, 4), (3, 3)]
    sequence_examples = {}

    for n, k in example_cases:
        print(f"Generating examples for n={n}, k={k}...")
        examples = generate_sequence_examples(n, k, max_examples=10)
        sequence_examples[f"n{n}_k{k}"] = examples

    # Save results to JSON
    output = {
        'results': results,
        'analysis': analysis,
        'sequence_examples': sequence_examples,
        'summary': {
            'conjectured_formula': analysis['conjectured_formula'],
            'tested_cases': len([r for r in results if r['valid']]),
            'formula_holds': all(
                abs(r['ratio'] - 2 ** (r['k'] - r['n'])) < 0.0001
                for r in results if r['valid'] and r['M'] > 0
            )
        }
    }

    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 60)
    print("Results saved to results.json")
    print("=" * 60)

    return output


if __name__ == "__main__":
    main()
