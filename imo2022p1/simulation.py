"""
IMO 2022 Problem 1 Simulation

This simulation implements the coin-ordering process described in IMO 2022 Problem 1.

Problem Summary:
- Marianne has n aluminium (A) and n bronze (B) coins in a row
- Given a fixed position k (1 <= k <= 2n), she repeatedly:
  1. Identifies the longest chain containing the k-th coin
  2. Moves that entire chain to the left end
- Question: For which pairs (n, k) will the leftmost n coins always become
  all the same type, regardless of initial ordering?
"""

import json
import itertools
from typing import List, Tuple, Dict, Set
from collections import defaultdict


class CoinGame:
    """Simulates the coin-ordering process."""

    def __init__(self, coins: str, k: int):
        """
        Initialize the game.

        Args:
            coins: String of 'A' and 'B' representing the coin sequence
            k: The position (1-indexed) to track
        """
        self.initial_coins = coins
        self.coins = coins
        self.k = k
        self.n = len(coins) // 2
        self.history = [coins]
        self.max_iterations = 1000

    def get_chain_at_position(self, pos: int) -> Tuple[int, int]:
        """
        Find the longest chain containing the coin at position pos (0-indexed).

        Returns:
            Tuple of (start_index, end_index) of the chain
        """
        coin_type = self.coins[pos]
        start = pos
        end = pos

        # Extend left
        while start > 0 and self.coins[start - 1] == coin_type:
            start -= 1

        # Extend right
        while end < len(self.coins) - 1 and self.coins[end + 1] == coin_type:
            end += 1

        return start, end

    def perform_operation(self) -> bool:
        """
        Perform one operation: move the chain containing the k-th coin to the left.

        Returns:
            True if state changed, False if already in fixed point
        """
        k_index = self.k - 1  # Convert to 0-indexed
        start, end = self.get_chain_at_position(k_index)

        # If chain is already at the left, we've reached a fixed point
        if start == 0:
            return False

        # Extract the chain and move it to the left
        chain = self.coins[start:end + 1]
        remaining = self.coins[:start] + self.coins[end + 1:]
        self.coins = chain + remaining

        return True

    def run(self) -> Dict:
        """
        Run the simulation until it reaches a fixed point or max iterations.

        Returns:
            Dictionary with results
        """
        iteration = 0

        while iteration < self.max_iterations:
            changed = self.perform_operation()

            if not changed:
                # Reached fixed point
                break

            self.history.append(self.coins)
            iteration += 1

            # Check for cycles (shouldn't happen but just in case)
            if len(self.history) > len(set(self.history)):
                break

        # Check if leftmost n coins are all the same type
        leftmost_n = self.coins[:self.n]
        all_same = len(set(leftmost_n)) == 1

        return {
            'initial': self.initial_coins,
            'final': self.coins,
            'iterations': iteration,
            'history': self.history,
            'leftmost_n_same': all_same,
            'leftmost_n': leftmost_n,
            'n': self.n,
            'k': self.k
        }


def generate_all_orderings(n: int) -> List[str]:
    """
    Generate all possible initial orderings with n A's and n B's.

    Args:
        n: Number of coins of each type

    Returns:
        List of all possible orderings
    """
    # Generate all combinations of positions for A's
    all_positions = range(2 * n)
    orderings = []

    for a_positions in itertools.combinations(all_positions, n):
        coins = ['B'] * (2 * n)
        for pos in a_positions:
            coins[pos] = 'A'
        orderings.append(''.join(coins))

    return orderings


def test_pair(n: int, k: int) -> Dict:
    """
    Test a specific (n, k) pair with all possible initial orderings.

    Args:
        n: Number of coins of each type
        k: Position to track (1-indexed)

    Returns:
        Dictionary with test results
    """
    orderings = generate_all_orderings(n)
    results = []
    success_count = 0

    for ordering in orderings:
        game = CoinGame(ordering, k)
        result = game.run()
        results.append(result)

        if result['leftmost_n_same']:
            success_count += 1

    all_succeed = (success_count == len(orderings))

    return {
        'n': n,
        'k': k,
        'total_orderings': len(orderings),
        'success_count': success_count,
        'all_succeed': all_succeed,
        'success_rate': success_count / len(orderings),
        'individual_results': results
    }


def find_patterns(max_n: int = 6) -> Dict:
    """
    Test all pairs (n, k) for n up to max_n and identify patterns.

    Args:
        max_n: Maximum value of n to test

    Returns:
        Dictionary with pattern analysis
    """
    valid_pairs = []
    invalid_pairs = []
    all_results = {}

    for n in range(1, max_n + 1):
        for k in range(1, 2 * n + 1):
            print(f"Testing (n={n}, k={k})...")
            result = test_pair(n, k)
            all_results[f"({n},{k})"] = result

            if result['all_succeed']:
                valid_pairs.append((n, k))
                print(f"  ✓ Valid: all orderings succeed")
            else:
                invalid_pairs.append((n, k))
                print(f"  ✗ Invalid: {result['success_count']}/{result['total_orderings']} succeed")

    # Analyze patterns
    patterns = analyze_patterns(valid_pairs, invalid_pairs, max_n)

    return {
        'valid_pairs': valid_pairs,
        'invalid_pairs': invalid_pairs,
        'patterns': patterns,
        'detailed_results': all_results
    }


def analyze_patterns(valid_pairs: List[Tuple[int, int]],
                     invalid_pairs: List[Tuple[int, int]],
                     max_n: int) -> Dict:
    """
    Analyze the valid and invalid pairs to find patterns.

    Args:
        valid_pairs: List of (n, k) pairs that work for all orderings
        invalid_pairs: List of (n, k) pairs that don't work for all orderings
        max_n: Maximum n tested

    Returns:
        Dictionary with pattern analysis
    """
    patterns = {
        'observations': [],
        'by_n': defaultdict(list),
        'formulas_tested': {}
    }

    # Group valid pairs by n
    for n, k in valid_pairs:
        patterns['by_n'][n].append(k)

    # Look for patterns
    observations = []

    # Check if k = n+1 always works
    k_equals_n_plus_1 = all((n, n + 1) in valid_pairs for n in range(1, max_n + 1))
    patterns['formulas_tested']['k = n + 1'] = k_equals_n_plus_1
    if k_equals_n_plus_1:
        observations.append("k = n + 1 works for all tested n")

    # Check if k = 1 always works
    k_equals_1 = all((n, 1) in valid_pairs for n in range(1, max_n + 1))
    patterns['formulas_tested']['k = 1'] = k_equals_1
    if k_equals_1:
        observations.append("k = 1 works for all tested n")

    # Check if k = 2n always works
    k_equals_2n = all((n, 2 * n) in valid_pairs for n in range(1, max_n + 1))
    patterns['formulas_tested']['k = 2n'] = k_equals_2n
    if k_equals_2n:
        observations.append("k = 2n works for all tested n")

    # Check for symmetric patterns
    for n in range(1, max_n + 1):
        valid_k = patterns['by_n'][n]
        symmetric = all(k in valid_k if (2 * n + 1 - k) in valid_k else True
                       for k in range(1, 2 * n + 1))
        if symmetric and len(valid_k) > 0:
            observations.append(f"For n={n}, valid k values appear symmetric around k={n+0.5}")

    # Identify the range of valid k for each n
    for n in range(1, max_n + 1):
        valid_k = sorted(patterns['by_n'][n])
        if valid_k:
            observations.append(f"For n={n}, valid k: {valid_k}")

    patterns['observations'] = observations

    return patterns


def create_example_traces(n: int = 4, k: int = 4, num_examples: int = 5) -> List[Dict]:
    """
    Create detailed traces for a few example orderings.

    Args:
        n: Number of coins of each type
        k: Position to track
        num_examples: Number of examples to generate

    Returns:
        List of detailed trace results
    """
    orderings = generate_all_orderings(n)
    traces = []

    # Get a diverse set of examples
    step = max(1, len(orderings) // num_examples)
    example_orderings = orderings[::step][:num_examples]

    for ordering in example_orderings:
        game = CoinGame(ordering, k)
        result = game.run()
        traces.append({
            'initial': ordering,
            'final': result['final'],
            'steps': result['history'],
            'iterations': result['iterations'],
            'leftmost_n_same': result['leftmost_n_same']
        })

    return traces


def main():
    """Main function to run the simulation and generate results."""
    print("=" * 60)
    print("IMO 2022 Problem 1 - Coin Game Simulation")
    print("=" * 60)
    print()

    # Generate example traces
    print("Generating example traces for n=4, k=4...")
    example_traces = create_example_traces(n=4, k=4, num_examples=5)

    print(f"Generated {len(example_traces)} example traces")
    print()

    # Find patterns for small values of n
    print("Testing all pairs (n, k) for n = 1 to 6...")
    print()
    pattern_results = find_patterns(max_n=6)

    print()
    print("=" * 60)
    print("PATTERN ANALYSIS")
    print("=" * 60)

    print(f"\nValid pairs (work for ALL initial orderings):")
    for n, k in pattern_results['valid_pairs']:
        print(f"  (n={n}, k={k})")

    print(f"\nTotal valid pairs: {len(pattern_results['valid_pairs'])}")
    print(f"Total invalid pairs: {len(pattern_results['invalid_pairs'])}")

    print("\nPattern observations:")
    for obs in pattern_results['patterns']['observations']:
        print(f"  - {obs}")

    # Save results to JSON
    output = {
        'example_traces': example_traces,
        'pattern_results': {
            'valid_pairs': pattern_results['valid_pairs'],
            'invalid_pairs': pattern_results['invalid_pairs'],
            'patterns': {
                'observations': pattern_results['patterns']['observations'],
                'by_n': dict(pattern_results['patterns']['by_n']),
                'formulas_tested': pattern_results['patterns']['formulas_tested']
            }
        },
        'detailed_results': pattern_results['detailed_results']
    }

    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("Results saved to results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
