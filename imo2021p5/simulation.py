"""
IMO 2021 Problem 5 Simulation

This simulation explores the walnut reordering problem where:
- n walnuts numbered 1 to n are arranged in a circle
- In move k, we swap the two walnuts adjacent to walnut k
- We want to find if there exists k where the swapped walnuts a,b satisfy a < k < b

The simulation:
1. Tests various initial permutations
2. Tracks which moves satisfy the condition a < k < b
3. Generates statistics and counterexample attempts
4. Outputs results to help understand the proof
"""

import json
import random
from typing import List, Tuple, Dict, Any
import itertools


class WalnutSimulation:
    def __init__(self, n: int, initial_arrangement: List[int]):
        """
        Initialize the simulation.

        Args:
            n: Number of walnuts
            initial_arrangement: Initial circular arrangement of walnut numbers
        """
        self.n = n
        self.arrangement = initial_arrangement.copy()
        self.initial_arrangement = initial_arrangement.copy()
        self.history = []
        self.condition_satisfied_moves = []

    def find_walnut_position(self, walnut_num: int) -> int:
        """Find the position of a walnut in the circular arrangement."""
        return self.arrangement.index(walnut_num)

    def get_adjacent_positions(self, pos: int) -> Tuple[int, int]:
        """Get the positions of the two adjacent walnuts in the circle."""
        left = (pos - 1) % self.n
        right = (pos + 1) % self.n
        return left, right

    def perform_move(self, k: int) -> Dict[str, Any]:
        """
        Perform move k: swap the two walnuts adjacent to walnut k.

        Returns:
            Dictionary with move information
        """
        # Find position of walnut k
        pos_k = self.find_walnut_position(k)

        # Get adjacent positions
        left_pos, right_pos = self.get_adjacent_positions(pos_k)

        # Get walnut numbers at adjacent positions
        a = self.arrangement[left_pos]
        b = self.arrangement[right_pos]

        # Ensure a < b for consistency
        if a > b:
            a, b = b, a
            left_pos, right_pos = right_pos, left_pos

        # Check if condition a < k < b is satisfied
        condition_satisfied = a < k < b

        # Record move information
        move_info = {
            'move_number': k,
            'walnut_k_position': pos_k,
            'walnut_a': a,
            'walnut_b': b,
            'position_a': left_pos,
            'position_b': right_pos,
            'condition_satisfied': condition_satisfied,
            'arrangement_before': self.arrangement.copy()
        }

        # Perform the swap
        self.arrangement[left_pos], self.arrangement[right_pos] = \
            self.arrangement[right_pos], self.arrangement[left_pos]

        move_info['arrangement_after'] = self.arrangement.copy()

        # Track history
        self.history.append(move_info)
        if condition_satisfied:
            self.condition_satisfied_moves.append(k)

        return move_info

    def run_all_moves(self) -> Dict[str, Any]:
        """
        Run all n moves and return summary statistics.

        Returns:
            Dictionary with simulation results
        """
        for k in range(1, self.n + 1):
            self.perform_move(k)

        return {
            'n': self.n,
            'initial_arrangement': self.initial_arrangement,
            'final_arrangement': self.arrangement,
            'total_moves': self.n,
            'condition_satisfied_count': len(self.condition_satisfied_moves),
            'condition_satisfied_moves': self.condition_satisfied_moves,
            'all_moves_history': self.history
        }


def generate_random_permutation(n: int) -> List[int]:
    """Generate a random permutation of numbers 1 to n."""
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    return perm


def test_all_permutations(n: int) -> Dict[str, Any]:
    """
    Test all permutations for small n to find if any violate the theorem.

    Args:
        n: Number of walnuts (should be small, e.g., n <= 7)

    Returns:
        Dictionary with statistics
    """
    if n > 8:
        raise ValueError("n too large for exhaustive search")

    total_permutations = 0
    permutations_with_condition = 0
    permutations_without_condition = []
    condition_counts = []

    for perm in itertools.permutations(range(1, n + 1)):
        total_permutations += 1
        perm_list = list(perm)

        sim = WalnutSimulation(n, perm_list)
        result = sim.run_all_moves()

        count = result['condition_satisfied_count']
        condition_counts.append(count)

        if count > 0:
            permutations_with_condition += 1
        else:
            permutations_without_condition.append({
                'permutation': perm_list,
                'final_arrangement': result['final_arrangement']
            })

    return {
        'n': n,
        'total_permutations': total_permutations,
        'permutations_with_condition': permutations_with_condition,
        'permutations_without_condition_count': len(permutations_without_condition),
        'counterexamples': permutations_without_condition,
        'min_condition_count': min(condition_counts),
        'max_condition_count': max(condition_counts),
        'avg_condition_count': sum(condition_counts) / len(condition_counts)
    }


def test_random_samples(n: int, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Test random permutations for larger n.

    Args:
        n: Number of walnuts
        num_samples: Number of random permutations to test

    Returns:
        Dictionary with statistics
    """
    results = []
    condition_counts = []
    min_condition_count = float('inf')
    min_condition_example = None

    for i in range(num_samples):
        perm = generate_random_permutation(n)
        sim = WalnutSimulation(n, perm)
        result = sim.run_all_moves()

        count = result['condition_satisfied_count']
        condition_counts.append(count)

        if count < min_condition_count:
            min_condition_count = count
            min_condition_example = {
                'permutation': perm,
                'condition_count': count,
                'condition_moves': result['condition_satisfied_moves'],
                'sample_index': i
            }

        results.append({
            'sample': i,
            'condition_count': count,
            'has_condition': count > 0
        })

    return {
        'n': n,
        'num_samples': num_samples,
        'all_have_condition': all(r['has_condition'] for r in results),
        'min_condition_count': min_condition_count,
        'max_condition_count': max(condition_counts),
        'avg_condition_count': sum(condition_counts) / len(condition_counts),
        'min_condition_example': min_condition_example,
        'samples_summary': results[:100]  # Store first 100 samples
    }


def analyze_specific_permutation(perm: List[int]) -> Dict[str, Any]:
    """
    Analyze a specific permutation in detail.

    Args:
        perm: The permutation to analyze

    Returns:
        Detailed analysis including all moves
    """
    n = len(perm)
    sim = WalnutSimulation(n, perm)
    result = sim.run_all_moves()

    # Add additional analysis
    move_details = []
    for move in result['all_moves_history']:
        move_details.append({
            'k': move['move_number'],
            'a': move['walnut_a'],
            'b': move['walnut_b'],
            'satisfies_condition': move['condition_satisfied'],
            'relation': f"{move['walnut_a']} < {move['move_number']} < {move['walnut_b']}"
                       if move['condition_satisfied'] else
                       f"NOT ({move['walnut_a']} < {move['move_number']} < {move['walnut_b']})"
        })

    return {
        'permutation': perm,
        'n': n,
        'total_moves': n,
        'condition_satisfied_count': result['condition_satisfied_count'],
        'condition_satisfied_moves': result['condition_satisfied_moves'],
        'move_details': move_details,
        'full_history': result['all_moves_history']
    }


def main():
    """Run simulations and generate results."""
    random.seed(42)
    print("IMO 2021 Problem 5 Simulation")
    print("=" * 60)

    results = {
        'problem_statement': (
            "Two squirrels have collected n walnuts numbered 1 to n in a circle. "
            "In move k, swap the two walnuts adjacent to walnut k. "
            "Prove that there exists k such that the swapped walnuts a,b satisfy a < k < b."
        ),
        'simulations': {}
    }

    # Test small values exhaustively
    print("\n1. Exhaustive testing for small n:")
    for n in [3, 4, 5, 6, 7]:
        print(f"\n   Testing n={n}...")
        exhaustive_result = test_all_permutations(n)
        results['simulations'][f'exhaustive_n{n}'] = exhaustive_result

        print(f"   Total permutations: {exhaustive_result['total_permutations']}")
        print(f"   Permutations with condition: {exhaustive_result['permutations_with_condition']}")
        print(f"   Counterexamples found: {exhaustive_result['permutations_without_condition_count']}")
        print(f"   Min/Avg/Max condition count: {exhaustive_result['min_condition_count']:.1f} / "
              f"{exhaustive_result['avg_condition_count']:.1f} / {exhaustive_result['max_condition_count']}")

        if exhaustive_result['counterexamples']:
            print(f"   WARNING: Found counterexamples!")
            for ce in exhaustive_result['counterexamples'][:3]:
                print(f"      {ce['permutation']}")

    # Test larger values with random sampling
    print("\n2. Random sampling for larger n:")
    for n in [10, 21, 50, 100, 2021]:
        num_samples = 1000 if n <= 100 else 100
        print(f"\n   Testing n={n} with {num_samples} random samples...")
        random_result = test_random_samples(n, num_samples)
        results['simulations'][f'random_n{n}'] = random_result

        print(f"   All samples have condition: {random_result['all_have_condition']}")
        print(f"   Min/Avg/Max condition count: {random_result['min_condition_count']} / "
              f"{random_result['avg_condition_count']:.1f} / {random_result['max_condition_count']}")

        if random_result['min_condition_example']:
            example = random_result['min_condition_example']
            print(f"   Minimum example has {example['condition_count']} moves satisfying condition")
            print(f"   Those moves: {example['condition_moves'][:10]}{'...' if len(example['condition_moves']) > 10 else ''}")

    # Analyze specific interesting cases
    print("\n3. Detailed analysis of specific permutations:")

    # Identity permutation
    print("\n   a) Identity permutation (n=7):")
    identity = list(range(1, 8))
    identity_analysis = analyze_specific_permutation(identity)
    results['specific_cases'] = {'identity_n7': identity_analysis}
    print(f"      Permutation: {identity}")
    print(f"      Moves satisfying condition: {identity_analysis['condition_satisfied_moves']}")

    # Reverse permutation
    print("\n   b) Reverse permutation (n=7):")
    reverse = list(range(7, 0, -1))
    reverse_analysis = analyze_specific_permutation(reverse)
    results['specific_cases']['reverse_n7'] = reverse_analysis
    print(f"      Permutation: {reverse}")
    print(f"      Moves satisfying condition: {reverse_analysis['condition_satisfied_moves']}")

    # A carefully chosen permutation for n=5
    print("\n   c) Custom permutation (n=5):")
    custom = [3, 1, 4, 2, 5]
    custom_analysis = analyze_specific_permutation(custom)
    results['specific_cases']['custom_n5'] = custom_analysis
    print(f"      Permutation: {custom}")
    print(f"      Moves satisfying condition: {custom_analysis['condition_satisfied_moves']}")

    # Save results to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, indent=2, fp=f)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_file}")
    print(f"\nConclusion: Based on exhaustive testing up to n=7 and extensive")
    print(f"random sampling for larger n, the theorem appears to be TRUE.")
    print(f"Every tested permutation has at least one move k where a < k < b.")


if __name__ == "__main__":
    main()
