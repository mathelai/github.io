"""
IMO 2017 Problem 5 Simulation

Problem: Given N >= 2, we have N(N+1) soccer players of distinct heights in a row.
We need to remove N(N-1) players, leaving 2N players such that:
- Condition 1: No one stands between the two tallest players
- Condition 2: No one stands between the 3rd and 4th tallest players
- ...
- Condition N: No one stands between the two shortest players

This simulation explores the problem by:
1. Generating random arrangements of players
2. Finding valid selections using a constructive algorithm
3. Analyzing patterns and generating ground truth data
"""

import json
import itertools
from typing import List, Tuple, Optional, Dict, Set
import random


def generate_players(n: int) -> List[int]:
    """Generate N(N+1) players with distinct heights."""
    num_players = n * (n + 1)
    return list(range(1, num_players + 1))


def check_condition(selected_positions: List[int], heights: List[int], rank_pair: Tuple[int, int]) -> bool:
    """
    Check if a condition is satisfied for a given rank pair.

    Args:
        selected_positions: List of positions (indices) that are selected
        heights: List of heights at each position
        rank_pair: Tuple of (rank1, rank2) where rank1 < rank2 (1 is tallest)

    Returns:
        True if no player stands between the two players of given ranks
    """
    # Find the heights corresponding to these ranks
    all_heights = sorted([heights[p] for p in selected_positions], reverse=True)

    if len(all_heights) < max(rank_pair):
        return False

    height1 = all_heights[rank_pair[0] - 1]
    height2 = all_heights[rank_pair[1] - 1]

    # Find positions of these two heights
    pos1 = None
    pos2 = None
    for p in selected_positions:
        if heights[p] == height1:
            pos1 = p
        if heights[p] == height2:
            pos2 = p

    if pos1 is None or pos2 is None:
        return False

    # Ensure pos1 < pos2
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1

    # Check if any selected position is between pos1 and pos2
    for p in selected_positions:
        if pos1 < p < pos2:
            return False

    return True


def check_all_conditions(selected_positions: List[int], heights: List[int], n: int) -> bool:
    """Check if all N conditions are satisfied."""
    # Generate the rank pairs: (1,2), (3,4), ..., (2N-1, 2N)
    for i in range(n):
        rank1 = 2 * i + 1
        rank2 = 2 * i + 2
        if not check_condition(selected_positions, heights, (rank1, rank2)):
            return False
    return True


def greedy_selection(heights: List[int], n: int) -> Optional[List[int]]:
    """
    Greedy algorithm: Select 2N players using a constructive approach.

    Strategy: Build pairs from tallest to shortest, ensuring each pair is adjacent
    in the final selection.
    """
    num_players = len(heights)

    # Create a list of (height, original_position) tuples
    indexed_heights = [(heights[i], i) for i in range(num_players)]
    indexed_heights.sort(reverse=True)  # Sort by height (tallest first)

    selected_positions = []

    # We need to select 2N players forming N pairs
    # Each pair should be adjacent in position among selected players

    # Try to build pairs greedily
    for pair_idx in range(n):
        # Get the next two tallest unselected players
        candidates = [x for x in indexed_heights if x[1] not in selected_positions]

        if len(candidates) < 2:
            return None

        # Take the two tallest remaining
        height1, pos1 = candidates[0]
        height2, pos2 = candidates[1]

        # Add them to selected positions
        selected_positions.extend([pos1, pos2])

    return sorted(selected_positions)


def recursive_selection(heights: List[int], n: int, max_depth: int = 1000) -> Optional[List[int]]:
    """
    Recursive algorithm: Build selection by processing pairs in order.

    This implements the constructive proof strategy:
    - Process pairs from shortest to tallest
    - For each pair, find two adjacent positions in the current arrangement
    """
    num_players = len(heights)

    # Create a list of (height, original_position) tuples
    indexed_heights = [(heights[i], i) for i in range(num_players)]
    indexed_heights.sort()  # Sort by height (shortest first)

    # Available positions (all initially available)
    available = set(range(num_players))
    selected = []

    # Process pairs from shortest to tallest
    for pair_idx in range(n):
        # Get the two shortest available players
        candidates = [(h, p) for h, p in indexed_heights if p in available]

        if len(candidates) < 2:
            return None

        # Get the two shortest
        height1, pos1 = candidates[0]
        height2, pos2 = candidates[1]

        # Find a way to select them such that they become adjacent
        # Remove all positions between them
        min_pos = min(pos1, pos2)
        max_pos = max(pos1, pos2)

        # Remove positions between them from available
        to_remove = [p for p in available if min_pos < p < max_pos]
        for p in to_remove:
            available.remove(p)

        # Mark these two as selected (remove from available)
        available.remove(pos1)
        available.remove(pos2)
        selected.extend([pos1, pos2])

    return sorted(selected) if len(selected) == 2 * n else None


def brute_force_search(heights: List[int], n: int) -> Optional[List[int]]:
    """Brute force search for valid selection (only for small N)."""
    num_players = len(heights)
    target_size = 2 * n

    if n > 3:  # Too expensive for large N
        return None

    # Try all combinations of target_size positions
    for combo in itertools.combinations(range(num_players), target_size):
        positions = list(combo)
        if check_all_conditions(positions, heights, n):
            return positions

    return None


def sir_alex_algorithm(heights: List[int], n: int) -> Optional[List[int]]:
    """
    Sir Alex's algorithm: The constructive proof approach.

    Algorithm (greedy approach):
    1. For each iteration i from 1 to n:
       - Find the two tallest players among those still remaining
       - Remove all players strictly between their positions
       - Mark these two as "processed" (they form a pair)
    2. After n iterations, exactly 2n players should remain

    This ensures that for pair i (ranks 2i-1 and 2i), no one stands between them.
    """
    num_players = len(heights)

    # Track which players are still active
    active = list(range(num_players))

    # Process n pairs
    for round_num in range(n):
        if len(active) < 2:
            return None

        # Find indices in active list of the two tallest players
        active_heights = [(heights[i], i, idx) for idx, i in enumerate(active)]
        active_heights.sort(reverse=True)

        # Get the two tallest
        _, pos1, idx1 = active_heights[0]
        _, pos2, idx2 = active_heights[1]

        # Find leftmost and rightmost positions
        left_pos = min(pos1, pos2)
        right_pos = max(pos1, pos2)

        # Remove all players strictly between left_pos and right_pos
        new_active = []
        for p in active:
            # Keep if not strictly between
            if not (left_pos < p < right_pos):
                new_active.append(p)

        active = new_active

    # Should have exactly 2n players remaining
    if len(active) != 2 * n:
        return None

    return sorted(active)


def analyze_selection(heights: List[int], selected: List[int], n: int) -> Dict:
    """Analyze a valid selection and return detailed information."""
    # Get heights of selected players in order
    selected_heights = [heights[p] for p in selected]

    # Compute ranks (1 = tallest)
    all_heights_sorted = sorted(selected_heights, reverse=True)
    ranks = [all_heights_sorted.index(h) + 1 for h in selected_heights]

    # Identify pairs
    pairs = []
    for i in range(n):
        rank1 = 2 * i + 1
        rank2 = 2 * i + 2

        # Find positions of these ranks
        pos1 = ranks.index(rank1)
        pos2 = ranks.index(rank2)

        pairs.append({
            'pair_number': i + 1,
            'ranks': (rank1, rank2),
            'positions_in_selection': (pos1, pos2),
            'original_positions': (selected[pos1], selected[pos2]),
            'heights': (selected_heights[pos1], selected_heights[pos2]),
            'adjacent': abs(pos1 - pos2) == 1
        })

    return {
        'selected_positions': selected,
        'selected_heights': selected_heights,
        'ranks': ranks,
        'pairs': pairs,
        'num_adjacent_pairs': sum(1 for p in pairs if p['adjacent'])
    }


def run_experiments(n_values: List[int], trials_per_n: int = 10) -> Dict:
    """Run experiments for different values of N."""
    results = {
        'experiments': [],
        'summary': {}
    }

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"Testing N = {n}")
        print(f"{'='*60}")
        print(f"Number of players: {n * (n + 1)}")
        print(f"Players to select: {2 * n}")
        print(f"Players to remove: {n * (n - 1)}")

        successes = 0
        failed_arrangements = []
        successful_selections = []

        for trial in range(trials_per_n):
            # Generate random arrangement
            players = generate_players(n)
            random.shuffle(players)

            # Try Sir Alex's algorithm
            selected = sir_alex_algorithm(players, n)

            if selected and check_all_conditions(selected, players, n):
                successes += 1
                analysis = analyze_selection(players, selected, n)
                successful_selections.append({
                    'trial': trial + 1,
                    'arrangement': players,
                    'analysis': analysis
                })
                print(f"  Trial {trial + 1}: SUCCESS")
            else:
                failed_arrangements.append({
                    'trial': trial + 1,
                    'arrangement': players,
                    'selected': selected
                })
                print(f"  Trial {trial + 1}: FAILED")

        success_rate = successes / trials_per_n
        print(f"\nSuccess Rate: {successes}/{trials_per_n} = {success_rate:.1%}")

        # Store results for this N
        experiment_result = {
            'n': n,
            'num_players': n * (n + 1),
            'num_to_select': 2 * n,
            'num_to_remove': n * (n - 1),
            'trials': trials_per_n,
            'successes': successes,
            'success_rate': success_rate,
            'successful_selections': successful_selections[:3],  # Keep first 3
            'failed_arrangements': failed_arrangements[:3]  # Keep first 3
        }

        results['experiments'].append(experiment_result)

    # Generate summary
    results['summary'] = {
        'n_values_tested': n_values,
        'total_trials': len(n_values) * trials_per_n,
        'overall_success_rate': sum(exp['successes'] for exp in results['experiments']) / (len(n_values) * trials_per_n),
        'all_successful': all(exp['success_rate'] == 1.0 for exp in results['experiments'])
    }

    return results


def demonstrate_small_case(n: int = 2):
    """Demonstrate the algorithm for a small case with detailed output."""
    print(f"\n{'='*60}")
    print(f"DETAILED DEMONSTRATION FOR N = {n}")
    print(f"{'='*60}\n")

    # Use a specific arrangement for reproducibility
    random.seed(42)
    players = generate_players(n)
    random.shuffle(players)

    print(f"Initial arrangement (positions 0 to {len(players)-1}):")
    print(f"Heights: {players}")
    print(f"\nWe need to select {2*n} players and remove {n*(n-1)} players")
    print(f"\nConditions to satisfy:")
    for i in range(n):
        print(f"  {i+1}. No one between rank {2*i+1} and rank {2*i+2} players")

    # Run algorithm with detailed trace
    print(f"\n{'='*60}")
    print("APPLYING SIR ALEX'S ALGORITHM")
    print(f"{'='*60}\n")

    selected = sir_alex_algorithm(players, n)

    if selected:
        print(f"\nSelected positions: {selected}")
        print(f"Selected heights: {[players[p] for p in selected]}")

        # Verify conditions
        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}\n")

        if check_all_conditions(selected, players, n):
            print("All conditions satisfied!")
            analysis = analyze_selection(players, selected, n)
            print(f"\nPair details:")
            for pair in analysis['pairs']:
                print(f"\n  Pair {pair['pair_number']} (ranks {pair['ranks'][0]}, {pair['ranks'][1]}):")
                print(f"    Heights: {pair['heights']}")
                print(f"    Original positions: {pair['original_positions']}")
                print(f"    Positions in selection: {pair['positions_in_selection']}")
                print(f"    Adjacent in selection: {pair['adjacent']}")
        else:
            print("Conditions NOT satisfied!")
    else:
        print("Algorithm failed to find a selection!")

    return players, selected


def main():
    """Main function to run all experiments."""
    print("IMO 2017 Problem 5 - Simulation")
    print("=" * 60)

    # Demonstrate small case
    demonstrate_small_case(n=2)
    demonstrate_small_case(n=3)

    # Run experiments
    n_values = [2, 3, 4, 5, 6]
    trials_per_n = 20

    results = run_experiments(n_values, trials_per_n)

    # Save results to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_file}")
    print(f"\nOverall Statistics:")
    print(f"  N values tested: {results['summary']['n_values_tested']}")
    print(f"  Total trials: {results['summary']['total_trials']}")
    print(f"  Overall success rate: {results['summary']['overall_success_rate']:.1%}")
    print(f"  All trials successful: {results['summary']['all_successful']}")

    if results['summary']['all_successful']:
        print("\nConclusion: The algorithm successfully found valid selections")
        print("for all tested cases, supporting the conjecture that it's always possible.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
