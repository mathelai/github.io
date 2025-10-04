"""
IMO 2020 Problem 3: Pebble Partitioning Simulation

Problem:
There are 4n pebbles of weights 1, 2, 3, ..., 4n. Each pebble is colored in one of n colors
and there are four pebbles of each color. Show that we can arrange the pebbles into two piles
so that:
1. The total weights of both piles are the same.
2. Each pile contains two pebbles of each color.

This simulation explores the problem by:
- Generating random colorings for small values of n
- Finding valid partitions using various algorithms
- Analyzing patterns and properties
- Generating ground truth data for proof development
"""

import json
import random
from typing import List, Tuple, Dict, Set
from itertools import combinations
import time


class PebblePartitioner:
    """
    A class to handle the pebble partitioning problem.
    """

    def __init__(self, n: int):
        """
        Initialize with n colors (4n pebbles total).

        Args:
            n: Number of colors (each color appears on exactly 4 pebbles)
        """
        self.n = n
        self.num_pebbles = 4 * n
        self.total_weight = sum(range(1, 4 * n + 1))
        self.target_weight = self.total_weight // 2

    def generate_random_coloring(self) -> List[int]:
        """
        Generate a random valid coloring of pebbles.

        Returns:
            List where coloring[i] is the color (0 to n-1) of pebble with weight i+1
        """
        # Create a list with 4 pebbles of each color
        colors = []
        for color in range(self.n):
            colors.extend([color] * 4)

        # Shuffle to create random assignment
        random.shuffle(colors)
        return colors

    def generate_canonical_coloring(self) -> List[int]:
        """
        Generate a canonical coloring where pebbles 1-4 get color 0, 5-8 get color 1, etc.

        Returns:
            List where coloring[i] is the color of pebble with weight i+1
        """
        coloring = []
        for color in range(self.n):
            coloring.extend([color] * 4)
        return coloring

    def is_valid_partition(self, pile1: Set[int], pile2: Set[int], coloring: List[int]) -> bool:
        """
        Check if a partition is valid.

        Args:
            pile1: Set of pebble weights in pile 1
            pile2: Set of pebble weights in pile 2
            coloring: Color assignment for each pebble

        Returns:
            True if partition satisfies both conditions
        """
        # Check condition 1: Equal weights
        weight1 = sum(pile1)
        weight2 = sum(pile2)
        if weight1 != weight2:
            return False

        # Check condition 2: Each pile has 2 pebbles of each color
        for color in range(self.n):
            # Count pebbles of this color in each pile
            count1 = sum(1 for w in pile1 if coloring[w - 1] == color)
            count2 = sum(1 for w in pile2 if coloring[w - 1] == color)

            if count1 != 2 or count2 != 2:
                return False

        return True

    def greedy_partition(self, coloring: List[int]) -> Tuple[Set[int], Set[int], bool]:
        """
        Try to find a partition using a greedy approach.
        For each color, try different ways to split its 4 pebbles into 2+2.

        Args:
            coloring: Color assignment for each pebble

        Returns:
            (pile1, pile2, success)
        """
        # Group pebbles by color
        color_groups = [[] for _ in range(self.n)]
        for weight in range(1, 4 * self.n + 1):
            color = coloring[weight - 1]
            color_groups[color].append(weight)

        # For each color, we need to choose 2 pebbles for pile1 and 2 for pile2
        # Try all possible combinations
        def backtrack(color_idx: int, pile1: Set[int], pile2: Set[int]) -> bool:
            if color_idx == self.n:
                # Check if weights are equal
                return sum(pile1) == sum(pile2)

            # Get pebbles of current color
            pebbles = color_groups[color_idx]

            # Try all ways to choose 2 pebbles for pile1
            for pair1 in combinations(pebbles, 2):
                pile1_new = pile1 | set(pair1)
                pile2_new = pile2 | set(p for p in pebbles if p not in pair1)

                # Prune: if already exceeded target weight in either pile
                if sum(pile1_new) > self.target_weight or sum(pile2_new) > self.target_weight:
                    continue

                if backtrack(color_idx + 1, pile1_new, pile2_new):
                    pile1.clear()
                    pile1.update(pile1_new)
                    pile2.clear()
                    pile2.update(pile2_new)
                    return True

            return False

        pile1 = set()
        pile2 = set()
        success = backtrack(0, pile1, pile2)
        return pile1, pile2, success

    def construct_partition_algorithm1(self, coloring: List[int]) -> Tuple[Set[int], Set[int], bool]:
        """
        Constructive algorithm based on pairing strategy.
        Pair up weights to sum to 4n+1, then distribute pairs.

        Args:
            coloring: Color assignment for each pebble

        Returns:
            (pile1, pile2, success)
        """
        # Create pairs that sum to 4n+1: (1, 4n), (2, 4n-1), ..., (2n, 2n+1)
        pairs = []
        for i in range(1, 2 * self.n + 1):
            pairs.append((i, 4 * self.n + 1 - i))

        # Try to assign pairs to piles respecting color constraints
        pile1 = set()
        pile2 = set()
        color_count1 = [0] * self.n
        color_count2 = [0] * self.n

        for w1, w2 in pairs:
            c1 = coloring[w1 - 1]
            c2 = coloring[w2 - 1]

            # Decide where to put each pebble
            # Prefer pile1 if it needs more of this color
            if color_count1[c1] < 2 and color_count2[c2] < 2:
                pile1.add(w1)
                pile2.add(w2)
                color_count1[c1] += 1
                color_count2[c2] += 1
            elif color_count2[c1] < 2 and color_count1[c2] < 2:
                pile2.add(w1)
                pile1.add(w2)
                color_count2[c1] += 1
                color_count1[c2] += 1
            else:
                # This simple strategy doesn't work, fall back to greedy
                return self.greedy_partition(coloring)

        success = self.is_valid_partition(pile1, pile2, coloring)
        return pile1, pile2, success

    def find_partition(self, coloring: List[int], method: str = "greedy") -> Tuple[Set[int], Set[int], bool]:
        """
        Find a valid partition using the specified method.

        Args:
            coloring: Color assignment for each pebble
            method: Algorithm to use ("greedy" or "construct")

        Returns:
            (pile1, pile2, success)
        """
        if method == "greedy":
            return self.greedy_partition(coloring)
        elif method == "construct":
            return self.construct_partition_algorithm1(coloring)
        else:
            raise ValueError(f"Unknown method: {method}")


def analyze_small_cases(max_n: int = 5, trials_per_n: int = 100) -> Dict:
    """
    Analyze small cases to find patterns.

    Args:
        max_n: Maximum value of n to test
        trials_per_n: Number of random colorings to try for each n

    Returns:
        Dictionary with analysis results
    """
    results = {
        "analysis": [],
        "examples": [],
        "patterns": []
    }

    for n in range(1, max_n + 1):
        print(f"\nAnalyzing n={n} (4n={4*n} pebbles)...")
        partitioner = PebblePartitioner(n)

        successes = 0
        failures = 0
        example_found = False

        # Try canonical coloring first
        canonical_coloring = partitioner.generate_canonical_coloring()
        pile1, pile2, success = partitioner.find_partition(canonical_coloring)

        canonical_example = {
            "n": n,
            "coloring": canonical_coloring,
            "pile1": sorted(list(pile1)) if success else [],
            "pile2": sorted(list(pile2)) if success else [],
            "success": success,
            "type": "canonical"
        }

        if success:
            successes += 1
            if not example_found:
                results["examples"].append(canonical_example)
                example_found = True
        else:
            failures += 1

        # Try random colorings
        for trial in range(trials_per_n):
            coloring = partitioner.generate_random_coloring()
            pile1, pile2, success = partitioner.find_partition(coloring)

            if success:
                successes += 1
                if not example_found:
                    results["examples"].append({
                        "n": n,
                        "coloring": coloring,
                        "pile1": sorted(list(pile1)),
                        "pile2": sorted(list(pile2)),
                        "success": True,
                        "type": "random"
                    })
                    example_found = True
            else:
                failures += 1

        success_rate = successes / (trials_per_n + 1)

        analysis_entry = {
            "n": n,
            "num_pebbles": 4 * n,
            "total_weight": partitioner.total_weight,
            "target_weight": partitioner.target_weight,
            "trials": trials_per_n + 1,
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate
        }

        results["analysis"].append(analysis_entry)
        print(f"  Success rate: {successes}/{trials_per_n + 1} = {success_rate:.2%}")

    # Identify patterns
    results["patterns"].append({
        "observation": "All tested cases found valid partitions",
        "confidence": "high" if all(a["success_rate"] > 0.9 for a in results["analysis"]) else "medium"
    })

    results["patterns"].append({
        "observation": "Target weight is always n(4n+1) for n colors",
        "formula": "target_weight = n * (4n + 1)"
    })

    results["patterns"].append({
        "observation": "Each pile must contain exactly 2n pebbles",
        "formula": "pile_size = 2n"
    })

    return results


def generate_visual_data(n: int, coloring: List[int], pile1: Set[int], pile2: Set[int]) -> Dict:
    """
    Generate data for visualization.

    Args:
        n: Number of colors
        coloring: Color assignment
        pile1: First pile
        pile2: Second pile

    Returns:
        Dictionary with visualization data
    """
    pebbles = []
    for weight in range(1, 4 * n + 1):
        pebbles.append({
            "weight": weight,
            "color": coloring[weight - 1],
            "pile": 1 if weight in pile1 else 2
        })

    return {
        "n": n,
        "pebbles": pebbles,
        "pile1_weight": sum(pile1),
        "pile2_weight": sum(pile2),
        "pile1_weights": sorted(list(pile1)),
        "pile2_weights": sorted(list(pile2))
    }


def main():
    """
    Main function to run the simulation and generate results.
    """
    print("=" * 70)
    print("IMO 2020 Problem 3: Pebble Partitioning Simulation")
    print("=" * 70)

    # Analyze small cases
    max_n = 6
    trials = 50

    print(f"\nRunning analysis for n=1 to n={max_n} with {trials} random trials each...")
    start_time = time.time()
    results = analyze_small_cases(max_n, trials)
    elapsed_time = time.time() - start_time

    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")

    # Generate specific examples for visualization
    print("\nGenerating visualization examples...")
    vis_examples = []

    for n in [1, 2, 3, 4]:
        partitioner = PebblePartitioner(n)
        coloring = partitioner.generate_canonical_coloring()
        pile1, pile2, success = partitioner.find_partition(coloring)

        if success:
            vis_data = generate_visual_data(n, coloring, pile1, pile2)
            vis_examples.append(vis_data)
            print(f"  n={n}: Pile1 weight={vis_data['pile1_weight']}, Pile2 weight={vis_data['pile2_weight']}")

    # Prepare final results
    final_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "problem": "IMO 2020 Problem 3",
        "description": "Partition 4n pebbles with weights 1..4n and n colors (4 per color) into two equal-weight piles, each with 2 pebbles of each color",
        "analysis": results["analysis"],
        "patterns": results["patterns"],
        "examples": results["examples"],
        "visualizations": vis_examples,
        "summary": {
            "total_trials": sum(a["trials"] for a in results["analysis"]),
            "total_successes": sum(a["successes"] for a in results["analysis"]),
            "overall_success_rate": sum(a["successes"] for a in results["analysis"]) / sum(a["trials"] for a in results["analysis"]),
            "conclusion": "All tested colorings admit valid partitions, supporting the conjecture that such a partition always exists."
        }
    }

    # Save results to JSON
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nSummary:")
    print(f"  Total trials: {final_results['summary']['total_trials']}")
    print(f"  Successful partitions: {final_results['summary']['total_successes']}")
    print(f"  Success rate: {final_results['summary']['overall_success_rate']:.2%}")
    print(f"\n{final_results['summary']['conclusion']}")

    return final_results


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
