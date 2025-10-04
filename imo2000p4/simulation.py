"""
IMO 2000 Problem 4: Magician's Card Trick Simulation

This simulation explores the problem of distributing 100 cards (numbered 1-100)
into three boxes (Red, White, Blue) such that given the sum of two cards chosen
from two different boxes, the magician can uniquely identify which box was NOT used.

Key Insight:
For the trick to work, the sets of possible sums for each pair of boxes must be disjoint.
- Sum from Red + White should uniquely identify that Blue was not chosen
- Sum from Red + Blue should uniquely identify that White was not chosen
- Sum from White + Blue should uniquely identify that Red was not chosen

Mathematical Analysis:
If we denote the boxes as R, W, B with minimum elements r, w, b and maximum elements
R_max, W_max, B_max, then for the ranges to be disjoint:
- Range(R+W): [r+w, R_max+W_max]
- Range(R+B): [r+b, R_max+B_max]
- Range(W+B): [w+b, W_max+B_max]

These ranges must not overlap for the trick to work.
"""

import json
import itertools
from typing import List, Tuple, Set, Dict
from collections import defaultdict


class CardTrick:
    """Simulates the magician's card trick with 100 cards in 3 boxes."""

    def __init__(self, n_cards: int = 100):
        """
        Initialize the card trick simulator.

        Args:
            n_cards: Total number of cards (default 100)
        """
        self.n_cards = n_cards
        self.cards = list(range(1, n_cards + 1))

    def check_distribution(self, red: List[int], white: List[int], blue: List[int]) -> bool:
        """
        Check if a distribution of cards allows the trick to work.

        Args:
            red, white, blue: Lists of card numbers in each box

        Returns:
            True if the trick works, False otherwise
        """
        # Verify all cards are used exactly once
        all_cards = sorted(red + white + blue)
        if all_cards != self.cards:
            return False

        # Each box must have at least one card
        if not (red and white and blue):
            return False

        # Compute all possible sums for each pair of boxes
        sums_rw = {r + w: 'blue' for r in red for w in white}
        sums_rb = {r + b: 'white' for r in red for b in blue}
        sums_wb = {w + b: 'red' for w in white for b in blue}

        # Check for conflicts: same sum from different pairs
        all_sums = {}
        for s, box in sums_rw.items():
            if s in all_sums:
                return False  # Conflict
            all_sums[s] = box

        for s, box in sums_rb.items():
            if s in all_sums:
                return False  # Conflict
            all_sums[s] = box

        for s, box in sums_wb.items():
            if s in all_sums:
                return False  # Conflict
            all_sums[s] = box

        return True

    def get_sum_ranges(self, red: List[int], white: List[int], blue: List[int]) -> Dict:
        """
        Get the ranges of possible sums for each pair of boxes.

        Returns:
            Dictionary with sum ranges for each pair
        """
        return {
            'red_white': (min(red) + min(white), max(red) + max(white)),
            'red_blue': (min(red) + min(blue), max(red) + max(blue)),
            'white_blue': (min(white) + min(blue), max(white) + max(blue))
        }

    def count_valid_distributions_pattern(self) -> Tuple[int, List[Dict]]:
        """
        Count valid distributions based on mathematical patterns.

        The key insight is that valid distributions must have disjoint sum ranges.
        This can be achieved through:
        1. Consecutive ranges
        2. Modulo-based partitions
        3. Other partition schemes

        Returns:
            (count, list of example distributions)
        """
        count = 0
        examples = []

        # TYPE 1: Try all consecutive partitions
        for i in range(1, self.n_cards):
            for j in range(i + 1, self.n_cards):
                group1 = list(range(1, i + 1))
                group2 = list(range(i + 1, j + 1))
                group3 = list(range(j + 1, self.n_cards + 1))

                for perm in itertools.permutations([group1, group2, group3]):
                    red, white, blue = perm
                    if self.check_distribution(red, white, blue):
                        count += 1
                        if len(examples) < 5:
                            examples.append({
                                'type': 'consecutive',
                                'red': red,
                                'white': white,
                                'blue': blue,
                                'ranges': self.get_sum_ranges(red, white, blue)
                            })

        # TYPE 2: Try modulo-based partitions
        # Test moduli from 3 to 10
        for mod in range(3, min(11, self.n_cards + 1)):
            # Partition by residue classes
            residue_classes = [[] for _ in range(mod)]
            for card in self.cards:
                residue_classes[card % mod].append(card)

            # Only use if we have exactly 3 non-empty classes
            non_empty = [rc for rc in residue_classes if rc]
            if len(non_empty) == 3:
                for perm in itertools.permutations(non_empty):
                    red, white, blue = perm
                    if self.check_distribution(red, white, blue):
                        count += 1
                        if len(examples) < 15:
                            examples.append({
                                'type': f'modulo-{mod}',
                                'red': sorted(red)[:3] + (['...'] if len(red) > 3 else []),
                                'white': sorted(white)[:3] + (['...'] if len(white) > 3 else []),
                                'blue': sorted(blue)[:3] + (['...'] if len(blue) > 3 else []),
                                'ranges': self.get_sum_ranges(red, white, blue)
                            })

        return count, examples

    def count_valid_consecutive_formula(self) -> int:
        """
        Count valid distributions using the mathematical formula.

        For n cards, we split into 3 consecutive groups at positions k and m.
        There are C(n-1, 2) ways to choose split points.
        Each split can be assigned to boxes in 6 ways, but only certain
        arrangements maintain disjoint sum ranges.

        Mathematical analysis shows that for 3 consecutive groups A < B < C:
        - If one box gets a singleton (1 card), valid arrangements exist
        - The pattern from small cases: specific permutations work

        Returns:
            Theoretical count based on formula
        """
        n = self.n_cards
        # Number of ways to split into 3 consecutive groups
        num_splits = (n - 1) * (n - 2) // 2  # C(n-1, 2)

        # From observation of small cases and the consecutive pattern:
        # Not all 6 permutations work. Analysis shows specific patterns.
        # The actual answer is related to the structure of valid assignments.

        return num_splits * 6  # Upper bound, actual may be lower

    def analyze_small_case(self, n: int = 4) -> Dict:
        """
        Analyze a small case exhaustively to find patterns.

        Args:
            n: Number of cards to use (default 4)

        Returns:
            Dictionary with analysis results
        """
        cards = list(range(1, n + 1))
        valid_distributions = []
        total_checked = 0

        # Generate all possible ways to partition n cards into 3 non-empty sets
        # This is computationally expensive, so only for small n

        def partition_into_3(items):
            """Generate all ways to partition items into 3 non-empty groups."""
            n_items = len(items)
            for i in range(1, 2**(n_items - 1)):
                for j in range(1, 2**(n_items - 1)):
                    set1, set2, set3 = [], [], []
                    for k, item in enumerate(items):
                        if i & (1 << k):
                            set1.append(item)
                        elif j & (1 << k):
                            set2.append(item)
                        else:
                            set3.append(item)

                    if set1 and set2 and set3:
                        yield (set1, set2, set3)

        seen = set()
        for red, white, blue in partition_into_3(cards):
            # Normalize to avoid counting permutations
            key = tuple(sorted([tuple(sorted(red)), tuple(sorted(white)), tuple(sorted(blue))]))
            if key in seen:
                continue
            seen.add(key)

            total_checked += 1

            # Check all permutations of box assignments
            for perm in itertools.permutations([red, white, blue]):
                r, w, b = perm
                if self.check_distribution(r, w, b):
                    valid_distributions.append({
                        'red': sorted(r),
                        'white': sorted(w),
                        'blue': sorted(b),
                        'ranges': self.get_sum_ranges(r, w, b)
                    })

        return {
            'n_cards': n,
            'total_partitions_checked': total_checked,
            'valid_count': len(valid_distributions),
            'valid_distributions': valid_distributions
        }


def find_pattern_and_formula():
    """
    Analyze the problem to find the pattern and derive the formula.

    Mathematical insight:
    For disjoint sum ranges, we need boxes with consecutive card ranges.
    If boxes contain [1..k], [k+1..m], [m+1..100], there are specific
    orderings that work.
    """

    results = {
        'problem_statement': 'IMO 2000 P4: Magician Card Trick',
        'n_cards': 100,
        'analysis': {}
    }

    # Analyze small cases
    trick = CardTrick(100)

    print("Analyzing small cases to find pattern...")
    for n in [3, 4, 5]:
        small_trick = CardTrick(n)
        small_analysis = small_trick.analyze_small_case(n)
        results['analysis'][f'n={n}'] = small_analysis
        print(f"n={n}: Found {small_analysis['valid_count']} valid distributions")

        # Show some examples
        if small_analysis['valid_distributions']:
            print(f"  Example: R={small_analysis['valid_distributions'][0]['red']}, "
                  f"W={small_analysis['valid_distributions'][0]['white']}, "
                  f"B={small_analysis['valid_distributions'][0]['blue']}")
            print(f"  Sum ranges: {small_analysis['valid_distributions'][0]['ranges']}")

    # For n=100, use the pattern-based approach
    print("\nCounting valid distributions for n=100 using pattern...")
    count, examples = trick.count_valid_distributions_pattern()
    theoretical_count = trick.count_valid_consecutive_formula()

    results['n_100_solution'] = {
        'total_valid_distributions': count,
        'theoretical_upper_bound': theoretical_count,
        'example_distributions': examples,
        'note': 'Includes both consecutive and modulo-based partitions. Other partition types may exist.'
    }

    consecutive_count = len([e for e in examples if e.get('type') == 'consecutive'])
    modulo_count = len([e for e in examples if 'modulo' in e.get('type', '')])

    print(f"\nFor n=100 cards:")
    print(f"  - Total valid distributions found: {count}")
    print(f"  - Consecutive type examples: {consecutive_count}")
    print(f"  - Modulo-based type examples: {modulo_count}")
    print(f"  - Theoretical upper bound: {theoretical_count}")

    # Mathematical insight
    results['mathematical_insight'] = {
        'key_observation': 'Valid distributions require disjoint sum ranges',
        'partition_types': {
            'consecutive': (
                'Only one consecutive partition works: {1}, {2..99}, {100}. '
                'This gives 6 distributions (3! permutations).'
            ),
            'modulo_based': (
                'Partitions by residue classes modulo k work when they create 3 non-empty classes. '
                'For example, modulo 3 partitions {1-100} into classes with residues 0, 1, 2. '
                'Each gives 6 distributions (3! permutations).'
            ),
            'other': 'Additional partition types may exist based on other mathematical structures.'
        },
        'modulo_3_example': {
            'partition': 'A={3,6,9,...,99}, B={1,4,7,...,100}, C={2,5,8,...,98}',
            'sum_residues': 'S_AB ≡ 1 (mod 3), S_AC ≡ 2 (mod 3), S_BC ≡ 0 (mod 3)',
            'conclusion': 'Sum sets are disjoint due to different residues mod 3'
        },
        'total_answer': (
            f'At least {count} valid distributions found. '
            'This includes consecutive and modulo-based partitions. '
            'More partition types may exist.'
        )
    }

    return results


def main():
    """Main function to run the simulation and save results."""

    print("=" * 70)
    print("IMO 2000 Problem 4: Magician's Card Trick Simulation")
    print("=" * 70)
    print()

    results = find_pattern_and_formula()

    # Save results to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nKey Findings:")
    print(f"- For small cases, the pattern is clear")
    print(f"- For n=100: {results['n_100_solution']['total_valid_distributions']} valid distributions found")
    print(f"- Includes consecutive partitions (6) and modulo-based partitions (≥6)")
    print(f"- Multiple partition types work: consecutive, modulo-3, and potentially others")
    print("\nSee results.json for complete data")


if __name__ == '__main__':
    main()
