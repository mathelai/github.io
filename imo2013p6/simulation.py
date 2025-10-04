"""
IMO 2013 Problem 6 Simulation

This simulation explores beautiful labellings of n+1 points on a circle.

A labelling is beautiful if for any four labels a<b<c<d with a+d=b+c,
the chord joining points labelled a and d does not intersect the chord
joining points labelled b and c.

The problem asks to prove that M = N + 1, where:
- M = number of beautiful labellings (up to rotation)
- N = number of ordered pairs (x,y) with x+y ≤ n and gcd(x,y) = 1
"""

import json
import math
from itertools import permutations
from typing import List, Tuple, Set
from collections import defaultdict


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def count_coprime_pairs(n: int) -> int:
    """
    Count ordered pairs (x, y) of positive integers where:
    - x + y ≤ n
    - gcd(x, y) = 1

    Returns N in the problem statement.
    """
    count = 0
    pairs = []
    for x in range(1, n):
        for y in range(1, n):
            if x + y <= n and gcd(x, y) == 1:
                count += 1
                pairs.append((x, y))
    return count, pairs


def chords_intersect(a_pos: int, d_pos: int, b_pos: int, c_pos: int, n_points: int) -> bool:
    """
    Check if chord (a_pos, d_pos) intersects chord (b_pos, c_pos) on a circle.

    Two chords intersect if and only if their endpoints alternate around the circle.
    For positions p1, p2, p3, p4 in cyclic order, chords (p1,p3) and (p2,p4) intersect.

    Args:
        a_pos, d_pos: positions of endpoints of first chord
        b_pos, c_pos: positions of endpoints of second chord
        n_points: total number of points on circle
    """
    # Get all positions in sorted order
    positions = sorted([a_pos, b_pos, c_pos, d_pos])

    # Find which positions correspond to which chord
    # Two chords intersect iff their endpoints alternate
    # i.e., if we have positions in order p1 < p2 < p3 < p4,
    # then chords (p1,p3) and (p2,p4) intersect,
    # but chords (p1,p2) and (p3,p4) do not intersect.

    # Map positions back to chord identities
    chord1 = {a_pos, d_pos}
    chord2 = {b_pos, c_pos}

    # Check if positions alternate
    # Pattern for intersection: first chord has positions[0] and positions[2]
    #                          second chord has positions[1] and positions[3]
    if (positions[0] in chord1 and positions[2] in chord1 and
        positions[1] in chord2 and positions[3] in chord2):
        return True
    if (positions[0] in chord2 and positions[2] in chord2 and
        positions[1] in chord1 and positions[3] in chord1):
        return True

    return False


def is_beautiful(labelling: List[int]) -> bool:
    """
    Check if a labelling is beautiful.

    A labelling is beautiful if for any four labels a<b<c<d with a+d=b+c,
    the chord joining points labelled a and d does not intersect the chord
    joining points labelled b and c.

    Args:
        labelling: list where labelling[i] is the label at position i
    """
    n = len(labelling) - 1

    # Create position map: label -> position
    pos = {}
    for position, label in enumerate(labelling):
        pos[label] = position

    # Check all quadruples a < b < c < d with a + d = b + c
    for a in range(n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                d = b + c - a
                if d > c and d <= n:
                    # We have a < b < c < d with a + d = b + c
                    # Check if chord (a,d) intersects chord (b,c)
                    if chords_intersect(pos[a], pos[d], pos[b], pos[c], n + 1):
                        return False

    return True


def rotate_labelling(labelling: List[int]) -> List[int]:
    """Rotate labelling by one position clockwise."""
    return [labelling[-1]] + labelling[:-1]


def canonical_form(labelling: List[int]) -> Tuple[int, ...]:
    """
    Return the lexicographically smallest rotation of a labelling.
    This gives a canonical representative for each equivalence class.
    """
    n = len(labelling)
    min_rotation = labelling
    current = labelling[:]

    for _ in range(n - 1):
        current = rotate_labelling(current)
        if tuple(current) < tuple(min_rotation):
            min_rotation = current

    return tuple(min_rotation)


def count_beautiful_labellings(n: int) -> Tuple[int, List[List[int]]]:
    """
    Count beautiful labellings up to rotation.

    Args:
        n: parameter where we have n+1 points labeled 0,1,...,n

    Returns:
        (count, list of canonical beautiful labellings)
    """
    # Generate all permutations of 0,1,...,n
    labels = list(range(n + 1))

    # Store canonical forms to avoid counting rotations multiple times
    canonical_beautiful = set()
    all_beautiful = []

    for perm in permutations(labels):
        perm_list = list(perm)
        if is_beautiful(perm_list):
            canonical = canonical_form(perm_list)
            if canonical not in canonical_beautiful:
                canonical_beautiful.add(canonical)
                all_beautiful.append(list(canonical))

    return len(canonical_beautiful), all_beautiful


def analyze_beautiful_labellings(beautiful_labellings: List[List[int]]) -> dict:
    """
    Analyze patterns in beautiful labellings.
    """
    analysis = {
        'count': len(beautiful_labellings),
        'labellings': beautiful_labellings,
        'patterns': []
    }

    # Look for patterns
    for labelling in beautiful_labellings:
        n = len(labelling) - 1
        pattern = {
            'labelling': labelling,
            'zero_position': labelling.index(0),
            'max_position': labelling.index(n),
            'inversions': 0  # count of pairs (i,j) where i<j but labelling[i]>labelling[j]
        }

        # Count inversions
        for i in range(len(labelling)):
            for j in range(i + 1, len(labelling)):
                if labelling[i] > labelling[j]:
                    pattern['inversions'] += 1

        analysis['patterns'].append(pattern)

    return analysis


def generate_ground_truth(max_n: int = 7) -> dict:
    """
    Generate ground truth data for small values of n.

    This helps identify patterns and verify the conjecture M = N + 1.
    """
    results = {
        'conjecture': 'M = N + 1',
        'data': []
    }

    for n in range(3, max_n + 1):
        print(f"Computing for n={n}...")

        # Count beautiful labellings (M)
        M, beautiful = count_beautiful_labellings(n)

        # Count coprime pairs (N)
        N, coprime_pairs = count_coprime_pairs(n)

        # Analyze beautiful labellings
        analysis = analyze_beautiful_labellings(beautiful)

        result = {
            'n': n,
            'M': M,
            'N': N,
            'N_plus_1': N + 1,
            'conjecture_holds': (M == N + 1),
            'coprime_pairs': coprime_pairs[:20],  # Store first 20 for display
            'coprime_pairs_count': N,
            'beautiful_labellings': beautiful[:20],  # Store first 20 for display
            'beautiful_labellings_count': M,
            'analysis': analysis
        }

        results['data'].append(result)

        print(f"  n={n}: M={M}, N={N}, N+1={N+1}, Holds: {M == N + 1}")

    return results


def main():
    """Main function to run simulation and save results."""
    print("=" * 60)
    print("IMO 2013 Problem 6 Simulation")
    print("=" * 60)
    print()

    # Generate ground truth for small values
    results = generate_ground_truth(max_n=7)

    # Save to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to {output_file}")
    print()
    print("Summary:")
    print("-" * 60)
    for data in results['data']:
        status = "✓" if data['conjecture_holds'] else "✗"
        print(f"n={data['n']:2d}: M={data['M']:3d}, N+1={data['N_plus_1']:3d} {status}")
    print("-" * 60)


if __name__ == '__main__':
    main()
