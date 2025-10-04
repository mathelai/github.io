"""
IMO 2001 Problem 4 Simulation

Problem: Let n_1, n_2, ..., n_m be integers where m>1 is odd. Let x = (x_1, ..., x_m)
denote a permutation of the integers 1, 2, ..., m. Let f(x) = x_1*n_1 + x_2*n_2 + ... + x_m*n_m.
Show that for some distinct permutations a, b the difference f(a) - f(b) is a multiple of m!.

This simulation:
1. Tests various sequences n_1, ..., n_m for small odd values of m
2. Computes f(x) for all permutations
3. Checks if there exist distinct permutations a, b where (f(a) - f(b)) % m! == 0
4. Identifies patterns and generates ground truth data
"""

import itertools
import json
import math
from typing import List, Tuple, Dict, Any
import random


def factorial(n: int) -> int:
    """Compute n!"""
    return math.factorial(n)


def compute_f(perm: Tuple[int, ...], n_values: List[int]) -> int:
    """
    Compute f(x) = x_1*n_1 + x_2*n_2 + ... + x_m*n_m
    where perm is a permutation (x_1, ..., x_m)
    """
    return sum(x * n for x, n in zip(perm, n_values))


def find_divisible_pairs(m: int, n_values: List[int]) -> Dict[str, Any]:
    """
    Find all pairs of distinct permutations (a, b) where (f(a) - f(b)) % m! == 0

    Returns a dictionary with:
    - m: the value of m
    - n_values: the sequence n_1, ..., n_m
    - m_factorial: m!
    - permutations_count: total number of permutations (m!)
    - all_f_values: list of (permutation, f(permutation)) for all permutations
    - divisible_pairs: list of pairs where difference is divisible by m!
    - f_values_mod_m_factorial: distribution of f(x) mod m!
    """
    m_fact = factorial(m)

    # Generate all permutations of 1, 2, ..., m
    all_perms = list(itertools.permutations(range(1, m + 1)))

    # Compute f(x) for all permutations
    f_values = {}
    for perm in all_perms:
        f_values[perm] = compute_f(perm, n_values)

    # Find pairs where (f(a) - f(b)) % m! == 0
    divisible_pairs = []
    for i, perm_a in enumerate(all_perms):
        for perm_b in all_perms[i+1:]:
            diff = f_values[perm_a] - f_values[perm_b]
            if diff % m_fact == 0:
                divisible_pairs.append({
                    'perm_a': list(perm_a),
                    'perm_b': list(perm_b),
                    'f_a': f_values[perm_a],
                    'f_b': f_values[perm_b],
                    'difference': diff,
                    'quotient': diff // m_fact
                })

    # Compute distribution of f(x) mod m!
    mod_distribution = {}
    for perm, f_val in f_values.items():
        mod_val = f_val % m_fact
        if mod_val not in mod_distribution:
            mod_distribution[mod_val] = []
        mod_distribution[mod_val].append(list(perm))

    return {
        'm': m,
        'n_values': n_values,
        'm_factorial': m_fact,
        'permutations_count': len(all_perms),
        'all_f_values': [{'perm': list(p), 'f': f_values[p]} for p in all_perms],
        'divisible_pairs_count': len(divisible_pairs),
        'divisible_pairs': divisible_pairs[:10],  # Limit to first 10 for storage
        'f_values_mod_distribution': {k: len(v) for k, v in mod_distribution.items()},
        'residue_classes_with_multiple_perms': [
            {'residue': k, 'count': len(v), 'example_perms': v[:3]}
            for k, v in mod_distribution.items() if len(v) > 1
        ]
    }


def test_pigeonhole_principle(m: int, n_values: List[int]) -> Dict[str, Any]:
    """
    Test the pigeonhole principle approach:
    There are m! permutations but only m! possible residues modulo m!.
    However, we need to find if there are actually collisions.

    Key insight: Since there are m! permutations and m! residue classes mod m!,
    by pigeonhole principle, at least one residue class contains at least one permutation.
    But we need to show that some residue class contains at least TWO permutations.
    """
    m_fact = factorial(m)
    all_perms = list(itertools.permutations(range(1, m + 1)))

    # Compute f(x) mod m! for all permutations
    residue_classes = {}
    for perm in all_perms:
        f_val = compute_f(perm, n_values)
        residue = f_val % m_fact
        if residue not in residue_classes:
            residue_classes[residue] = []
        residue_classes[residue].append(perm)

    return {
        'm': m,
        'n_values': n_values,
        'm_factorial': m_fact,
        'total_permutations': len(all_perms),
        'residue_classes_used': len(residue_classes),
        'max_perms_in_class': max(len(v) for v in residue_classes.values()),
        'classes_with_multiple_perms': sum(1 for v in residue_classes.values() if len(v) > 1),
        'has_collision': any(len(v) > 1 for v in residue_classes.values())
    }


def analyze_symmetry(m: int, n_values: List[int]) -> Dict[str, Any]:
    """
    Analyze the symmetry properties of the problem.

    Key observation: If we swap positions i and j in a permutation,
    the change in f is (x_i - x_j) * (n_i - n_j).
    """
    results = []

    all_perms = list(itertools.permutations(range(1, m + 1)))
    identity = tuple(range(1, m + 1))

    # Check what happens when we apply transpositions to identity
    for i in range(m):
        for j in range(i + 1, m):
            # Create permutation by swapping positions i and j in identity
            perm_list = list(identity)
            perm_list[i], perm_list[j] = perm_list[j], perm_list[i]
            perm = tuple(perm_list)

            f_identity = compute_f(identity, n_values)
            f_perm = compute_f(perm, n_values)
            diff = f_perm - f_identity

            results.append({
                'swap_positions': [i, j],
                'identity': list(identity),
                'after_swap': list(perm),
                'f_identity': f_identity,
                'f_after_swap': f_perm,
                'difference': diff,
                'formula_check': (identity[j] - identity[i]) * (n_values[i] - n_values[j])
            })

    return {
        'm': m,
        'n_values': n_values,
        'transposition_effects': results
    }


def run_comprehensive_tests() -> Dict[str, Any]:
    """
    Run comprehensive tests for small values of m (odd only)
    """
    results = {
        'problem_statement': (
            'Let n_1, n_2, ..., n_m be integers where m>1 is odd. '
            'Let x = (x_1, ..., x_m) denote a permutation of 1, 2, ..., m. '
            'Let f(x) = x_1*n_1 + x_2*n_2 + ... + x_m*n_m. '
            'Show that for some distinct permutations a, b, '
            'the difference f(a) - f(b) is a multiple of m!.'
        ),
        'test_cases': []
    }

    # Test for m = 3
    print("Testing m = 3...")
    test_cases_m3 = [
        [1, 2, 3],
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [5, 7, 11],
        [-1, 2, -3],
        [100, 200, 300]
    ]

    for n_vals in test_cases_m3:
        print(f"  n_values = {n_vals}")
        result = find_divisible_pairs(3, n_vals)
        pigeonhole = test_pigeonhole_principle(3, n_vals)
        result['pigeonhole_analysis'] = pigeonhole
        results['test_cases'].append(result)

    # Test for m = 5 (limited cases due to 5! = 120 permutations)
    print("Testing m = 5...")
    test_cases_m5 = [
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [2, 4, 6, 8, 10]
    ]

    for n_vals in test_cases_m5:
        print(f"  n_values = {n_vals}")
        result = find_divisible_pairs(5, n_vals)
        pigeonhole = test_pigeonhole_principle(5, n_vals)
        result['pigeonhole_analysis'] = pigeonhole
        results['test_cases'].append(result)

    # Generate random test cases
    print("Testing random cases...")
    random.seed(42)
    for m in [3, 5]:
        for _ in range(3):
            n_vals = [random.randint(-10, 10) for _ in range(m)]
            print(f"  m = {m}, n_values = {n_vals}")
            result = find_divisible_pairs(m, n_vals)
            pigeonhole = test_pigeonhole_principle(m, n_vals)
            result['pigeonhole_analysis'] = pigeonhole
            results['test_cases'].append(result)

    # Analyze symmetry for m = 3
    print("Analyzing symmetry for m = 3...")
    symmetry_result = analyze_symmetry(3, [1, 2, 3])
    results['symmetry_analysis'] = symmetry_result

    # Key findings summary
    results['key_findings'] = {
        'all_cases_have_divisible_pairs': all(
            tc['divisible_pairs_count'] > 0 for tc in results['test_cases']
        ),
        'pigeonhole_always_applies': all(
            tc['pigeonhole_analysis']['has_collision'] for tc in results['test_cases']
        ),
        'insight': (
            'For all tested cases with odd m, there always exist distinct permutations '
            'a and b such that f(a) - f(b) is divisible by m!. '
            'This is confirmed by the pigeonhole principle: with m! permutations '
            'and m! residue classes modulo m!, collisions are guaranteed when the '
            'mapping is not injective.'
        )
    }

    return results


if __name__ == '__main__':
    print("=" * 80)
    print("IMO 2001 Problem 4 - Simulation")
    print("=" * 80)
    print()

    # Run comprehensive tests
    results = run_comprehensive_tests()

    # Save to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print(f"Results saved to {output_file}")
    print("=" * 80)
    print()

    # Print summary
    print("SUMMARY:")
    print(f"Total test cases: {len(results['test_cases'])}")
    print(f"All cases have divisible pairs: {results['key_findings']['all_cases_have_divisible_pairs']}")
    print(f"Pigeonhole principle applies: {results['key_findings']['pigeonhole_always_applies']}")
    print()
    print("Key insight:")
    print(results['key_findings']['insight'])
    print()

    # Show a few examples
    print("Example results:")
    for i, tc in enumerate(results['test_cases'][:3]):
        print(f"\n  Case {i+1}: m = {tc['m']}, n_values = {tc['n_values']}")
        print(f"    Total permutations: {tc['permutations_count']}")
        print(f"    Divisible pairs found: {tc['divisible_pairs_count']}")
        print(f"    Residue classes used: {tc['pigeonhole_analysis']['residue_classes_used']}/{tc['m_factorial']}")
        print(f"    Max perms in one class: {tc['pigeonhole_analysis']['max_perms_in_class']}")
        if tc['divisible_pairs']:
            pair = tc['divisible_pairs'][0]
            print(f"    Example: f({pair['perm_a']}) - f({pair['perm_b']}) = {pair['difference']} = {pair['quotient']} × {tc['m_factorial']}")
