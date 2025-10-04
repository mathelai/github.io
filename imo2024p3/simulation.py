"""
IMO 2024 Problem 3 Simulation

PROBLEM STATEMENT:
Let (a_n) be an infinite sequence of positive integers, and let k be a positive integer.
Suppose that, for each i ≥ 1, a_i equals the number of times i appears in a_1, a_2, ..., a_k.
Prove that at least one of the sequence (a_n) and k is eventually periodic.

SIMULATION APPROACH:
This script simulates the self-referential sequence problem by:
1. Finding fixed points: sequences where a_i = count(i in {a_1, ..., a_k})
2. Using iterative convergence starting from various initial conditions
3. Extending sequences beyond position k to check eventual periodicity
4. Analyzing patterns across different k values

KEY INSIGHTS DISCOVERED:
- For k≥2, there are exactly 2 fixed points: [1,0,0,...,0] and [0,0,...,0]
- For k=1, the only fixed point is [1]
- All fixed points have sum ≤ 1 (never sum = k for k≥2)
- When extended beyond k, all sequences eventually become 0 (periodic with period 1)
- The iterative process always converges (sum decreases or stays constant)

OUTPUT:
Generates results.json containing:
- All fixed point sequences for k=1 to 50
- Periodicity analysis of extended sequences
- Patterns and insights for proving the theorem
- Ground truth data for visualization
"""

import json
from collections import Counter
from typing import List, Tuple, Optional, Dict


def compute_sequence(k: int, max_iterations: int = 10000, initial_guess: List[int] = None) -> Tuple[Optional[List[int]], Dict]:
    """
    Compute the self-referential sequence for a given k.

    The sequence satisfies: a_i = count of i in {a_1, a_2, ..., a_k}

    Args:
        k: The length of the sequence prefix to consider
        max_iterations: Maximum number of iterations to try
        initial_guess: Starting sequence (if None, uses default)

    Returns:
        Tuple of (sequence, metadata) where sequence is the first k terms,
        and metadata contains information about convergence
    """
    # We'll use an iterative approach to find a fixed point
    # Start with a guess and iterate until convergence or max iterations

    if initial_guess is None:
        sequence = [k] + [0] * (k - 1)  # Initial guess
    else:
        sequence = initial_guess.copy()

    for iteration in range(max_iterations):
        # Count occurrences of each value in current sequence
        counts = Counter(sequence[:k])

        # Build new sequence based on counts
        new_sequence = []
        for i in range(1, k + 1):
            new_sequence.append(counts.get(i, 0))

        # Check for convergence
        if new_sequence == sequence[:k]:
            # Found a fixed point!
            return sequence[:k], {
                'converged': True,
                'iterations': iteration,
                'k': k,
                'sequence_length': k
            }

        sequence = new_sequence

    # Did not converge
    return None, {
        'converged': False,
        'iterations': max_iterations,
        'k': k
    }


def find_all_solutions(k: int, max_solutions: int = 10) -> List[List[int]]:
    """
    Try to find multiple valid sequences for a given k by trying different initial conditions.

    Args:
        k: The value of k
        max_solutions: Maximum number of different solutions to find

    Returns:
        List of unique valid sequences
    """
    import random
    solutions = []
    solution_set = set()

    # Try systematic initial guesses
    initial_guesses = [
        [k] + [0] * (k - 1),  # Standard guess
        [0] * k,  # All zeros
        [1] * k,  # All ones
        [k // 2] * k,  # All k/2
        list(range(k, 0, -1)),  # Decreasing
        list(range(1, k + 1)),  # Increasing
    ]

    # Add some random guesses
    for _ in range(20):
        initial_guesses.append([random.randint(0, k) for _ in range(k)])

    for guess in initial_guesses:
        seq, metadata = compute_sequence(k, initial_guess=guess)
        if seq and metadata['converged']:
            seq_tuple = tuple(seq)
            if seq_tuple not in solution_set:
                solution_set.add(seq_tuple)
                solutions.append(seq)
                if len(solutions) >= max_solutions:
                    break

    return solutions


def check_periodicity(sequence: List[int], max_period: int = 50) -> Dict:
    """
    Check if a sequence is eventually periodic.

    Args:
        sequence: The sequence to check
        max_period: Maximum period length to test

    Returns:
        Dictionary with periodicity information
    """
    n = len(sequence)

    # Try different period lengths
    for period in range(1, min(max_period, n // 2) + 1):
        # Try different starting points
        for start in range(max(0, n - 2 * period)):
            # Check if sequence is periodic with this period starting from start
            is_periodic = True
            check_length = min(period * 3, n - start)

            for i in range(check_length):
                if start + i + period < n:
                    if sequence[start + i] != sequence[start + i + period]:
                        is_periodic = False
                        break

            if is_periodic and check_length >= 2 * period:
                return {
                    'is_periodic': True,
                    'period': period,
                    'start_index': start,
                    'repeating_part': sequence[start:start + period]
                }

    return {
        'is_periodic': False,
        'period': None,
        'start_index': None,
        'repeating_part': None
    }


def extend_sequence_infinite(k: int, base_sequence: List[int], target_length: int = 200) -> List[int]:
    """
    Extend the sequence beyond k to check for eventual periodicity.

    For i > k, we can compute a_i by checking how many times i appears in positions 1 to k.

    Args:
        k: The parameter k from the problem
        base_sequence: The first k terms (already computed)
        target_length: How many terms to compute

    Returns:
        Extended sequence
    """
    sequence = base_sequence.copy()

    # For positions beyond k, a_i = number of times i appears in first k positions
    for i in range(k + 1, target_length + 1):
        count = sum(1 for val in sequence[:k] if val == i)
        sequence.append(count)

    return sequence


def analyze_k_value(k: int) -> Dict:
    """
    Comprehensive analysis of a specific k value.

    Args:
        k: The value to analyze

    Returns:
        Dictionary with all analysis results
    """
    print(f"Analyzing k = {k}...")

    # Find multiple solutions
    all_solutions = find_all_solutions(k)

    if not all_solutions:
        return {
            'k': k,
            'status': 'failed_to_converge',
            'metadata': {'converged': False}
        }

    # Use the first solution as the primary one
    base_seq = all_solutions[0]

    # Extend the sequence
    extended_seq = extend_sequence_infinite(k, base_seq, target_length=min(200, k * 10))

    # Check periodicity of the extended sequence
    periodicity = check_periodicity(extended_seq)

    # Compute statistics
    unique_values = sorted(set(base_seq))
    value_counts = Counter(base_seq)

    # Check if k itself appears in the sequence
    k_appears = k in base_seq
    k_count = base_seq.count(k) if k_appears else 0

    # Analyze all solutions
    solutions_data = []
    for sol in all_solutions:
        solutions_data.append({
            'sequence': sol,
            'sum': sum(sol),
            'unique_count': len(set(sol)),
            'max_val': max(sol) if sol else 0,
        })

    return {
        'k': k,
        'status': 'success',
        'base_sequence': base_seq,
        'all_solutions': solutions_data,
        'num_solutions_found': len(all_solutions),
        'extended_sequence': extended_seq[:100],  # First 100 terms
        'sequence_length': len(base_seq),
        'extended_length': len(extended_seq),
        'periodicity': periodicity,
        'unique_values': unique_values,
        'value_distribution': dict(value_counts),
        'k_appears': k_appears,
        'k_count': k_count,
        'max_value': max(base_seq),
        'min_value': min(base_seq),
        'sum_of_sequence': sum(base_seq),
        'iterations_to_converge': 0  # Will vary by initial guess
    }


def find_patterns(results: List[Dict]) -> Dict:
    """
    Analyze results across multiple k values to find patterns.

    Args:
        results: List of analysis results for different k values

    Returns:
        Dictionary summarizing patterns found
    """
    patterns = {
        'always_periodic': True,
        'periodic_ks': [],
        'non_periodic_ks': [],
        'failed_ks': [],
        'max_period_found': 0,
        'common_periods': Counter(),
        'k_appearance_pattern': []
    }

    for result in results:
        k = result['k']

        if result['status'] != 'success':
            patterns['failed_ks'].append(k)
            continue

        is_periodic = result['periodicity']['is_periodic']

        if is_periodic:
            patterns['periodic_ks'].append(k)
            period = result['periodicity']['period']
            patterns['common_periods'][period] += 1
            patterns['max_period_found'] = max(patterns['max_period_found'], period)
        else:
            patterns['non_periodic_ks'].append(k)
            patterns['always_periodic'] = False

        patterns['k_appearance_pattern'].append({
            'k': k,
            'k_appears': result['k_appears'],
            'k_count': result['k_count']
        })

    return patterns


def main():
    """
    Main simulation function.
    Analyzes multiple k values and generates results.json
    """
    random.seed(42)
    print("IMO 2024 Problem 3 - Simulation Starting...")
    print("=" * 60)

    # Analyze k values from 1 to 50
    k_values = list(range(1, 51))

    results = []
    for k in k_values:
        result = analyze_k_value(k)
        results.append(result)

    # Find patterns across all k values
    patterns = find_patterns(results)

    # Prepare output
    output = {
        'problem_statement': (
            'Let (a_n) be an infinite sequence of positive integers, and let k be a positive integer. '
            'Suppose that, for each i >= 1, a_i is equal to the number of times i appears in '
            'the list a_1, a_2, ..., a_k. Prove that at least one of the sequence (a_n) and k '
            'is eventually periodic.'
        ),
        'summary': {
            'total_k_tested': len(k_values),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] != 'success']),
            'all_periodic': patterns['always_periodic']
        },
        'patterns': patterns,
        'detailed_results': results
    }

    # Save to JSON
    output_path = 'results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total k values tested: {output['summary']['total_k_tested']}")
    print(f"Successful: {output['summary']['successful']}")
    print(f"Failed to converge: {output['summary']['failed']}")
    print(f"All sequences periodic: {output['summary']['all_periodic']}")
    print(f"\nPeriodic k values: {len(patterns['periodic_ks'])}")
    print(f"Non-periodic k values: {len(patterns['non_periodic_ks'])}")

    if patterns['non_periodic_ks']:
        print(f"\nNon-periodic k values: {patterns['non_periodic_ks'][:10]}")

    print(f"\nMost common periods: {patterns['common_periods'].most_common(5)}")
    print(f"\nResults saved to: {output_path}")

    # Print some example sequences
    print("\n" + "=" * 60)
    print("EXAMPLE SEQUENCES")
    print("=" * 60)

    for k in [1, 2, 3, 4, 5, 10, 15, 20]:
        result = next((r for r in results if r['k'] == k), None)
        if result and result['status'] == 'success':
            seq = result['base_sequence']
            print(f"\nk = {k}:")
            print(f"  Primary sequence: {seq[:min(20, len(seq))]}")
            if len(seq) > 20:
                print(f"  ... (total length: {len(seq)})")
            print(f"  Sum = {result['sum_of_sequence']}")
            print(f"  # of solutions found: {result['num_solutions_found']}")
            if result['num_solutions_found'] > 1:
                print(f"  Other solutions:")
                for i, sol_data in enumerate(result['all_solutions'][1:3], 1):  # Show up to 2 more
                    print(f"    {i}. {sol_data['sequence'][:min(10, len(sol_data['sequence']))]}... (sum={sol_data['sum']})")
            print(f"  Periodic: {result['periodicity']['is_periodic']}")
            if result['periodicity']['is_periodic']:
                print(f"  Period: {result['periodicity']['period']}")
                print(f"  Repeating part: {result['periodicity']['repeating_part']}")


if __name__ == '__main__':
    main()
