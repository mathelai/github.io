#!/usr/bin/env python3
"""
IMO 2019 Problem 5 Simulation
Bank of Bath coin flipping problem
"""

from typing import List, Tuple
import itertools
from collections import defaultdict


def config_to_string(config: List[int]) -> str:
    """Convert configuration to string (0=B, 1=A)"""
    return ''.join('A' if c else 'B' for c in config)


def count_A(config: List[int]) -> int:
    """Count number of A's (1's) in configuration"""
    return sum(config)


def step(config: List[int]) -> Tuple[List[int], bool]:
    """
    Perform one operation.
    Returns: (new_config, continues)
    - continues=True if operation was performed
    - continues=False if all coins show B and we stop
    """
    k = count_A(config)

    if k == 0:
        return config, False  # All B's, stop

    # Turn over the k-th coin (1-indexed, so index k-1)
    new_config = config.copy()
    if k <= len(config):
        new_config[k-1] = 1 - new_config[k-1]  # Flip the coin

    return new_config, True


def simulate(initial_config: List[int], max_steps: int = 10000) -> Tuple[int, List[List[int]]]:
    """
    Simulate the process from an initial configuration.
    Returns: (number of steps, trajectory of configurations)
    """
    config = initial_config.copy()
    trajectory = [config.copy()]
    steps = 0

    while steps < max_steps:
        config, continues = step(config)
        trajectory.append(config.copy())
        steps += 1

        if not continues:
            break

    return steps, trajectory


def f(config: List[int]) -> int:
    """
    Compute f(C): number of operations before Harry stops.
    """
    steps, _ = simulate(config)
    return steps


def generate_all_configs(n: int) -> List[List[int]]:
    """Generate all 2^n possible configurations for n coins"""
    return [list(config) for config in itertools.product([0, 1], repeat=n)]


def analyze_all_configs(n: int) -> dict:
    """
    Analyze all configurations for n coins.
    Returns statistics and ground truth data.
    """
    configs = generate_all_configs(n)
    results = []

    for config in configs:
        steps, trajectory = simulate(config)
        results.append({
            'config': config_to_string(config),
            'steps': steps,
            'trajectory': [config_to_string(c) for c in trajectory]
        })

    # Calculate average
    total_steps = sum(r['steps'] for r in results)
    average = total_steps / len(results)

    # Distribution of steps
    step_distribution = defaultdict(int)
    for r in results:
        step_distribution[r['steps']] += 1

    return {
        'n': n,
        'num_configs': len(results),
        'results': results,
        'total_steps': total_steps,
        'average_steps': average,
        'step_distribution': dict(step_distribution)
    }


def find_interesting_examples(n: int):
    """Find interesting examples: max steps, min steps, etc."""
    configs = generate_all_configs(n)

    max_steps = 0
    max_config = None
    max_trajectory = None

    interesting = []

    for config in configs:
        steps, trajectory = simulate(config)

        if steps > max_steps:
            max_steps = steps
            max_config = config
            max_trajectory = trajectory

        # Collect some interesting cases
        if steps >= 5 or config == [1]*n or config == [0]*n:
            interesting.append({
                'config': config_to_string(config),
                'steps': steps,
                'trajectory': [config_to_string(c) for c in trajectory]
            })

    return {
        'max_steps': max_steps,
        'max_config': config_to_string(max_config),
        'max_trajectory': [config_to_string(c) for c in max_trajectory],
        'interesting_examples': interesting
    }


def print_analysis(n: int):
    """Print detailed analysis for n coins"""
    print(f"\n{'='*60}")
    print(f"Analysis for n = {n}")
    print(f"{'='*60}")

    analysis = analyze_all_configs(n)

    print(f"\nTotal configurations: {analysis['num_configs']}")
    print(f"Total steps across all configs: {analysis['total_steps']}")
    print(f"Average f(C): {analysis['average_steps']:.6f}")

    print(f"\nStep distribution:")
    for steps in sorted(analysis['step_distribution'].keys()):
        count = analysis['step_distribution'][steps]
        print(f"  {steps} steps: {count} configurations")

    interesting = find_interesting_examples(n)
    print(f"\nMaximum steps: {interesting['max_steps']}")
    print(f"Configuration with max steps: {interesting['max_config']}")
    print(f"Trajectory: {' -> '.join(interesting['max_trajectory'])}")

    print(f"\nSample configurations:")
    for ex in interesting['interesting_examples'][:10]:
        print(f"  {ex['config']}: {ex['steps']} steps")
        if ex['steps'] <= 5:
            print(f"    {' -> '.join(ex['trajectory'])}")

    return analysis


def verify_example():
    """Verify the given example: AAB -> ABB -> BBB (3 steps)"""
    config = [1, 1, 0]  # AAB
    steps, trajectory = simulate(config)

    print("\nVerifying given example:")
    print(f"Initial: AAB")
    print(f"Trajectory: {' -> '.join(config_to_string(c) for c in trajectory)}")
    print(f"Steps: {steps}")
    print(f"Expected: 3, Got: {steps} ✓" if steps == 3 else f"Expected: 3, Got: {steps} ✗")


if __name__ == "__main__":
    # Verify the example
    verify_example()

    # Analyze small instances
    for n in range(1, 7):
        analysis = print_analysis(n)

    # Look for pattern in averages
    print(f"\n{'='*60}")
    print("Summary of Averages")
    print(f"{'='*60}")
    for n in range(1, 7):
        analysis = analyze_all_configs(n)
        print(f"n={n}: Average f(C) = {analysis['average_steps']:.6f} = {analysis['total_steps']}/{analysis['num_configs']}")
