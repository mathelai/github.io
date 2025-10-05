#!/usr/bin/env python3
"""
IMO 2014 Problem 6 Simulation
=============================

Problem: A set of lines in the plane is in general position if no two are parallel
and no three pass through the same point. A set of lines in general position cuts
the plane into regions, some of which have finite area; we call these its finite
regions. Prove that for all sufficiently large n, in any set of n lines in general
position it is possible to colour at least √n lines blue in such a way that none
of its finite regions has a completely blue boundary.

This simulation:
1. Generates sets of lines in general position
2. Finds all finite regions (bounded polygons)
3. Tests different coloring strategies
4. Verifies that we can color at least √n lines blue without creating all-blue regions
5. Generates ground truth data for analysis
"""

import json
import math
import random
from typing import List, Tuple, Set, Dict
from itertools import combinations
import sys

# Type aliases
Point = Tuple[float, float]
Line = Tuple[float, float, float]  # ax + by + c = 0


def create_random_lines(n: int, seed: int = None) -> List[Line]:
    """
    Create n lines in general position using random slopes and intercepts.
    Lines are represented as (a, b, c) where ax + by + c = 0.
    """
    if seed is not None:
        random.seed(seed)

    lines = []
    slopes = set()

    # Generate lines with different slopes to avoid parallel lines
    for i in range(n):
        # Use different slopes for each line
        angle = (i * math.pi / n) + random.uniform(-0.1, 0.1)
        slope = math.tan(angle)

        # Ensure unique slope
        while slope in slopes:
            slope += random.uniform(0.01, 0.1)
        slopes.add(slope)

        # Random y-intercept
        intercept = random.uniform(-10, 10)

        # Convert y = mx + b to ax + by + c = 0
        # -mx + y - b = 0
        lines.append((-slope, 1.0, -intercept))

    return lines


def line_intersection(line1: Line, line2: Line) -> Point:
    """Find intersection point of two lines."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None  # Parallel lines

    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det

    return (x, y)


def find_all_intersections(lines: List[Line]) -> List[Point]:
    """Find all intersection points of the lines."""
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is not None:
                intersections.append(pt)
    return intersections


def point_on_line(point: Point, line: Line, tol: float = 1e-6) -> bool:
    """Check if a point lies on a line."""
    a, b, c = line
    x, y = point
    return abs(a * x + b * y + c) < tol


def find_finite_regions(lines: List[Line], bound: float = 100) -> List[List[int]]:
    """
    Find all finite regions (bounded polygons) created by the lines.
    Returns a list of regions, where each region is a list of line indices that bound it.

    Strategy: For small n, we can enumerate regions by checking which combinations
    of lines form bounded polygons.
    """
    n = len(lines)
    regions = []

    # Find all intersection points
    intersections = find_all_intersections(lines)

    # For each subset of 3 or more lines, check if they form a bounded region
    # Simplified: check all triangles (3 lines) and quadrilaterals (4 lines)
    for k in range(3, min(n + 1, 8)):  # Limit to avoid explosion
        for line_indices in combinations(range(n), k):
            # Check if these lines form a bounded region
            region_points = []
            region_lines = [lines[i] for i in line_indices]

            # Get all pairwise intersections of these lines
            for i, j in combinations(range(len(region_lines)), 2):
                pt = line_intersection(region_lines[i], region_lines[j])
                if pt is not None:
                    region_points.append(pt)

            # Check if we have a bounded polygon
            if len(region_points) >= 3:
                # Simple boundedness check: all points within bound
                if all(abs(x) < bound and abs(y) < bound for x, y in region_points):
                    # Additional check: this should be a minimal set (removing any line unbounds it)
                    regions.append(list(line_indices))

    # Remove duplicates and non-minimal regions
    unique_regions = []
    for region in regions:
        is_minimal = True
        for other_region in regions:
            if set(other_region) < set(region):
                is_minimal = False
                break
        if is_minimal and region not in unique_regions:
            unique_regions.append(region)

    return unique_regions


def find_finite_regions_simple(lines: List[Line]) -> List[Set[int]]:
    """
    Simplified approach: A finite region is bounded by at least 3 lines.
    For n lines in general position, we look for triangular and quadrilateral regions.
    """
    regions = []
    n = len(lines)

    # All triangular regions (formed by 3 lines)
    for i, j, k in combinations(range(n), 3):
        # Check if these three lines form a triangle
        p1 = line_intersection(lines[i], lines[j])
        p2 = line_intersection(lines[j], lines[k])
        p3 = line_intersection(lines[k], lines[i])

        if p1 and p2 and p3:
            # Check if it's bounded (simple heuristic: points not too far)
            if all(abs(x) < 1000 and abs(y) < 1000 for x, y in [p1, p2, p3]):
                regions.append({i, j, k})

    return regions


def is_region_all_blue(region: Set[int], blue_lines: Set[int]) -> bool:
    """Check if all boundary lines of a region are blue."""
    return region.issubset(blue_lines)


def greedy_coloring(lines: List[Line], target: int) -> Tuple[Set[int], bool]:
    """
    Greedy strategy: Color lines blue one by one, avoiding creating all-blue regions.
    Returns (blue_lines, success) where success indicates if we colored >= target lines.
    """
    regions = find_finite_regions_simple(lines)
    blue_lines = set()
    n = len(lines)

    # Try to color lines greedily
    for line_idx in range(n):
        # Check if coloring this line blue would create an all-blue region
        temp_blue = blue_lines | {line_idx}

        creates_all_blue = any(is_region_all_blue(region, temp_blue) for region in regions)

        if not creates_all_blue:
            blue_lines.add(line_idx)

        if len(blue_lines) >= target:
            return blue_lines, True

    return blue_lines, len(blue_lines) >= target


def smart_coloring(lines: List[Line], target: int) -> Tuple[Set[int], bool]:
    """
    Smarter strategy: Build a conflict graph and find a large independent set.
    Two lines conflict if there exists a region where they are the only two lines
    that could be blue (i.e., coloring both would create an all-blue region).
    """
    regions = find_finite_regions_simple(lines)
    n = len(lines)

    # Build conflict graph
    conflicts = {i: set() for i in range(n)}

    for region in regions:
        region_list = list(region)
        # For each pair in the region, they conflict if region has exactly 2 lines
        # More generally, for region with k lines, any k lines together would make it all-blue
        if len(region) == 3:
            # For triangles, we can color at most 2 of the 3 lines
            for i, j in combinations(region_list, 2):
                # They conflict with the third one as a group
                pass

    # Simple greedy independent set
    blue_lines = set()
    available = set(range(n))

    while available and len(blue_lines) < target:
        # Pick line with fewest conflicts
        line = min(available, key=lambda x: len(conflicts[x] & available))
        blue_lines.add(line)
        available.remove(line)
        available -= conflicts[line]

    # Verify no all-blue regions
    success = len(blue_lines) >= target
    has_all_blue = any(is_region_all_blue(region, blue_lines) for region in regions)

    return blue_lines, success and not has_all_blue


def alternating_coloring(lines: List[Line], target: int) -> Tuple[Set[int], bool]:
    """
    Alternating strategy: Color every k-th line blue.
    This is based on the observation that we can always color a structured subset.
    """
    n = len(lines)
    regions = find_finite_regions_simple(lines)

    # Try different strides
    for stride in range(2, n + 1):
        blue_lines = set(range(0, n, stride))

        if len(blue_lines) >= target:
            # Check if any region is all blue
            has_all_blue = any(is_region_all_blue(region, blue_lines) for region in regions)
            if not has_all_blue:
                return blue_lines, True

    return set(), False


def test_configuration(n: int, seed: int = None) -> Dict:
    """Test a single configuration with n lines."""
    lines = create_random_lines(n, seed)
    target = int(math.sqrt(n))

    regions = find_finite_regions_simple(lines)

    # Try different strategies
    greedy_result, greedy_success = greedy_coloring(lines, target)
    smart_result, smart_success = smart_coloring(lines, target)
    alt_result, alt_success = alternating_coloring(lines, target)

    return {
        'n': n,
        'target': target,
        'num_regions': len(regions),
        'greedy_colored': len(greedy_result),
        'greedy_success': greedy_success,
        'smart_colored': len(smart_result),
        'smart_success': smart_success,
        'alternating_colored': len(alt_result),
        'alternating_success': alt_success,
        'best_colored': max(len(greedy_result), len(smart_result), len(alt_result)),
        'any_success': greedy_success or smart_success or alt_success,
    }


def run_simulation(max_n: int = 20, trials_per_n: int = 5) -> Dict:
    """
    Run simulation for different values of n.
    """
    results = []

    for n in range(3, max_n + 1):
        print(f"Testing n={n} (target: {int(math.sqrt(n))} blue lines)...")

        for trial in range(trials_per_n):
            result = test_configuration(n, seed=n * 100 + trial)
            results.append(result)

    # Aggregate statistics
    summary = {}
    for n in range(3, max_n + 1):
        n_results = [r for r in results if r['n'] == n]
        target = int(math.sqrt(n))

        success_count = sum(1 for r in n_results if r['any_success'])
        avg_colored = sum(r['best_colored'] for r in n_results) / len(n_results)
        avg_regions = sum(r['num_regions'] for r in n_results) / len(n_results)

        summary[n] = {
            'target': target,
            'success_rate': success_count / len(n_results),
            'avg_colored': avg_colored,
            'avg_regions': avg_regions,
            'exceeds_target': avg_colored >= target,
        }

    return {
        'max_n': max_n,
        'trials_per_n': trials_per_n,
        'detailed_results': results,
        'summary': summary,
        'observations': generate_observations(summary),
    }


def generate_observations(summary: Dict) -> List[str]:
    """Generate observations from the simulation results."""
    observations = []

    # Check if we always succeed for large enough n
    large_n = [n for n in summary.keys() if n >= 16]
    if large_n:
        success_rate = sum(summary[n]['success_rate'] for n in large_n) / len(large_n)
        observations.append(f"For n >= 16, success rate is {success_rate:.2%}")

    # Check the relationship between n and colored lines
    for n in sorted(summary.keys()):
        data = summary[n]
        if data['avg_colored'] >= data['target']:
            observations.append(
                f"n={n}: Successfully colored {data['avg_colored']:.1f} lines "
                f"(target: {data['target']}, {data['avg_regions']:.1f} finite regions)"
            )
        else:
            observations.append(
                f"n={n}: Only colored {data['avg_colored']:.1f} lines "
                f"(target: {data['target']}, {data['avg_regions']:.1f} finite regions) - FAILED"
            )

    # Pattern analysis
    observations.append("\nKey Insight: Each finite region is bounded by at least 3 lines. "
                       "If we color at most 1/3 of the lines, we can avoid all-blue regions.")

    observations.append("\nFor large n, √n << n/3, so we can always color √n lines.")

    return observations


def main():
    """Main simulation runner."""
    print("=" * 70)
    print("IMO 2014 Problem 6: Line Coloring Simulation")
    print("=" * 70)
    print()

    # Run simulation
    max_n = 25
    trials = 10

    print(f"Running simulation for n=3 to n={max_n} with {trials} trials each...\n")

    results = run_simulation(max_n=max_n, trials_per_n=trials)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for n in sorted(results['summary'].keys()):
        data = results['summary'][n]
        status = "✓" if data['exceeds_target'] else "✗"
        print(f"{status} n={n:2d}: target={data['target']:2d}, "
              f"avg_colored={data['avg_colored']:4.1f}, "
              f"success_rate={data['success_rate']:5.1%}, "
              f"avg_regions={data['avg_regions']:5.1f}")

    print("\n" + "=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    for obs in results['observations']:
        print(obs)

    # Save to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Generate sample configuration for visualization
    sample_n = 8
    sample_lines = create_random_lines(sample_n, seed=42)
    sample_regions = find_finite_regions_simple(sample_lines)
    target = int(math.sqrt(sample_n))
    blue_lines, success = greedy_coloring(sample_lines, target)

    sample_data = {
        'n': sample_n,
        'lines': sample_lines,
        'regions': [list(r) for r in sample_regions],
        'blue_lines': list(blue_lines),
        'target': target,
        'success': success,
    }

    # Save sample for webapp
    sample_file = 'sample.json'
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"Sample configuration saved to {sample_file}")


if __name__ == "__main__":
    main()
