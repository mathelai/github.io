#!/usr/bin/env python3
"""
IMO 2015 Problem 1 Simulation

Definitions:
- Balanced: For any two different points A, B in S, there exists C in S such that AC = BC
- Centre-free: For any three points A, B, C in S, there is no point P in S such that PA = PB = PC

Goal:
- Show that for all n >= 3, there exists a balanced set of n points
- Determine all n >= 3 for which there exists a balanced centre-free set
"""

import numpy as np
import itertools
from typing import List, Tuple, Set
import json

Point = Tuple[float, float]

def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_balanced(points: List[Point], epsilon: float = 1e-9) -> bool:
    """
    Check if a set of points is balanced.
    For any two different points A, B, there must exist C such that AC = BC.
    """
    n = len(points)
    if n < 3:
        return False

    for i in range(n):
        for j in range(i + 1, n):
            A, B = points[i], points[j]
            # Check if there exists C such that AC = BC
            found = False
            for k in range(n):
                if k != i and k != j:
                    C = points[k]
                    dist_AC = distance(A, C)
                    dist_BC = distance(B, C)
                    if abs(dist_AC - dist_BC) < epsilon:
                        found = True
                        break
            if not found:
                return False
    return True

def is_centre_free(points: List[Point], epsilon: float = 1e-9) -> bool:
    """
    Check if a set of points is centre-free.
    For any three points A, B, C, there should be no point P such that PA = PB = PC.
    """
    n = len(points)
    if n < 4:
        return True

    # Check all combinations of 3 points
    for combo in itertools.combinations(range(n), 3):
        i, j, k = combo
        A, B, C = points[i], points[j], points[k]

        # Check if any other point P is equidistant to A, B, C
        for m in range(n):
            if m not in combo:
                P = points[m]
                dist_PA = distance(P, A)
                dist_PB = distance(P, B)
                dist_PC = distance(P, C)

                if abs(dist_PA - dist_PB) < epsilon and abs(dist_PB - dist_PC) < epsilon:
                    return False
    return True

def generate_regular_polygon(n: int, radius: float = 1.0) -> List[Point]:
    """Generate n points arranged in a regular polygon"""
    points = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append((x, y))
    return points

def generate_line_points(n: int) -> List[Point]:
    """Generate n points on a line"""
    return [(float(i), 0.0) for i in range(n)]

def generate_isosceles_triangle_with_points(n: int) -> List[Point]:
    """Generate points based on isosceles triangle pattern"""
    if n < 3:
        return []

    # Start with equilateral triangle
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, np.sqrt(3)/2)
    ]

    # Add points on perpendicular bisector for odd n
    if n > 3:
        for i in range(n - 3):
            y_offset = np.sqrt(3)/2 + (i + 1) * 0.5
            points.append((0.5, y_offset))

    return points[:n]

def test_configuration(name: str, points: List[Point]) -> dict:
    """Test a configuration and return results"""
    n = len(points)
    balanced = is_balanced(points)
    centre_free = is_centre_free(points)

    return {
        'name': name,
        'n': n,
        'balanced': balanced,
        'centre_free': centre_free,
        'both': balanced and centre_free,
        'points': [(round(p[0], 6), round(p[1], 6)) for p in points]
    }

def main():
    """Run simulation on various configurations"""
    results = []

    print("=" * 80)
    print("IMO 2015 Problem 1 Simulation")
    print("=" * 80)

    # Test regular polygons for n = 3 to 12
    print("\n### Regular Polygons ###")
    for n in range(3, 13):
        points = generate_regular_polygon(n)
        result = test_configuration(f"Regular {n}-gon", points)
        results.append(result)

        status = []
        if result['balanced']:
            status.append("BALANCED")
        if result['centre_free']:
            status.append("CENTRE-FREE")

        print(f"n={n:2d}: {', '.join(status) if status else 'NEITHER'}")

    # Test line configurations
    print("\n### Points on a Line ###")
    for n in range(3, 8):
        points = generate_line_points(n)
        result = test_configuration(f"Line {n} points", points)
        results.append(result)

        status = []
        if result['balanced']:
            status.append("BALANCED")
        if result['centre_free']:
            status.append("CENTRE-FREE")

        print(f"n={n:2d}: {', '.join(status) if status else 'NEITHER'}")

    # Test triangle-based configurations
    print("\n### Triangle-based Configurations ###")
    for n in range(3, 10):
        points = generate_isosceles_triangle_with_points(n)
        result = test_configuration(f"Triangle-based {n} points", points)
        results.append(result)

        status = []
        if result['balanced']:
            status.append("BALANCED")
        if result['centre_free']:
            status.append("CENTRE-FREE")

        print(f"n={n:2d}: {', '.join(status) if status else 'NEITHER'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    balanced_centre_free = {}
    for result in results:
        n = result['n']
        if result['both']:
            if n not in balanced_centre_free:
                balanced_centre_free[n] = []
            balanced_centre_free[n].append(result['name'])

    print("\nBalanced AND Centre-free configurations found:")
    for n in sorted(balanced_centre_free.keys()):
        print(f"  n={n}: {balanced_centre_free[n]}")

    # Check pattern
    bcf_values = sorted(balanced_centre_free.keys())
    if bcf_values:
        print(f"\nValues of n with balanced centre-free sets: {bcf_values}")
        if all(n % 2 == 1 for n in bcf_values):
            print("Pattern: All ODD integers >= 3 ✓")
        else:
            print("Pattern: Mixed odd and even")

    # Save results to JSON
    with open('simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to simulation_results.json")

if __name__ == "__main__":
    main()
