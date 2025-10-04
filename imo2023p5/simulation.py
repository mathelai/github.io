"""
IMO 2023 Problem 5: Japanese Triangle Simulation

This simulation explores Japanese triangles to find the greatest k such that
every Japanese triangle of size n has a ninja path containing at least k red circles.

A Japanese triangle has:
- n rows where row i has i circles
- Exactly one red circle per row
- A ninja path starts at row 1, moves to one of two circles below, ends at row n

The goal is to find the minimum over all possible triangles of the maximum red circles
that can be collected in any ninja path.
"""

import json
import itertools
from typing import List, Tuple, Set
from collections import defaultdict


class JapaneseTriangle:
    """Represents a Japanese triangle with n rows and one red circle per row."""

    def __init__(self, n: int, red_positions: List[int]):
        """
        Initialize a Japanese triangle.

        Args:
            n: Number of rows
            red_positions: List of length n where red_positions[i] is the 0-indexed
                          position of the red circle in row i (0 <= red_positions[i] < i+1)
        """
        self.n = n
        self.red_positions = red_positions

        # Validate red positions
        for i, pos in enumerate(red_positions):
            if not (0 <= pos <= i):
                raise ValueError(f"Row {i} has invalid red position {pos}")

    def is_red(self, row: int, col: int) -> bool:
        """Check if circle at (row, col) is red."""
        return self.red_positions[row] == col

    def get_all_paths(self) -> List[List[Tuple[int, int]]]:
        """Generate all possible ninja paths through the triangle."""
        paths = []

        def generate_paths(row: int, col: int, current_path: List[Tuple[int, int]]):
            current_path.append((row, col))

            if row == self.n - 1:
                # Reached bottom row
                paths.append(current_path.copy())
            else:
                # Continue to one of two circles below
                # From (row, col) we can go to (row+1, col) or (row+1, col+1)
                generate_paths(row + 1, col, current_path)
                generate_paths(row + 1, col + 1, current_path)

            current_path.pop()

        # Start from the single circle in row 0
        generate_paths(0, 0, [])
        return paths

    def count_red_in_path(self, path: List[Tuple[int, int]]) -> int:
        """Count red circles in a path."""
        return sum(1 for row, col in path if self.is_red(row, col))

    def max_red_in_any_path(self) -> int:
        """Find the maximum number of red circles in any ninja path."""
        paths = self.get_all_paths()
        return max(self.count_red_in_path(path) for path in paths)

    def get_best_path(self) -> Tuple[List[Tuple[int, int]], int]:
        """Get the path with maximum red circles and the count."""
        paths = self.get_all_paths()
        best_path = max(paths, key=lambda p: self.count_red_in_path(p))
        return best_path, self.count_red_in_path(best_path)

    def to_dict(self) -> dict:
        """Convert triangle to dictionary for JSON serialization."""
        return {
            'n': self.n,
            'red_positions': self.red_positions,
            'max_red': self.max_red_in_any_path()
        }


def generate_all_triangles(n: int) -> List[JapaneseTriangle]:
    """
    Generate all possible Japanese triangles of size n.

    For each row i (0-indexed), there are i+1 possible positions for the red circle.
    Total number of triangles: 1 * 2 * 3 * ... * n = n!
    """
    triangles = []

    # Generate all combinations of red positions
    # Row i has positions 0, 1, ..., i
    ranges = [range(i + 1) for i in range(n)]

    for red_positions in itertools.product(*ranges):
        triangles.append(JapaneseTriangle(n, list(red_positions)))

    return triangles


def find_min_max_red(n: int) -> Tuple[int, JapaneseTriangle]:
    """
    Find the minimum over all triangles of the maximum red circles in any path.

    This is the answer to the problem: the greatest k such that every triangle
    has a path with at least k red circles.

    Returns:
        (k, worst_triangle): The minimum max_red value and an example triangle achieving it
    """
    triangles = generate_all_triangles(n)

    min_max_red = float('inf')
    worst_triangle = None

    for triangle in triangles:
        max_red = triangle.max_red_in_any_path()
        if max_red < min_max_red:
            min_max_red = max_red
            worst_triangle = triangle

    return min_max_red, worst_triangle


def analyze_pattern(max_n: int = 6) -> dict:
    """
    Analyze the pattern for different values of n.

    Returns a dictionary with results for each n.
    """
    results = {
        'analysis': [],
        'pattern': [],
        'worst_case_examples': []
    }

    print("Analyzing Japanese Triangles")
    print("=" * 60)

    for n in range(1, max_n + 1):
        print(f"\nAnalyzing n = {n}...")

        k, worst_triangle = find_min_max_red(n)
        total_triangles = 1
        for i in range(1, n + 1):
            total_triangles *= i

        print(f"  Total triangles: {total_triangles}")
        print(f"  Minimum max_red (k): {k}")
        print(f"  Worst case red positions: {worst_triangle.red_positions}")

        best_path, red_count = worst_triangle.get_best_path()
        print(f"  Best path in worst triangle: {best_path}")
        print(f"  Red circles in best path: {red_count}")

        # Store results
        results['analysis'].append({
            'n': n,
            'k': k,
            'total_triangles': total_triangles,
            'worst_red_positions': worst_triangle.red_positions,
            'best_path': best_path,
            'red_in_best_path': red_count
        })

        results['pattern'].append({'n': n, 'k': k})

        # Store detailed worst case example
        results['worst_case_examples'].append({
            'n': n,
            'red_positions': worst_triangle.red_positions,
            'max_red': k
        })

    # Analyze the pattern
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS:")
    print("=" * 60)
    pattern_data = results['pattern']

    for item in pattern_data:
        n, k = item['n'], item['k']
        print(f"n = {n}: k = {k}")

    # Check if k follows a pattern (hypothesis: k = ceil(log2(n+1)))
    print("\nHypothesis: k = ⌈log₂(n+1)⌉")
    import math
    for item in pattern_data:
        n, k = item['n'], item['k']
        expected = math.ceil(math.log2(n + 1))
        match = "✓" if k == expected else "✗"
        print(f"  n = {n}: k = {k}, expected = {expected} {match}")

    return results


def visualize_triangle(triangle: JapaneseTriangle):
    """Print a visual representation of a triangle."""
    print("\nTriangle visualization:")
    for row in range(triangle.n):
        # Add spacing for centering
        spaces = " " * (triangle.n - row - 1)
        circles = []
        for col in range(row + 1):
            if triangle.is_red(row, col):
                circles.append("R")
            else:
                circles.append("O")
        print(spaces + " ".join(circles))


def run_simulation():
    """Main simulation runner."""
    # Analyze for small values of n
    results = analyze_pattern(max_n=6)

    # Add some specific examples for visualization
    examples = []

    # Example 1: Small triangle (n=3)
    t1 = JapaneseTriangle(3, [0, 1, 0])
    path1, red1 = t1.get_best_path()
    examples.append({
        'description': 'Example triangle with n=3',
        'n': 3,
        'red_positions': [0, 1, 0],
        'best_path': path1,
        'red_in_path': red1
    })

    # Example 2: Worst case for n=4
    _, worst4 = find_min_max_red(4)
    path4, red4 = worst4.get_best_path()
    examples.append({
        'description': 'Worst case triangle for n=4',
        'n': 4,
        'red_positions': worst4.red_positions,
        'best_path': path4,
        'red_in_path': red4
    })

    # Example 3: Worst case for n=5
    _, worst5 = find_min_max_red(5)
    path5, red5 = worst5.get_best_path()
    examples.append({
        'description': 'Worst case triangle for n=5',
        'n': 5,
        'red_positions': worst5.red_positions,
        'best_path': path5,
        'red_in_path': red5
    })

    results['examples'] = examples

    # Mathematical insight
    import math
    results['conjecture'] = {
        'formula': 'k = ceil(log2(n+1))',
        'explanation': (
            'The answer is k = ⌈log₂(n+1)⌉. '
            'This can be proven by showing: '
            '(1) There exists a triangle where the maximum path has exactly ⌈log₂(n+1)⌉ red circles (lower bound), '
            'and (2) Every triangle has a path with at least ⌈log₂(n+1)⌉ red circles (upper bound). '
            'The worst case occurs when red circles are positioned to minimize overlap with any single path. '
            'The logarithmic pattern emerges because each path through the triangle makes n binary choices (left or right at each level), '
            'and by the pigeonhole principle, some path must collect at least log₂(n+1) red circles.'
        )
    }

    # Save results to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")
    print("\nConclusion:")
    print(f"The greatest k such that every Japanese triangle has a ninja path")
    print(f"with at least k red circles is: k = ⌈log₂(n+1)⌉")

    return results


if __name__ == "__main__":
    results = run_simulation()
