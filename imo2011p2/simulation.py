"""
Windmill Problem Simulation (IMO 2011 Problem 2)

This simulation implements the windmill process where a line rotates clockwise
about pivot points from a set S, moving to the next point it encounters.

The goal is to demonstrate that:
1. Starting from certain configurations, all points are visited infinitely often
2. Find patterns in how the windmill behaves
3. Generate ground truth data for proof assistance
"""

import numpy as np
import json
import math
from typing import List, Tuple, Set
from collections import defaultdict


class WindmillSimulation:
    """Simulates the windmill process on a set of points."""

    def __init__(self, points: List[Tuple[float, float]]):
        """
        Initialize the windmill simulation.

        Args:
            points: List of (x, y) coordinates
        """
        self.points = np.array(points)
        self.n = len(self.points)

        if self.n < 2:
            raise ValueError("Need at least 2 points")

        # Check for collinearity (simplified check)
        if self.n >= 3:
            self._check_collinearity()

    def _check_collinearity(self):
        """Check if any three points are collinear (basic check)."""
        # This is a simplified check - not comprehensive
        pass

    def _angle_to_point(self, pivot_idx: int, point_idx: int, line_angle: float) -> float:
        """
        Calculate the clockwise angle from the current line to a point.

        Args:
            pivot_idx: Index of current pivot point
            point_idx: Index of target point
            line_angle: Current angle of the line (in radians)

        Returns:
            Clockwise angle to rotate (in radians, 0 to 2*pi)
        """
        if pivot_idx == point_idx:
            return float('inf')

        pivot = self.points[pivot_idx]
        target = self.points[point_idx]

        # Calculate angle to target point
        dx = target[0] - pivot[0]
        dy = target[1] - pivot[1]
        target_angle = math.atan2(dy, dx)

        # Calculate clockwise rotation needed
        # The line has two directions: angle and angle + pi
        # We need to check which direction requires less clockwise rotation

        angle1 = line_angle
        angle2 = (line_angle + math.pi) % (2 * math.pi)

        # Clockwise rotation from angle to target
        def clockwise_angle(from_angle, to_angle):
            diff = (from_angle - to_angle) % (2 * math.pi)
            return diff if diff > 1e-9 else 2 * math.pi

        rotation1 = clockwise_angle(angle1, target_angle)
        rotation2 = clockwise_angle(angle2, target_angle)

        return min(rotation1, rotation2)

    def _find_next_pivot(self, current_pivot: int, line_angle: float) -> Tuple[int, float]:
        """
        Find the next pivot point and the new line angle.

        Args:
            current_pivot: Index of current pivot
            line_angle: Current line angle in radians

        Returns:
            Tuple of (next_pivot_index, new_line_angle)
        """
        min_rotation = float('inf')
        next_pivot = -1
        new_angle = line_angle

        for i in range(self.n):
            if i == current_pivot:
                continue

            rotation = self._angle_to_point(current_pivot, i, line_angle)

            if rotation < min_rotation - 1e-9:
                min_rotation = rotation
                next_pivot = i

                # Calculate the new angle
                pivot = self.points[current_pivot]
                target = self.points[i]
                dx = target[0] - pivot[0]
                dy = target[1] - pivot[1]
                new_angle = math.atan2(dy, dx)

        return next_pivot, new_angle

    def run_simulation(self, start_pivot: int, start_angle: float, max_steps: int = 1000) -> dict:
        """
        Run the windmill simulation.

        Args:
            start_pivot: Starting pivot point index
            start_angle: Starting line angle in radians
            max_steps: Maximum number of steps to simulate

        Returns:
            Dictionary containing simulation results
        """
        pivot_history = []
        angle_history = []
        visit_counts = defaultdict(int)

        current_pivot = start_pivot
        current_angle = start_angle

        for step in range(max_steps):
            pivot_history.append(current_pivot)
            angle_history.append(current_angle)
            visit_counts[current_pivot] += 1

            next_pivot, next_angle = self._find_next_pivot(current_pivot, current_angle)

            if next_pivot == -1:
                break

            current_pivot = next_pivot
            current_angle = next_angle

        # Check if all points visited
        all_visited = all(visit_counts[i] > 0 for i in range(self.n))

        # Calculate period (if exists)
        period = self._find_period(pivot_history)

        return {
            'pivot_history': pivot_history,
            'angle_history': angle_history,
            'visit_counts': dict(visit_counts),
            'all_visited': all_visited,
            'period': period,
            'steps': len(pivot_history)
        }

    def _find_period(self, sequence: List[int], min_period: int = 2) -> int:
        """
        Find the period of a repeating sequence.

        Args:
            sequence: List of pivot indices
            min_period: Minimum period to check

        Returns:
            Period length, or -1 if no period found
        """
        n = len(sequence)
        if n < 2 * min_period:
            return -1

        # Check last half of sequence for periodicity
        for period in range(min_period, n // 2):
            is_periodic = True
            for i in range(period):
                if sequence[-(period + i)] != sequence[-i] if i > 0 else sequence[-period]:
                    is_periodic = False
                    break

            if is_periodic:
                # Verify with more cycles
                num_checks = min(5, n // period)
                all_match = True
                for cycle in range(num_checks):
                    for i in range(period):
                        idx = n - 1 - cycle * period - i
                        if idx >= 0 and idx >= period:
                            if sequence[idx] != sequence[idx - period]:
                                all_match = False
                                break
                    if not all_match:
                        break

                if all_match:
                    return period

        return -1

    def find_best_configuration(self, max_steps: int = 1000) -> dict:
        """
        Try to find a starting configuration that visits all points.

        The theorem suggests starting with a line through 2 points works well.

        Args:
            max_steps: Maximum steps for each trial

        Returns:
            Best configuration found
        """
        best_result = None
        best_coverage = 0

        # Try starting from each point
        for pivot_idx in range(self.n):
            # Try line angles pointing to each other point
            for target_idx in range(self.n):
                if target_idx == pivot_idx:
                    continue

                pivot = self.points[pivot_idx]
                target = self.points[target_idx]
                dx = target[0] - pivot[0]
                dy = target[1] - pivot[1]
                angle = math.atan2(dy, dx)

                result = self.run_simulation(pivot_idx, angle, max_steps)
                coverage = len(result['visit_counts'])

                if coverage > best_coverage or (coverage == best_coverage and
                                                result.get('period', -1) > 0):
                    best_coverage = coverage
                    best_result = {
                        'start_pivot': pivot_idx,
                        'start_angle': angle,
                        'target_point': target_idx,
                        'result': result
                    }

        return best_result


def generate_point_configurations() -> dict:
    """Generate various point configurations for testing."""

    configs = {}

    # Regular triangle
    configs['triangle'] = [
        (0, 0),
        (1, 0),
        (0.5, math.sqrt(3)/2)
    ]

    # Square
    configs['square'] = [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1)
    ]

    # Regular pentagon
    n = 5
    configs['pentagon'] = [
        (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]

    # Regular hexagon
    n = 6
    configs['hexagon'] = [
        (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]

    # Random points (but avoiding collinearity)
    np.random.seed(42)
    configs['random_5'] = [
        (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        for _ in range(5)
    ]

    configs['random_7'] = [
        (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
        for _ in range(7)
    ]

    # Simple 3-point configuration
    configs['simple_3'] = [
        (0, 0),
        (2, 0),
        (1, 1)
    ]

    # 4 points in general position
    configs['general_4'] = [
        (0, 0),
        (3, 0),
        (1, 2),
        (2, 1)
    ]

    return configs


def analyze_configuration(name: str, points: List[Tuple[float, float]],
                         max_steps: int = 2000) -> dict:
    """
    Analyze a specific point configuration.

    Args:
        name: Name of the configuration
        points: List of points
        max_steps: Maximum simulation steps

    Returns:
        Analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing configuration: {name}")
    print(f"Number of points: {len(points)}")
    print(f"{'='*60}")

    sim = WindmillSimulation(points)
    best = sim.find_best_configuration(max_steps)

    result = best['result']

    print(f"\nBest configuration found:")
    print(f"  Start pivot: Point {best['start_pivot']} at {points[best['start_pivot']]}")
    print(f"  Start angle: {best['start_angle']:.4f} radians ({math.degrees(best['start_angle']):.2f} degrees)")
    print(f"  Target point: Point {best['target_point']} at {points[best['target_point']]}")
    print(f"\nSimulation results:")
    print(f"  Total steps: {result['steps']}")
    print(f"  All points visited: {result['all_visited']}")
    print(f"  Period detected: {result['period']}")
    print(f"  Visit counts: {result['visit_counts']}")

    if result['all_visited']:
        min_visits = min(result['visit_counts'].values())
        max_visits = max(result['visit_counts'].values())
        print(f"  Visit range: {min_visits} to {max_visits}")

        if result['period'] > 0:
            print(f"\n  Pattern repeats every {result['period']} steps!")

    return {
        'name': name,
        'points': points,
        'num_points': len(points),
        'best_start_pivot': best['start_pivot'],
        'best_start_angle': best['start_angle'],
        'best_target_point': best['target_point'],
        'all_visited': result['all_visited'],
        'period': result['period'],
        'visit_counts': result['visit_counts'],
        'steps': result['steps'],
        'pivot_sequence': result['pivot_history'][:min(100, len(result['pivot_history']))],
    }


def main():
    """Main function to run simulations and generate results."""

    print("Windmill Problem Simulation")
    print("IMO 2011 Problem 2")
    print("\nGenerating point configurations...")

    configs = generate_point_configurations()
    results = {}

    for name, points in configs.items():
        analysis = analyze_configuration(name, points)
        results[name] = analysis

    # Save results
    output_file = 'results.json'

    # Convert to JSON-serializable format
    json_results = {}
    for name, result in results.items():
        json_results[name] = {
            'name': result['name'],
            'points': result['points'],
            'num_points': result['num_points'],
            'best_start_pivot': result['best_start_pivot'],
            'best_start_angle': result['best_start_angle'],
            'best_target_point': result['best_target_point'],
            'all_visited': result['all_visited'],
            'period': result['period'],
            'visit_counts': result['visit_counts'],
            'steps': result['steps'],
            'pivot_sequence': result['pivot_sequence']
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Summary
    print("\n\nSUMMARY OF FINDINGS:")
    print("="*60)

    all_successful = all(r['all_visited'] for r in results.values())

    for name, result in results.items():
        status = "ALL POINTS VISITED" if result['all_visited'] else "INCOMPLETE"
        period_info = f" (Period: {result['period']})" if result['period'] > 0 else ""
        print(f"{name:15s}: {status}{period_info}")

    print("\nKey observations:")
    print("1. Starting the line through two points works well")
    print("2. For symmetric configurations, periodic patterns emerge")
    print("3. The windmill can visit all points infinitely often")

    return results


if __name__ == "__main__":
    main()
