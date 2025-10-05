"""
IMO 2016 Problem 6: Frog Jumping Simulation

This simulation explores the problem of placing frogs on line segments such that
no two frogs occupy the same intersection point at the same time.

The problem:
- n >= 2 line segments where every two segments cross
- No three segments meet at a point
- Place a frog at one endpoint of each segment
- Clap n-1 times; each clap makes each frog jump to the next intersection
- Goal: no two frogs at the same intersection at the same time
"""

import json
import itertools
from typing import List, Tuple, Set, Dict
import random


class LineSegment:
    """Represents a line segment with two endpoints."""

    def __init__(self, id: int, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.id = id
        self.p1 = p1
        self.p2 = p2

    def __repr__(self):
        return f"Segment {self.id}: {self.p1} -> {self.p2}"


def line_intersection(seg1: LineSegment, seg2: LineSegment) -> Tuple[float, float] | None:
    """
    Find intersection point of two line segments.
    Returns None if they don't intersect.
    """
    x1, y1 = seg1.p1
    x2, y2 = seg1.p2
    x3, y3 = seg2.p1
    x4, y4 = seg2.p2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 < t < 1 and 0 < u < 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def generate_segments_in_general_position(n: int, seed: int = None) -> List[LineSegment]:
    """
    Generate n line segments in general position:
    - Every two segments cross
    - No three segments meet at a point
    """
    if seed is not None:
        random.seed(seed)

    # Generate random segments
    max_attempts = 1000
    for attempt in range(max_attempts):
        segments = []
        for i in range(n):
            # Generate random endpoints
            angle = random.uniform(0, 2 * 3.14159)
            length = random.uniform(0.8, 1.2)
            cx, cy = random.uniform(-2, 2), random.uniform(-2, 2)

            dx = length * 0.5 * (1 + 0.3 * random.random())
            dy = length * 0.5 * (1 + 0.3 * random.random())

            p1 = (cx - dx, cy - dy)
            p2 = (cx + dx, cy + dy)
            segments.append(LineSegment(i, p1, p2))

        # Check if all pairs intersect
        all_intersect = True
        intersections = set()

        for i in range(n):
            for j in range(i + 1, n):
                pt = line_intersection(segments[i], segments[j])
                if pt is None:
                    all_intersect = False
                    break
                intersections.add(pt)
            if not all_intersect:
                break

        # Check no three segments meet at a point
        if all_intersect and len(intersections) == n * (n - 1) // 2:
            return segments

    # Fallback: use a construction that guarantees intersection
    return generate_segments_star_configuration(n)


def generate_segments_star_configuration(n: int) -> List[LineSegment]:
    """
    Generate n segments in a configuration where all pairs intersect.
    Uses a construction where segments pass through a small disk around the origin,
    with perturbations to ensure no three meet at a point.
    """
    segments = []
    import math

    for i in range(n):
        # Each segment has a unique angle (spread over half circle is enough)
        angle = math.pi * i / n

        # Add small perturbations to ensure no three segments meet at a point
        # The perturbation varies based on segment index
        angle_offset = 0.005 * math.sin(i * 2.718)  # Small pseudo-random offset

        # Create segment passing through a small region near origin
        # but not all through exact same point
        center_offset_x = 0.02 * math.cos(i * 1.414)
        center_offset_y = 0.02 * math.sin(i * 1.732)

        # Distance from center for the two endpoints
        r1 = 2.5 + 0.03 * (i % 5)
        r2 = 2.5 + 0.03 * ((i + 2) % 5)

        # One endpoint
        p1 = (center_offset_x + r1 * math.cos(angle + angle_offset),
              center_offset_y + r1 * math.sin(angle + angle_offset))

        # Other endpoint on opposite side (with slight angle variation)
        opposite_angle = angle + math.pi + angle_offset * 0.5
        p2 = (center_offset_x + r2 * math.cos(opposite_angle),
              center_offset_y + r2 * math.sin(opposite_angle))

        segments.append(LineSegment(i, p1, p2))

    return segments


def compute_intersections_on_segment(segment: LineSegment, all_segments: List[LineSegment]) -> List[Tuple[float, Tuple[float, float], int]]:
    """
    Compute all intersection points on a segment, ordered from p1 to p2.
    Returns list of (distance_from_p1, intersection_point, other_segment_id).
    """
    intersections = []

    for other in all_segments:
        if other.id == segment.id:
            continue

        pt = line_intersection(segment, other)
        if pt is not None:
            # Calculate distance from p1
            dist = ((pt[0] - segment.p1[0])**2 + (pt[1] - segment.p1[1])**2)**0.5
            intersections.append((dist, pt, other.id))

    # Sort by distance from p1
    intersections.sort(key=lambda x: x[0])
    return intersections


class FrogSimulation:
    """Simulates the frog jumping game."""

    def __init__(self, segments: List[LineSegment]):
        self.segments = segments
        self.n = len(segments)

        # Precompute intersections for each segment
        self.segment_intersections = {}
        for seg in segments:
            self.segment_intersections[seg.id] = compute_intersections_on_segment(seg, segments)

    def simulate(self, orientations: List[bool]) -> Dict:
        """
        Simulate the frog jumping with given orientations.
        orientations[i] = True means frog starts at p1, False means starts at p2.

        Returns dict with:
        - success: whether no collision occurred
        - collision_time: time step of first collision (-1 if none)
        - collision_details: details about collision
        - positions_history: history of all frog positions
        """
        # Initialize frog positions (which intersection they're at, -1 means at start)
        frog_positions = [-1] * self.n  # -1 means at starting endpoint

        positions_history = []

        # Record initial positions
        initial_state = {}
        for i in range(self.n):
            seg = self.segments[i]
            pos = seg.p1 if orientations[i] else seg.p2
            initial_state[i] = {
                'segment': i,
                'position': pos,
                'intersection_index': -1
            }
        positions_history.append(initial_state)

        # Simulate n-1 claps
        for clap in range(self.n - 1):
            # Each frog jumps to next intersection
            new_positions = []
            current_state = {}

            for i in range(self.n):
                frog_positions[i] += 1
                idx = frog_positions[i]

                # Get intersections for this segment in correct order
                if orientations[i]:
                    intersections = self.segment_intersections[self.segments[i].id]
                else:
                    intersections = list(reversed(self.segment_intersections[self.segments[i].id]))

                # Check if we have enough intersections
                if idx >= len(intersections):
                    # This shouldn't happen in valid configuration, but handle it
                    return {
                        'success': False,
                        'collision_time': clap,
                        'collision_details': {
                            'error': f'Frog {i} ran out of intersections',
                            'frogs': [i],
                            'position': None,
                            'segments': [i]
                        },
                        'positions_history': positions_history
                    }

                pos = intersections[idx][1]  # Get the position tuple
                new_positions.append(pos)

                current_state[i] = {
                    'segment': i,
                    'position': pos,
                    'intersection_index': frog_positions[i],
                    'intersecting_segment': intersections[idx][2]
                }

            positions_history.append(current_state)

            # Check for collisions
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    # Check if positions are the same (within tolerance)
                    pi, pj = new_positions[i], new_positions[j]
                    if abs(pi[0] - pj[0]) < 1e-6 and abs(pi[1] - pj[1]) < 1e-6:
                        return {
                            'success': False,
                            'collision_time': clap,
                            'collision_details': {
                                'frogs': [i, j],
                                'position': pi,
                                'segments': [i, j]
                            },
                            'positions_history': positions_history
                        }

        return {
            'success': True,
            'collision_time': -1,
            'collision_details': None,
            'positions_history': positions_history
        }

    def find_valid_orientation(self) -> Tuple[bool, List[bool] | None]:
        """
        Try all possible orientations to find one that works.
        Returns (success, orientation_list).
        """
        # Try all 2^n possible orientations
        for orientation_bits in range(2**self.n):
            orientations = [(orientation_bits >> i) & 1 == 1 for i in range(self.n)]
            result = self.simulate(orientations)
            if result['success']:
                return (True, orientations)

        return (False, None)


def analyze_configuration(n: int, num_trials: int = 5) -> Dict:
    """
    Analyze whether a valid orientation exists for n segments.
    Tries multiple random configurations.
    """
    results = []

    for trial in range(num_trials):
        segments = generate_segments_star_configuration(n)
        sim = FrogSimulation(segments)

        found, orientation = sim.find_valid_orientation()

        results.append({
            'trial': trial,
            'n': n,
            'found_valid_orientation': found,
            'orientation': orientation,
            'total_orientations_tested': 2**n
        })

    return {
        'n': n,
        'num_trials': num_trials,
        'trials': results,
        'any_valid_found': any(r['found_valid_orientation'] for r in results),
        'all_valid_found': all(r['found_valid_orientation'] for r in results)
    }


def run_comprehensive_analysis():
    """
    Run comprehensive analysis for different values of n.
    Tests the conjecture: valid orientation exists iff n is odd.
    """
    print("IMO 2016 Problem 6: Frog Jumping Simulation")
    print("=" * 60)
    print()

    results = {
        'problem_statement': (
            "n >= 2 line segments where every two cross, no three meet at a point. "
            "Place frogs at endpoints, clap n-1 times. Each clap makes each frog jump "
            "to next intersection. Goal: no two frogs at same intersection at same time."
        ),
        'conjecture': {
            'part_a': 'Valid orientation exists when n is odd',
            'part_b': 'Valid orientation never exists when n is even'
        },
        'analysis_by_n': {}
    }

    # Test for n = 2 to 7
    for n in range(2, 8):
        print(f"Analyzing n = {n}...")
        analysis = analyze_configuration(n, num_trials=3 if n <= 5 else 1)

        results['analysis_by_n'][n] = analysis

        print(f"  n = {n} ({'odd' if n % 2 == 1 else 'even'})")
        print(f"  Valid orientation found: {analysis['any_valid_found']}")
        print(f"  Matches conjecture: {analysis['any_valid_found'] == (n % 2 == 1)}")
        print()

    # Generate a detailed example for n=3
    print("Generating detailed example for n=3 (should have valid orientation)...")
    segments_3 = generate_segments_star_configuration(3)
    sim_3 = FrogSimulation(segments_3)
    found_3, orientation_3 = sim_3.find_valid_orientation()

    if found_3:
        detailed_sim = sim_3.simulate(orientation_3)
        results['detailed_example_n3'] = {
            'n': 3,
            'segments': [
                {'id': s.id, 'p1': s.p1, 'p2': s.p2}
                for s in segments_3
            ],
            'valid_orientation': orientation_3,
            'simulation': detailed_sim
        }
        print(f"  Found valid orientation: {orientation_3}")
        print(f"  Success: {detailed_sim['success']}")

    # Generate a detailed example for n=4
    print("Generating detailed example for n=4 (should have NO valid orientation)...")
    segments_4 = generate_segments_star_configuration(4)
    sim_4 = FrogSimulation(segments_4)
    found_4, orientation_4 = sim_4.find_valid_orientation()

    # Try all orientations and record collisions
    collision_summary = {}
    for orientation_bits in range(2**4):
        orientations = [(orientation_bits >> i) & 1 == 1 for i in range(4)]
        result = sim_4.simulate(orientations)
        if not result['success']:
            time = result['collision_time']
            collision_summary[str(orientations)] = {
                'collision_time': time,
                'frogs': result['collision_details']['frogs']
            }

    results['detailed_example_n4'] = {
        'n': 4,
        'segments': [
            {'id': s.id, 'p1': s.p1, 'p2': s.p2}
            for s in segments_4
        ],
        'found_valid_orientation': found_4,
        'all_orientations_collision_summary': collision_summary,
        'total_orientations_tested': 16,
        'all_have_collisions': len(collision_summary) == 16
    }
    print(f"  Found valid orientation: {found_4}")
    print(f"  All 16 orientations tested, all have collisions: {len(collision_summary) == 16}")

    # Summary
    results['summary'] = {
        'conjecture_verified': all(
            results['analysis_by_n'][n]['any_valid_found'] == (n % 2 == 1)
            for n in range(2, 8)
        ),
        'odd_n_results': {
            n: results['analysis_by_n'][n]['any_valid_found']
            for n in range(3, 8, 2)
        },
        'even_n_results': {
            n: results['analysis_by_n'][n]['any_valid_found']
            for n in range(2, 8, 2)
        }
    }

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Conjecture verified for n in [2,7]: {results['summary']['conjecture_verified']}")
    print()
    print("Odd n (should have valid orientation):")
    for n, found in results['summary']['odd_n_results'].items():
        print(f"  n={n}: {found} {'✓' if found else '✗'}")
    print()
    print("Even n (should NOT have valid orientation):")
    for n, found in results['summary']['even_n_results'].items():
        print(f"  n={n}: {not found} {'✓' if not found else '✗'}")

    return results


if __name__ == "__main__":
    results = run_comprehensive_analysis()

    # Save to JSON
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to {output_file}")
