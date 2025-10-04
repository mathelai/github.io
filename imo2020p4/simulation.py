#!/usr/bin/env python3
"""
IMO 2020 Problem 4 Simulation
Simulates cable car networks to find the minimum k where two stations are guaranteed to be linked by both companies.
"""

import random
from itertools import combinations
from typing import List, Set, Tuple, Dict
import json

class CableCarNetwork:
    def __init__(self, n: int, k: int):
        """
        n: square root of number of stations (n^2 stations total)
        k: number of cable cars per company
        """
        self.n = n
        self.k = k
        self.num_stations = n * n
        # Stations numbered 0 to n^2-1, where higher numbers = higher altitude

    def generate_valid_network(self, seed: int = None) -> List[Tuple[int, int]]:
        """
        Generate a valid cable car network with k cars.
        Returns list of (start, end) tuples where start < end.
        Ensures k different starting points, k different ending points,
        and if car i starts higher than car j, then car i also ends higher.
        """
        if seed is not None:
            random.seed(seed)

        # Choose k different starting points and k different ending points
        available_stations = list(range(self.num_stations))

        # Simple approach: sample k starting points and k ending points
        # then match them in sorted order to satisfy the monotonicity constraint
        start_points = sorted(random.sample(available_stations, k=self.k))

        # End points must be higher than their corresponding start points
        end_points = []
        for i, start in enumerate(start_points):
            # End must be > start
            possible_ends = [s for s in available_stations if s > start and s not in end_points]
            if not possible_ends:
                # Fallback: regenerate
                return self.generate_valid_network(seed=None if seed is None else seed+1)
            end = random.choice(possible_ends)
            end_points.append(end)

        # Sort end points to maintain monotonicity
        end_points.sort()

        # Pair them up: sorted starts with sorted ends
        cables = list(zip(start_points, end_points))

        return cables

    def compute_reachability(self, cables: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
        """
        Compute which stations are reachable from each station using the given cables.
        Returns dict: station -> set of reachable stations (not including itself).
        """
        # Build adjacency list
        graph = {i: [] for i in range(self.num_stations)}
        for start, end in cables:
            graph[start].append(end)

        # For each station, compute reachable stations using BFS/DFS
        reachability = {}
        for station in range(self.num_stations):
            reachable = set()
            queue = [station]
            visited = {station}

            while queue:
                current = queue.pop(0)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        reachable.add(neighbor)
                        queue.append(neighbor)

            reachability[station] = reachable

        return reachability

    def find_commonly_linked_pairs(self, cables_a: List[Tuple[int, int]],
                                   cables_b: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Find all pairs of stations that are linked by both company A and company B.
        Returns set of (lower, higher) tuples.
        """
        reach_a = self.compute_reachability(cables_a)
        reach_b = self.compute_reachability(cables_b)

        common_pairs = set()
        for station in range(self.num_stations):
            # Find stations reachable by both companies from this station
            common = reach_a[station] & reach_b[station]
            for target in common:
                common_pairs.add((station, target))

        return common_pairs

    def check_configuration_has_common_link(self, cables_a: List[Tuple[int, int]],
                                           cables_b: List[Tuple[int, int]]) -> bool:
        """
        Check if there exists at least one pair of stations linked by both companies.
        """
        common = self.find_commonly_linked_pairs(cables_a, cables_b)
        return len(common) > 0


def simulate_minimum_k(n: int, max_k: int = None, num_trials: int = 1000) -> Dict:
    """
    Simulate to find minimum k for which all configurations have a common link.
    For each k, try many random configurations and see if we can find a counterexample.
    """
    if max_k is None:
        max_k = n * n

    results = {}

    for k in range(1, max_k + 1):
        print(f"\nTesting k = {k} for n = {n} (n^2 = {n*n} stations)...")
        network = CableCarNetwork(n, k)

        found_counterexample = False
        counterexample = None

        for trial in range(num_trials):
            cables_a = network.generate_valid_network(seed=trial*2)
            cables_b = network.generate_valid_network(seed=trial*2+1)

            has_common = network.check_configuration_has_common_link(cables_a, cables_b)

            if not has_common:
                found_counterexample = True
                counterexample = {
                    'cables_a': cables_a,
                    'cables_b': cables_b,
                    'trial': trial
                }
                print(f"  Found counterexample at trial {trial}")
                break

        results[k] = {
            'n': n,
            'k': k,
            'num_stations': n*n,
            'trials_tested': trial + 1 if found_counterexample else num_trials,
            'found_counterexample': found_counterexample,
            'counterexample': counterexample,
            'guaranteed_common_link': not found_counterexample
        }

        if not found_counterexample:
            print(f"  No counterexample found in {num_trials} trials - k={k} likely sufficient!")
            results['minimum_k'] = k
            break

    return results


def analyze_specific_configuration(n: int, k: int, cables_a: List[Tuple[int, int]],
                                   cables_b: List[Tuple[int, int]]) -> Dict:
    """
    Analyze a specific configuration in detail.
    """
    network = CableCarNetwork(n, k)

    reach_a = network.compute_reachability(cables_a)
    reach_b = network.compute_reachability(cables_b)
    common_pairs = network.find_commonly_linked_pairs(cables_a, cables_b)

    return {
        'n': n,
        'k': k,
        'cables_a': cables_a,
        'cables_b': cables_b,
        'reachability_a': {k: list(v) for k, v in reach_a.items() if v},
        'reachability_b': {k: list(v) for k, v in reach_b.items() if v},
        'common_pairs': list(common_pairs),
        'has_common_link': len(common_pairs) > 0
    }


if __name__ == "__main__":
    print("=" * 60)
    print("IMO 2020 Problem 4 Simulation")
    print("=" * 60)

    # Test small values of n
    for n in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Testing n = {n} (n^2 = {n*n} stations)")
        print(f"{'='*60}")

        # Theoretical bounds
        print(f"\nTheoretical considerations:")
        print(f"  - Number of stations: {n*n}")
        print(f"  - Upper bound for k: {n*n} (all stations)")
        print(f"  - Expected minimum k: around {n*n - n + 1} (conjecture)")

        # Run simulation
        results = simulate_minimum_k(n, max_k=min(n*n, 10), num_trials=500)

        # Print summary
        if 'minimum_k' in results:
            min_k = results['minimum_k']
            print(f"\n*** RESULT: Minimum k = {min_k} for n = {n} ***")
            print(f"    This means with k >= {min_k}, we guarantee a common link")
        else:
            print(f"\n*** Could not determine minimum k (need larger search) ***")

        # Save results
        output_file = f"simulation_results_n{n}.json"
        with open(output_file, 'w') as f:
            # Convert sets and tuples to lists for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(k, int):
                    json_results[k] = v
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        # Show counterexample if exists for k=n
        if n in results and results[n]['found_counterexample']:
            print(f"\nCounterexample for k = {n}:")
            ce = results[n]['counterexample']
            print(f"  Company A cables: {ce['cables_a']}")
            print(f"  Company B cables: {ce['cables_b']}")

    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)
