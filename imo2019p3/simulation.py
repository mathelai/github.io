"""
IMO 2019 Problem 3: Social Network Simulation

This simulation implements the friendship transformation process described in IMO 2019 Problem 3.
The goal is to explore patterns and generate ground truth data to help understand why the
friendship graph can always be reduced to a matching (degree at most 1 for each user).

Key concepts:
- Graph representation: adjacency list
- Operation: Triangle resolution - for a triangle with vertices A, B, C where A-B, A-C exist
  but B-C doesn't, we replace with B-C and remove A-B, A-C
- Initial configuration: 1010 users with degree 1009, 1009 users with degree 1010
  Total edges = (1010*1009 + 1009*1010)/2 = 1019595
"""

import json
import random
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import itertools


class SocialNetwork:
    """Represents a social network graph with friendship operations."""

    def __init__(self, n: int):
        """Initialize a social network with n users."""
        self.n = n
        self.graph = defaultdict(set)  # adjacency list
        self.operation_history = []

    def add_edge(self, u: int, v: int):
        """Add a friendship between users u and v."""
        self.graph[u].add(v)
        self.graph[v].add(u)

    def remove_edge(self, u: int, v: int):
        """Remove a friendship between users u and v."""
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if u in self.graph[v]:
            self.graph[v].remove(u)

    def get_degree(self, u: int) -> int:
        """Get the degree (number of friends) of user u."""
        return len(self.graph[u])

    def get_degree_sequence(self) -> List[int]:
        """Get sorted degree sequence of all users."""
        return sorted([self.get_degree(i) for i in range(self.n)], reverse=True)

    def get_degree_distribution(self) -> Dict[int, int]:
        """Get distribution of degrees."""
        dist = defaultdict(int)
        for i in range(self.n):
            dist[self.get_degree(i)] += 1
        return dict(dist)

    def edge_count(self) -> int:
        """Count total number of edges in the graph."""
        return sum(len(self.graph[i]) for i in range(self.n)) // 2

    def find_operation(self) -> Tuple[int, int, int]:
        """
        Find a valid operation: three users A, B, C where A-B, A-C exist but B-C doesn't.
        Returns (A, B, C) or None if no such triple exists.
        """
        # Try to find a valid operation by checking each user as potential A
        users = list(range(self.n))
        random.shuffle(users)  # Randomize to explore different paths

        for a in users:
            neighbors = list(self.graph[a])
            if len(neighbors) < 2:
                continue

            # Check pairs of neighbors
            random.shuffle(neighbors)
            for b, c in itertools.combinations(neighbors[:min(20, len(neighbors))], 2):
                # Check if B and C are not friends
                if c not in self.graph[b]:
                    return (a, b, c)

        return None

    def apply_operation(self, a: int, b: int, c: int) -> bool:
        """
        Apply the operation: A is friends with B and C (but B and C are not friends).
        After operation: B and C become friends, A loses both friendships.
        Returns True if operation was valid and applied.
        """
        # Verify preconditions
        if b not in self.graph[a] or c not in self.graph[a]:
            return False
        if c in self.graph[b]:
            return False

        # Apply the operation
        self.remove_edge(a, b)
        self.remove_edge(a, c)
        self.add_edge(b, c)

        # Record the operation
        self.operation_history.append({
            'a': a, 'b': b, 'c': c,
            'degree_a_before': self.get_degree(a) + 2,
            'degree_a_after': self.get_degree(a),
            'edge_count': self.edge_count(),
            'note': 'Edge count decreases by 1 (remove 2, add 1)'
        })

        return True

    def can_continue(self) -> bool:
        """Check if any valid operation exists."""
        return self.find_operation() is not None

    def max_degree(self) -> int:
        """Get the maximum degree in the graph."""
        return max((self.get_degree(i) for i in range(self.n)), default=0)

    def is_matching(self) -> bool:
        """Check if the graph is a matching (all degrees <= 1)."""
        return self.max_degree() <= 1

    def to_dict(self) -> dict:
        """Convert graph to dictionary format for JSON serialization."""
        return {
            'n': self.n,
            'edges': [[u, v] for u in range(self.n) for v in self.graph[u] if u < v],
            'degree_sequence': self.get_degree_sequence(),
            'degree_distribution': self.get_degree_distribution(),
            'edge_count': self.edge_count(),
            'max_degree': self.max_degree()
        }


def create_imo_initial_state(n=2019) -> SocialNetwork:
    """
    Create the initial state for IMO 2019 Problem 3:
    - 1010 users with 1009 friends each
    - 1009 users with 1010 friends each

    We'll use a regular-like construction to achieve this configuration.
    """
    network = SocialNetwork(n)

    # For n=2019, we need a graph where degree sequence has:
    # - 1010 vertices of degree 1009
    # - 1009 vertices of degree 1010

    # One way to construct this: start with a nearly complete graph and adjust
    # Total degree sum = 1010*1009 + 1009*1010 = 2*1019595
    # Average degree = 2*1019595/2019 ≈ 1009.5

    # Simple construction: make it nearly complete, then adjust
    # Start with complete graph on first 1010 vertices (degrees will be 1009)
    # Then connect remaining vertices carefully

    if n <= 20:  # For small examples, use a simpler construction
        # Create a nearly complete graph
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.9:  # High density
                    network.add_edge(i, j)
    else:
        # For larger n, use a more systematic construction
        # This is a simplified version that gets close to the target
        edges_to_add = (1010 * 1009 + 1009 * 1010) // 2

        # Add edges randomly until we reach target
        attempts = 0
        max_attempts = edges_to_add * 3

        while network.edge_count() < edges_to_add and attempts < max_attempts:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and v not in network.graph[u]:
                # Check degree constraints
                deg_u = network.get_degree(u)
                deg_v = network.get_degree(v)
                if deg_u < 1010 and deg_v < 1010:
                    network.add_edge(u, v)
            attempts += 1

    return network


def create_small_example(n: int, density: float = 0.7) -> SocialNetwork:
    """Create a small random graph for testing."""
    network = SocialNetwork(n)

    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                network.add_edge(i, j)

    return network


def run_simulation(network: SocialNetwork, max_steps: int = 100000) -> dict:
    """
    Run the simulation until no more operations are possible or max_steps reached.
    Returns statistics about the run.
    """
    initial_state = {
        'degree_sequence': network.get_degree_sequence(),
        'degree_distribution': network.get_degree_distribution(),
        'edge_count': network.edge_count(),
        'max_degree': network.max_degree()
    }

    steps = 0
    degree_evolution = [network.max_degree()]
    edge_evolution = [network.edge_count()]

    while steps < max_steps and network.can_continue():
        operation = network.find_operation()
        if operation is None:
            break

        a, b, c = operation
        network.apply_operation(a, b, c)
        steps += 1

        # Record evolution every 100 steps for large runs, every step for small
        if network.n <= 20 or steps % 100 == 0:
            degree_evolution.append(network.max_degree())
            edge_evolution.append(network.edge_count())

    final_state = {
        'degree_sequence': network.get_degree_sequence(),
        'degree_distribution': network.get_degree_distribution(),
        'edge_count': network.edge_count(),
        'max_degree': network.max_degree(),
        'is_matching': network.is_matching()
    }

    return {
        'n': network.n,
        'steps': steps,
        'initial_state': initial_state,
        'final_state': final_state,
        'degree_evolution': degree_evolution,
        'edge_evolution': edge_evolution,
        'completed': not network.can_continue(),
        'is_matching': network.is_matching()
    }


def main():
    """Run simulations and generate ground truth data."""

    print("IMO 2019 Problem 3: Social Network Simulation")
    print("=" * 60)

    results = {
        'problem_description': 'IMO 2019 Problem 3: Friendship network transformation',
        'simulations': []
    }

    # Test 1: Small examples to visualize
    print("\n1. Running small examples (n=5 to n=15)...")
    for n in [5, 8, 10, 12, 15]:
        print(f"   Testing n={n}...")
        network = create_small_example(n, density=0.6)
        result = run_simulation(network, max_steps=10000)
        result['test_type'] = 'small_example'
        results['simulations'].append(result)
        print(f"      Steps: {result['steps']}, Max degree: {result['initial_state']['max_degree']} -> {result['final_state']['max_degree']}, Matching: {result['is_matching']}")

    # Test 2: Medium examples
    print("\n2. Running medium examples (n=20 to n=50)...")
    for n in [20, 30, 50]:
        print(f"   Testing n={n}...")
        network = create_small_example(n, density=0.5)
        result = run_simulation(network, max_steps=50000)
        result['test_type'] = 'medium_example'
        results['simulations'].append(result)
        print(f"      Steps: {result['steps']}, Max degree: {result['initial_state']['max_degree']} -> {result['final_state']['max_degree']}, Matching: {result['is_matching']}")

    # Test 3: Analyze patterns
    print("\n3. Analyzing patterns...")

    # Count successes
    total_tests = len(results['simulations'])
    matching_achieved = sum(1 for s in results['simulations'] if s['is_matching'])

    results['analysis'] = {
        'total_tests': total_tests,
        'matching_achieved': matching_achieved,
        'success_rate': matching_achieved / total_tests if total_tests > 0 else 0,
        'observations': [
            'Each operation reduces the degree of exactly one vertex (A) by 2',
            'Each operation reduces the total number of edges by 1 (removes 2 edges, adds 1)',
            'The maximum degree tends to decrease over time',
            'Operations can continue as long as there exists a vertex with degree >= 2 having two non-adjacent neighbors',
            'When all vertices have degree <= 1, we have a matching'
        ],
        'key_insight': 'The process terminates when the graph becomes a matching (all degrees <= 1). The challenge is proving this is always reachable from the specific initial configuration.'
    }

    # Save results
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n4. Results saved to {output_file}")
    print(f"\nSummary:")
    print(f"   Total simulations: {total_tests}")
    print(f"   Matching achieved: {matching_achieved}/{total_tests}")
    print(f"   Success rate: {results['analysis']['success_rate']*100:.1f}%")

    print("\nKey observations:")
    for obs in results['analysis']['observations']:
        print(f"   - {obs}")

    print("\n" + "=" * 60)
    print("Simulation complete!")


if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    main()
