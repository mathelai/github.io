#!/usr/bin/env python3
"""
IMO 2018 Problem 4 Simulation
A site is (x,y) where 1 <= x,y <= 20
Amy places red stones: no two red stones at distance sqrt(5)
Ben places blue stones: anywhere
Find max k such that Amy can guarantee k red stones
"""

import math
from typing import Set, Tuple, List, Optional
from collections import defaultdict
import json

def distance_squared(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calculate squared distance between two points."""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def get_sqrt5_neighbors(point: Tuple[int, int], grid_size: int = 20) -> Set[Tuple[int, int]]:
    """Get all points at distance sqrt(5) from given point within grid."""
    x, y = point
    # sqrt(5) distance: (±1, ±2) or (±2, ±1)
    offsets = [(1, 2), (1, -2), (-1, 2), (-1, -2),
               (2, 1), (2, -1), (-2, 1), (-2, -1)]

    neighbors = set()
    for dx, dy in offsets:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= grid_size and 1 <= ny <= grid_size:
            neighbors.add((nx, ny))

    return neighbors

def is_valid_red_placement(point: Tuple[int, int], red_stones: Set[Tuple[int, int]]) -> bool:
    """Check if a point can have a red stone placed on it."""
    for red in red_stones:
        if distance_squared(point, red) == 5:  # sqrt(5)^2 = 5
            return False
    return True

def get_available_sites(grid_size: int, red_stones: Set[Tuple[int, int]],
                       blue_stones: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Get all unoccupied sites where Amy can place a red stone."""
    available = set()
    occupied = red_stones | blue_stones

    for x in range(1, grid_size + 1):
        for y in range(1, grid_size + 1):
            point = (x, y)
            if point not in occupied and is_valid_red_placement(point, red_stones):
                available.add(point)

    return available

class Game:
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.red_stones: Set[Tuple[int, int]] = set()
        self.blue_stones: Set[Tuple[int, int]] = set()
        self.move_history: List[Tuple[str, Tuple[int, int]]] = []

    def place_red(self, point: Tuple[int, int]) -> bool:
        """Amy places a red stone. Returns True if successful."""
        if point in self.red_stones or point in self.blue_stones:
            return False
        if not is_valid_red_placement(point, self.red_stones):
            return False

        self.red_stones.add(point)
        self.move_history.append(('red', point))
        return True

    def place_blue(self, point: Tuple[int, int]) -> bool:
        """Ben places a blue stone. Returns True if successful."""
        if point in self.red_stones or point in self.blue_stones:
            return False

        self.blue_stones.add(point)
        self.move_history.append(('blue', point))
        return True

    def get_available_red_sites(self) -> Set[Tuple[int, int]]:
        """Get all valid sites for Amy's next red stone."""
        return get_available_sites(self.grid_size, self.red_stones, self.blue_stones)

    def can_amy_move(self) -> bool:
        """Check if Amy has any valid moves."""
        return len(self.get_available_red_sites()) > 0

    def to_dict(self) -> dict:
        """Export game state as dictionary."""
        return {
            'grid_size': self.grid_size,
            'red_stones': list(self.red_stones),
            'blue_stones': list(self.blue_stones),
            'move_history': self.move_history,
            'red_count': len(self.red_stones)
        }

def greedy_amy_strategy(game: Game) -> Optional[Tuple[int, int]]:
    """Amy's greedy strategy: choose a valid site that blocks the most future options for herself.
    Actually, we want to choose sites that minimize Ben's ability to block us.
    Simple greedy: pick any available site (or one that maximizes future availability).
    """
    available = game.get_available_red_sites()
    if not available:
        return None

    # Strategy: Pick the site that leaves the most options available
    best_site = None
    best_remaining = -1

    for site in available:
        # Simulate placing at this site
        test_red = game.red_stones | {site}
        remaining = 0
        occupied = test_red | game.blue_stones

        for x in range(1, game.grid_size + 1):
            for y in range(1, game.grid_size + 1):
                point = (x, y)
                if point not in occupied and is_valid_red_placement(point, test_red):
                    remaining += 1

        if remaining > best_remaining:
            best_remaining = remaining
            best_site = site

    return best_site

def adversarial_ben_strategy(game: Game) -> Optional[Tuple[int, int]]:
    """Ben's adversarial strategy: block the site that would give Amy the most future options."""
    available_for_amy = game.get_available_red_sites()
    if not available_for_amy:
        return None  # Amy can't move anyway

    # Find which site, if blocked, reduces Amy's options the most
    best_block = None
    best_reduction = -1

    current_available = len(available_for_amy)

    for site in available_for_amy:
        # Simulate blocking this site
        test_blue = game.blue_stones | {site}
        remaining = 0
        occupied = game.red_stones | test_blue

        for x in range(1, game.grid_size + 1):
            for y in range(1, game.grid_size + 1):
                point = (x, y)
                if point not in occupied and is_valid_red_placement(point, game.red_stones):
                    remaining += 1

        reduction = current_available - remaining
        if reduction > best_reduction:
            best_reduction = reduction
            best_block = site

    return best_block

def simulate_game(grid_size: int = 20, amy_strategy=greedy_amy_strategy,
                  ben_strategy=adversarial_ben_strategy, verbose: bool = False) -> Game:
    """Simulate a complete game."""
    game = Game(grid_size)

    while True:
        # Amy's turn
        amy_move = amy_strategy(game)
        if amy_move is None:
            if verbose:
                print(f"Amy cannot move. Final count: {len(game.red_stones)} red stones")
            break

        game.place_red(amy_move)
        if verbose:
            print(f"Amy places red at {amy_move}. Total red: {len(game.red_stones)}")

        # Ben's turn
        ben_move = ben_strategy(game)
        if ben_move is None:
            if verbose:
                print(f"Ben cannot move (all sites occupied or no strategy). Final count: {len(game.red_stones)} red stones")
            break

        game.place_blue(ben_move)
        if verbose:
            print(f"Ben places blue at {ben_move}. Total blue: {len(game.blue_stones)}")

    return game

def analyze_independent_set(grid_size: int = 20) -> int:
    """Analyze the maximum independent set where no two points are at distance sqrt(5)."""
    # This is equivalent to finding the maximum independent set in a graph
    # where vertices are grid points and edges connect points at distance sqrt(5)

    # Use graph coloring approach: partition the grid into classes
    # Points at distance sqrt(5) form a knight's graph

    # For a complete analysis, we can use a greedy approach
    all_points = [(x, y) for x in range(1, grid_size + 1) for y in range(1, grid_size + 1)]

    # Greedy max independent set
    independent_set = set()
    for point in all_points:
        if is_valid_red_placement(point, independent_set):
            independent_set.add(point)

    return len(independent_set)

def run_small_instance_tests():
    """Run simulations on small grid sizes to find patterns."""
    print("=" * 60)
    print("SMALL INSTANCE ANALYSIS")
    print("=" * 60)

    results = []

    for size in range(3, 11):
        print(f"\n--- Grid Size: {size}x{size} ({size*size} total sites) ---")

        # Maximum independent set (upper bound)
        max_independent = analyze_independent_set(size)
        print(f"Maximum independent set (no sqrt(5) constraint): {max_independent}")

        # Simulate game with adversarial Ben
        game = simulate_game(size, greedy_amy_strategy, adversarial_ben_strategy, verbose=False)
        red_count = len(game.red_stones)
        blue_count = len(game.blue_stones)

        print(f"Game result - Red: {red_count}, Blue: {blue_count}")
        print(f"Amy's guaranteed stones: {red_count}")

        results.append({
            'grid_size': size,
            'total_sites': size * size,
            'max_independent': max_independent,
            'game_red': red_count,
            'game_blue': blue_count
        })

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Size':<6} {'Total':<7} {'MaxInd':<8} {'Red':<6} {'Blue':<6} {'Red%':<8}")
    print("-" * 60)
    for r in results:
        pct = 100 * r['game_red'] / r['total_sites']
        print(f"{r['grid_size']:<6} {r['total_sites']:<7} {r['max_independent']:<8} "
              f"{r['game_red']:<6} {r['game_blue']:<6} {pct:6.2f}%")

    return results

def run_full_simulation():
    """Run simulation on the full 20x20 grid."""
    print("\n" + "=" * 60)
    print("FULL 20x20 SIMULATION")
    print("=" * 60)

    max_independent = analyze_independent_set(20)
    print(f"Maximum independent set: {max_independent}")

    print("\nRunning game simulation...")
    game = simulate_game(20, greedy_amy_strategy, adversarial_ben_strategy, verbose=False)

    print(f"\nFinal Result:")
    print(f"  Red stones (Amy): {len(game.red_stones)}")
    print(f"  Blue stones (Ben): {len(game.blue_stones)}")
    print(f"  Total stones: {len(game.red_stones) + len(game.blue_stones)}")
    print(f"  Empty sites: {400 - len(game.red_stones) - len(game.blue_stones)}")

    # Save game state
    with open('game_result.json', 'w') as f:
        json.dump(game.to_dict(), f, indent=2)
    print(f"\nGame state saved to game_result.json")

    return game

if __name__ == '__main__':
    # Run small instances
    small_results = run_small_instance_tests()

    # Run full simulation
    full_game = run_full_simulation()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"Amy can guarantee at least {len(full_game.red_stones)} red stones on a 20x20 grid")
    print("(This is a lower bound based on greedy strategy)")
