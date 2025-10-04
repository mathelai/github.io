"""
IMO 2004 Problem 3: Hook Tiling Simulation

PROBLEM:
Define a hook to be a figure made up of six unit squares as shown below,
or any of the figures obtained by applying rotations and reflections to this figure.

The hook shape (one of 4 unique orientations):
  XX
  X
  X
  XX

TASK:
Determine all m×n rectangles that can be covered without gaps and without overlaps
with hooks, such that no part of a hook covers area outside the rectangle.

USAGE:
1. Run full analysis: python simulation.py
   - Tests rectangles up to 10×10
   - Generates results.json with findings
   - Displays summary statistics

2. Test specific rectangle: python simulation.py m n
   - Example: python simulation.py 4 6
   - Shows whether the m×n rectangle can be tiled

IMPLEMENTATION:
- Uses backtracking algorithm to exhaustively search for tilings
- Generates all unique hook orientations via rotation and reflection
- Each hook orientation is represented as a list of (row, col) offsets

KEY FINDINGS:
- Among all rectangles tested (up to 10×10), NONE can be tiled with hooks
- Even rectangles with area divisible by 6 cannot be tiled
- This suggests the answer is that NO m×n rectangle can be tiled with hooks,
  or only very specific large rectangles (if any) can be tiled
"""

import json
from typing import List, Tuple, Set
from itertools import product

# Define all 8 possible hook orientations (rotations and reflections)
# Each hook is represented as a list of (row, col) offsets from a reference point
# Standard hook shape (6 unit squares) - looks like:
#   XX
#   X
#   XX
# This is 3 rows, 2 cols with 6 cells total

def generate_all_hooks():
    """Generate all unique hook orientations through rotation and reflection."""
    # Base hook shape (6 unit squares):
    #   XX
    #   X
    #   X
    #   XX
    base = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (3, 1)]

    def rotate_90(shape):
        """Rotate shape 90 degrees clockwise."""
        # (r, c) -> (c, -r) but we normalize to keep min at (0,0)
        rotated = [(c, -r) for r, c in shape]
        min_r = min(r for r, c in rotated)
        min_c = min(c for r, c in rotated)
        return sorted([(r - min_r, c - min_c) for r, c in rotated])

    def flip_horizontal(shape):
        """Flip shape horizontally."""
        flipped = [(r, -c) for r, c in shape]
        min_r = min(r for r, c in flipped)
        min_c = min(c for r, c in flipped)
        return sorted([(r - min_r, c - min_c) for r, c in flipped])

    # Generate all rotations and reflections
    all_hooks = []
    current = base

    # 4 rotations
    for _ in range(4):
        all_hooks.append(current)
        current = rotate_90(current)

    # Flip and 4 more rotations
    current = flip_horizontal(base)
    for _ in range(4):
        all_hooks.append(current)
        current = rotate_90(current)

    # Remove duplicates
    unique_hooks = []
    for hook in all_hooks:
        if hook not in unique_hooks:
            unique_hooks.append(hook)

    return unique_hooks

HOOKS = generate_all_hooks()

def visualize_hook(hook: List[Tuple[int, int]]) -> str:
    """Create a text visualization of a hook."""
    max_r = max(r for r, c in hook)
    max_c = max(c for r, c in hook)

    grid = [['.' for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for r, c in hook:
        grid[r][c] = 'X'

    return '\n'.join(''.join(row) for row in grid)

def visualize_grid(grid: List[List[bool]]) -> str:
    """Create a text visualization of a grid."""
    return '\n'.join(''.join('X' if cell else '.' for cell in row) for row in grid)

class HookTiler:
    """Attempts to tile an m×n rectangle with hooks using backtracking."""

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.grid = [[False] * n for _ in range(m)]
        self.placements = []  # List of (hook_index, row, col, orientation)

    def can_place_hook(self, hook: List[Tuple[int, int]], row: int, col: int) -> bool:
        """Check if a hook can be placed at (row, col)."""
        for dr, dc in hook:
            r, c = row + dr, col + dc
            if r < 0 or r >= self.m or c < 0 or c >= self.n:
                return False
            if self.grid[r][c]:
                return False
        return True

    def place_hook(self, hook: List[Tuple[int, int]], row: int, col: int, orientation: int):
        """Place a hook on the grid."""
        cells = []
        for dr, dc in hook:
            r, c = row + dr, col + dc
            self.grid[r][c] = True
            cells.append((r, c))
        self.placements.append({
            'row': row,
            'col': col,
            'orientation': orientation,
            'cells': cells
        })

    def remove_hook(self, hook: List[Tuple[int, int]], row: int, col: int):
        """Remove a hook from the grid."""
        for dr, dc in hook:
            r, c = row + dr, col + dc
            self.grid[r][c] = False
        self.placements.pop()

    def find_first_empty(self) -> Tuple[int, int]:
        """Find the first empty cell (top-left scanning)."""
        for i in range(self.m):
            for j in range(self.n):
                if not self.grid[i][j]:
                    return (i, j)
        return None

    def solve(self, max_solutions: int = 1) -> List[List[dict]]:
        """Attempt to tile the rectangle using backtracking."""
        solutions = []

        def backtrack():
            if len(solutions) >= max_solutions:
                return

            # Find first empty cell
            pos = self.find_first_empty()
            if pos is None:
                # Found a complete tiling!
                solutions.append([p.copy() for p in self.placements])
                return

            row, col = pos

            # Try each hook orientation
            for orientation_idx, hook in enumerate(HOOKS):
                if self.can_place_hook(hook, row, col):
                    self.place_hook(hook, row, col, orientation_idx)
                    backtrack()
                    self.remove_hook(hook, row, col)

        backtrack()
        return solutions

def check_divisibility_condition(m: int, n: int) -> dict:
    """Check if m*n is divisible by 6 (necessary condition)."""
    area = m * n
    return {
        'area': area,
        'divisible_by_6': area % 6 == 0,
        'divisible_by_2': area % 2 == 0,
        'divisible_by_3': area % 3 == 0
    }

def check_tileability(m: int, n: int, timeout_attempts: int = 100000) -> dict:
    """
    Check if an m×n rectangle can be tiled with hooks.
    Uses backtracking with early termination.
    """
    # Check necessary condition
    if (m * n) % 6 != 0:
        return {
            'tileable': False,
            'reason': 'Area not divisible by 6',
            'solutions': []
        }

    # Try to find a tiling
    tiler = HookTiler(m, n)
    solutions = tiler.solve(max_solutions=1)

    return {
        'tileable': len(solutions) > 0,
        'reason': 'Tiling found' if solutions else 'No tiling found',
        'solutions': solutions[:1],  # Return at most one solution
        'num_hooks': len(solutions[0]) if solutions else 0
    }

def analyze_small_rectangles(max_size: int = 10) -> dict:
    """Analyze which small rectangles can be tiled."""
    results = {
        'tileable': [],
        'not_tileable': [],
        'patterns': {}
    }

    # Check all rectangles up to max_size
    for m in range(1, max_size + 1):
        for n in range(m, max_size + 1):  # n >= m to avoid duplicates
            print(f"Checking {m}×{n}...")

            div_check = check_divisibility_condition(m, n)
            if not div_check['divisible_by_6']:
                results['not_tileable'].append({
                    'm': m,
                    'n': n,
                    'reason': 'Area not divisible by 6'
                })
                continue

            tile_result = check_tileability(m, n)

            if tile_result['tileable']:
                results['tileable'].append({
                    'm': m,
                    'n': n,
                    'num_hooks': tile_result['num_hooks'],
                    'solution': tile_result['solutions'][0] if tile_result['solutions'] else None
                })
                print(f"  ✓ Tileable with {tile_result['num_hooks']} hooks")
            else:
                results['not_tileable'].append({
                    'm': m,
                    'n': n,
                    'reason': tile_result['reason']
                })
                print(f"  ✗ Not tileable")

    return results

def find_patterns(results: dict) -> dict:
    """Analyze results to find patterns."""
    patterns = {
        'observations': []
    }

    tileable = results['tileable']
    not_tileable = results['not_tileable']

    # Check which dimensions appear
    tileable_dims = set()
    for rect in tileable:
        tileable_dims.add((rect['m'], rect['n']))
        if rect['m'] != rect['n']:
            tileable_dims.add((rect['n'], rect['m']))

    # Observation 1: Check modulo conditions
    tileable_mod_2 = [(r['m'] % 2, r['n'] % 2) for r in tileable]
    tileable_mod_3 = [(r['m'] % 3, r['n'] % 3) for r in tileable]

    patterns['observations'].append({
        'type': 'modulo_2',
        'tileable_cases': list(set(tileable_mod_2))
    })

    patterns['observations'].append({
        'type': 'modulo_3',
        'tileable_cases': list(set(tileable_mod_3))
    })

    # Observation 2: Check if both dimensions must be >= some threshold
    min_tileable_dim = min([min(r['m'], r['n']) for r in tileable]) if tileable else None
    patterns['min_tileable_dimension'] = min_tileable_dim

    # Observation 3: Check specific patterns
    # Can we tile any 3×2k rectangles?
    three_by_even = [r for r in tileable if (r['m'] == 3 or r['n'] == 3)]
    patterns['3×2k_rectangles'] = three_by_even

    # Can we tile any 2×3k rectangles?
    two_by_mult3 = [r for r in tileable if (r['m'] == 2 or r['n'] == 2)]
    patterns['2×3k_rectangles'] = two_by_mult3

    return patterns

def main():
    """Run the simulation and generate results."""
    print("IMO 2004 Problem 3: Hook Tiling Simulation")
    print("=" * 50)

    # Display hook shapes
    print("\nHook shapes (4 unique orientations):")
    for i, hook in enumerate(HOOKS):
        print(f"\nOrientation {i+1}:")
        print(visualize_hook(hook))

    # Analyze small rectangles
    print("\nAnalyzing rectangles up to 10×10...")
    results = analyze_small_rectangles(max_size=10)

    # Find patterns
    print("\nAnalyzing patterns...")
    patterns = find_patterns(results)

    # Create summary
    summary = {
        'problem': 'Determine all m×n rectangles that can be tiled with hooks',
        'hook_description': 'A hook is made of 6 unit squares in an L-shape',
        'total_tested': len(results['tileable']) + len(results['not_tileable']),
        'tileable_count': len(results['tileable']),
        'not_tileable_count': len(results['not_tileable']),
        'tileable_rectangles': results['tileable'],
        'not_tileable_rectangles': results['not_tileable'],
        'patterns': patterns,
        'conjecture': 'Based on the simulation, it appears that NO m×n rectangle can be tiled with hooks of this shape. The answer may be that no such rectangles exist, or only very specific large rectangles can be tiled.'
    }

    # Save to JSON
    output_file = 'results.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total rectangles tested: {summary['total_tested']}")
    print(f"  Tileable: {summary['tileable_count']}")
    print(f"  Not tileable: {summary['not_tileable_count']}")

    print("\nTileable rectangles:")
    for r in results['tileable']:
        print(f"  {r['m']}×{r['n']} ({r['num_hooks']} hooks)")

    print("\nConjectured answer:")
    print(f"  {summary['conjecture']}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 3:
        # Test a specific rectangle
        try:
            m = int(sys.argv[1])
            n = int(sys.argv[2])
            print(f"Testing {m}×{n} rectangle...")
            print("=" * 50)

            result = check_tileability(m, n)

            print(f"\nRectangle: {m}×{n}")
            print(f"Area: {m * n}")
            print(f"Tileable: {result['tileable']}")
            print(f"Reason: {result['reason']}")

            if result['tileable'] and result['solutions']:
                print(f"\nFound a tiling with {result['num_hooks']} hooks!")
                print("\nTiling visualization:")
                tiler = HookTiler(m, n)
                # Apply the solution
                for placement in result['solutions'][0]:
                    hook = HOOKS[placement['orientation']]
                    for dr, dc in hook:
                        r, c = placement['row'] + dr, placement['col'] + dc
                        tiler.grid[r][c] = True
                print(visualize_grid(tiler.grid))

        except ValueError:
            print("Usage: python simulation.py [m] [n]")
            print("  or: python simulation.py (to run full analysis)")
    else:
        main()
