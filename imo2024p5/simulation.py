"""
IMO 2024 Problem 5: Turbo the Snail Simulation

This simulation analyzes the game where Turbo must navigate a grid with hidden monsters.

Problem Setup:
- Board: 2024 rows × 2023 columns
- Monsters: 2022 total (one per row, except first and last row)
- Each column has at most one monster
- Turbo starts in first row, must reach last row
- Each attempt ends when hitting a monster or reaching the last row

Goal: Find minimum n such that Turbo can guarantee reaching the last row within n attempts.
"""

import json
import itertools
from typing import List, Tuple, Set, Dict
from collections import defaultdict


class TurboGame:
    """Simulates Turbo's game with configurable board size."""

    def __init__(self, rows: int, cols: int):
        """
        Initialize game board.

        Args:
            rows: Number of rows (must be >= 2)
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.num_monsters = rows - 2  # One per row except first and last
        self.monster_rows = list(range(1, rows - 1))  # Rows 1 to rows-2 (0-indexed)

    def is_valid_monster_config(self, monsters: List[Tuple[int, int]]) -> bool:
        """
        Check if monster configuration is valid.

        Args:
            monsters: List of (row, col) positions

        Returns:
            True if valid configuration
        """
        if len(monsters) != self.num_monsters:
            return False

        # Check each monster is in correct row range
        rows_used = set()
        cols_used = set()
        for r, c in monsters:
            if r not in self.monster_rows:
                return False
            if c < 0 or c >= self.cols:
                return False
            if r in rows_used:  # Only one monster per row
                return False
            if c in cols_used:  # At most one monster per column
                return False
            rows_used.add(r)
            cols_used.add(c)

        return len(rows_used) == self.num_monsters

    def generate_all_valid_configs(self) -> List[List[Tuple[int, int]]]:
        """
        Generate all valid monster configurations.
        For large boards, this is infeasible, so we sample.
        """
        configs = []

        # Each monster row must have exactly one monster
        # We need to assign columns to each monster row
        # Such that no column is used twice

        # This is equivalent to finding permutations of columns
        if self.num_monsters > self.cols:
            return []  # Impossible configuration

        # For small boards, enumerate all
        if self.num_monsters <= 8:  # Computational limit
            from itertools import permutations
            for col_perm in permutations(range(self.cols), self.num_monsters):
                config = [(self.monster_rows[i], col_perm[i])
                         for i in range(self.num_monsters)]
                configs.append(config)

        return configs

    def simulate_strategy_vertical_sweep(self, monsters: List[Tuple[int, int]]) -> int:
        """
        Simulate vertical sweep strategy: try each column sequentially.

        Args:
            monsters: Monster configuration

        Returns:
            Number of attempts needed
        """
        monster_set = set(monsters)

        for attempt in range(1, self.cols + 2):
            # Try column (attempt - 1) % cols
            col = (attempt - 1) % self.cols

            # Walk down this column
            blocked = False
            for row in range(self.rows):
                if (row, col) in monster_set:
                    blocked = True
                    break
                if row == self.rows - 1:
                    # Reached last row!
                    return attempt

        # Should never reach here if strategy is correct
        return self.cols + 1

    def simulate_strategy_binary_search(self, monsters: List[Tuple[int, int]]) -> int:
        """
        Simulate binary search strategy on columns.

        This is more sophisticated but may not be optimal for this problem.
        """
        monster_set = set(monsters)
        known_safe = set()
        known_blocked = set()

        attempts = 0

        # First, try to find a safe column using binary search approach
        left, right = 0, self.cols - 1

        while attempts < self.cols * 2:
            attempts += 1

            # Try middle column
            col = (left + right) // 2 if left <= right else attempts % self.cols

            if col in known_safe:
                return attempts
            if col in known_blocked:
                left = col + 1
                continue

            # Walk down this column
            blocked = False
            for row in range(self.rows):
                if (row, col) in monster_set:
                    blocked = True
                    known_blocked.add(col)
                    break
                if row == self.rows - 1:
                    # Reached last row!
                    return attempts

            if not blocked:
                known_safe.add(col)
                return attempts

        return attempts

    def find_worst_case_for_strategy(self, strategy_name: str) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Find worst-case number of attempts for a given strategy.

        Args:
            strategy_name: Name of strategy to test

        Returns:
            (max_attempts, worst_config)
        """
        configs = self.generate_all_valid_configs()

        if not configs:
            return (0, [])

        max_attempts = 0
        worst_config = None

        for config in configs:
            if strategy_name == "vertical_sweep":
                attempts = self.simulate_strategy_vertical_sweep(config)
            elif strategy_name == "binary_search":
                attempts = self.simulate_strategy_binary_search(config)
            else:
                attempts = 0

            if attempts > max_attempts:
                max_attempts = attempts
                worst_config = config

        return (max_attempts, worst_config)


def analyze_small_cases():
    """Analyze small board sizes to find patterns."""
    results = {
        "small_cases": [],
        "pattern_analysis": {},
        "theoretical_bound": {}
    }

    # Test small boards
    test_cases = [
        (3, 2),   # 3 rows, 2 cols -> 1 monster
        (4, 3),   # 4 rows, 3 cols -> 2 monsters
        (5, 4),   # 5 rows, 4 cols -> 3 monsters
        (6, 5),   # 6 rows, 5 cols -> 4 monsters
        (7, 6),   # 7 rows, 6 cols -> 5 monsters
        (8, 7),   # 8 rows, 7 cols -> 6 monsters
    ]

    for rows, cols in test_cases:
        game = TurboGame(rows, cols)

        print(f"\nAnalyzing {rows} rows × {cols} cols ({game.num_monsters} monsters)...")

        # Find worst case for vertical sweep
        max_attempts_vs, worst_config_vs = game.find_worst_case_for_strategy("vertical_sweep")

        # Calculate theoretical bounds
        # Lower bound: At least 1 attempt
        # Upper bound: In worst case, might need to try all columns
        # But actually, since there are cols columns and num_monsters < cols,
        # at least one column must be monster-free (pigeonhole principle)

        num_safe_cols = cols - game.num_monsters

        result = {
            "rows": rows,
            "cols": cols,
            "num_monsters": game.num_monsters,
            "num_safe_columns": num_safe_cols,
            "worst_case_attempts": max_attempts_vs,
            "worst_config": worst_config_vs,
            "total_configs": len(game.generate_all_valid_configs())
        }

        results["small_cases"].append(result)

        print(f"  Safe columns: {num_safe_cols}")
        print(f"  Worst case attempts (vertical sweep): {max_attempts_vs}")
        print(f"  Total configurations: {result['total_configs']}")

    # Pattern analysis
    print("\n=== PATTERN ANALYSIS ===")

    # Check if worst case = cols - num_safe_cols + 1
    pattern_holds = True
    for case in results["small_cases"]:
        cols = case["cols"]
        num_safe = case["num_safe_columns"]
        worst = case["worst_case_attempts"]

        # Expected: In worst case, we try (cols - num_safe_cols + 1) columns
        # because we might hit monsters in first (cols - num_safe_cols) attempts
        # and succeed on the next one
        expected = cols - num_safe + 1

        print(f"Rows={case['rows']}, Cols={cols}, Safe={num_safe}, "
              f"Worst={worst}, Expected={expected}, Match={worst==expected}")

        if worst != expected:
            pattern_holds = False

    results["pattern_analysis"]["formula"] = "worst_case = cols - num_safe_cols + 1"
    results["pattern_analysis"]["formula_holds"] = pattern_holds

    return results


def analyze_imo_problem():
    """Analyze the actual IMO problem (2024 rows × 2023 cols)."""
    rows = 2024
    cols = 2023
    num_monsters = rows - 2  # 2022

    print("\n=== IMO 2024 PROBLEM 5 ANALYSIS ===")
    print(f"Board: {rows} rows × {cols} columns")
    print(f"Monsters: {num_monsters}")

    # Number of safe columns (columns with no monsters)
    num_safe_cols = cols - num_monsters
    print(f"Safe columns (no monsters): {num_safe_cols}")

    # Theoretical analysis:
    # Strategy: Try columns in order (0, 1, 2, ..., cols-1)
    # In each attempt, go straight down the column
    # If hit monster, that column is blocked
    # If reach bottom, success!

    # Worst case: All safe columns are at the end
    # We might try (cols - num_safe_cols) blocked columns first
    # Then try a safe column and succeed

    worst_case_attempts = cols - num_safe_cols + 1

    print(f"\nVertical sweep strategy:")
    print(f"  Worst case: {worst_case_attempts} attempts")
    print(f"  Logic: Try up to {cols - num_safe_cols} blocked columns, then hit a safe one")

    # Alternative analysis:
    # Since num_safe_cols = cols - num_monsters = 2023 - 2022 = 1
    # There is exactly 1 column with no monsters!

    print(f"\nKey insight:")
    print(f"  There is exactly {num_safe_cols} column(s) with no monsters")
    print(f"  Worst case: this column is the last one we try")
    print(f"  Therefore: n = {worst_case_attempts}")

    result = {
        "rows": rows,
        "cols": cols,
        "num_monsters": num_monsters,
        "num_safe_columns": num_safe_cols,
        "minimum_n": worst_case_attempts,
        "strategy": "vertical_sweep",
        "explanation": f"Try each column in sequence. In worst case, the {num_safe_cols} safe column(s) "
                      f"are the last ones tried, requiring {worst_case_attempts} attempts."
    }

    return result


def verify_answer():
    """
    Verify the answer with mathematical reasoning.
    """
    print("\n=== VERIFICATION ===")

    cols = 2023
    num_monsters = 2022
    num_safe_cols = cols - num_monsters  # = 1

    print(f"Given: {cols} columns, {num_monsters} monsters")
    print(f"Since each column has at most one monster:")
    print(f"  Number of safe columns >= {cols} - {num_monsters} = {num_safe_cols}")

    print(f"\nPigeonhole principle:")
    print(f"  {num_monsters} monsters placed in {cols} columns")
    print(f"  At least {num_safe_cols} column(s) must be monster-free")

    print(f"\nStrategy: Try columns sequentially")
    print(f"  Best case: Hit safe column on attempt 1")
    print(f"  Worst case: Hit safe column on attempt {cols}")
    print(f"  Guaranteed success by attempt: {cols}")

    print(f"\nBut we can do better:")
    print(f"  Since exactly {num_safe_cols} column is safe,")
    print(f"  and {num_monsters} columns have monsters,")
    print(f"  we need at most {num_monsters + 1} attempts")
    print(f"  (try {num_monsters} blocked columns, then the {num_monsters + 1}th must be safe)")

    answer = num_monsters + 1

    print(f"\n*** ANSWER: n = {answer} ***")

    return {
        "answer": answer,
        "reasoning": "By pigeonhole principle, at least one column is monster-free. "
                    "In worst case, we try all monster-containing columns first, "
                    f"then succeed on attempt {answer}."
    }


def main():
    """Main simulation execution."""
    print("=" * 70)
    print("IMO 2024 Problem 5: Turbo the Snail")
    print("=" * 70)

    # Analyze small cases to find patterns
    small_case_results = analyze_small_cases()

    # Analyze the actual IMO problem
    imo_result = analyze_imo_problem()

    # Verify the answer
    verification = verify_answer()

    # Compile all results
    all_results = {
        "problem": {
            "rows": 2024,
            "cols": 2023,
            "num_monsters": 2022
        },
        "answer": verification["answer"],
        "reasoning": verification["reasoning"],
        "small_cases": small_case_results["small_cases"],
        "pattern": small_case_results["pattern_analysis"],
        "imo_analysis": imo_result
    }

    # Save to JSON
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    main()
