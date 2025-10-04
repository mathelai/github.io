"""
IMO 2012 Problem 3: The Liar's Guessing Game Simulation

This simulation implements the liar's guessing game where:
- Player A chooses a secret integer x (1 ≤ x ≤ N)
- Player B asks questions about sets S: "Is x in S?"
- Player A can lie, but among any k+1 consecutive answers, at least one must be truthful
- Player B wins if they can identify x in a final guess set X of size ≤ n

The problem asks to prove:
(a) If n ≥ 2^k, then B can guarantee a win
(b) For large k, there exists n ≥ (1.99)^k where B cannot guarantee a win
"""

import json
import itertools
from typing import List, Set, Tuple, Dict, Any
from collections import deque


class LiarsGame:
    """Simulates the liar's guessing game."""

    def __init__(self, k: int, N: int, x: int):
        """
        Initialize the game.

        Args:
            k: Maximum number of consecutive lies allowed is k (at least 1 truth in k+1 answers)
            N: Upper bound on the secret number
            x: The secret number (1 ≤ x ≤ N)
        """
        self.k = k
        self.N = N
        self.x = x
        self.answers = []  # History of answers given by A
        self.questions = []  # History of questions asked by B

    def can_lie(self, current_set: Set[int]) -> bool:
        """
        Check if player A can lie about the current question without violating the lying constraint.

        Among any k+1 consecutive answers, at least one must be truthful.

        Args:
            current_set: The set being queried in the current question

        Returns:
            True if A can lie, False if A must tell the truth
        """
        # If we haven't answered k questions yet, we can lie
        if len(self.answers) < self.k:
            return True

        # Check the last k answers - if all were lies, this one must be truthful
        recent_answers = self.answers[-self.k:]
        recent_questions = self.questions[-self.k:]

        # Count how many of the last k answers were lies
        lies_count = 0
        for i in range(len(recent_answers)):
            true_ans = self.x in recent_questions[i]
            if recent_answers[i] != true_ans:
                lies_count += 1

        # If all k previous answers were lies, this one must be truthful
        return lies_count < self.k

    def ask_question(self, S: Set[int], answer: bool = None) -> bool:
        """
        Player B asks if x is in set S. Player A answers (possibly lying).

        Args:
            S: The set being queried
            answer: If provided, A tries to give this answer (if allowed). Otherwise, A answers adversarially.

        Returns:
            The answer given by A
        """
        true_answer = self.x in S

        if answer is None:
            # Adversarial strategy: lie if possible
            if self.can_lie(S):
                answer = not true_answer
            else:
                answer = true_answer
        else:
            # Check if the desired answer is allowed
            if answer != true_answer and not self.can_lie(S):
                # Cannot lie, must give truthful answer
                answer = true_answer

        self.questions.append(S)
        self.answers.append(answer)
        return answer

    def reset_history(self):
        """Reset the game history."""
        self.answers = []
        self.questions = []


def binary_search_strategy(k: int, N: int) -> Tuple[List[Set[int]], int]:
    """
    Implement Player B's binary search strategy.

    This strategy works when n ≥ 2^k. The idea is to use a binary search
    approach that handles potential lies.

    Returns:
        (questions, max_final_set_size): The questions to ask and the maximum final set size needed
    """
    questions = []

    # For simplicity, we'll use a halving strategy
    # We need to narrow down from N possibilities to at most 2^k

    # Number of questions needed: we ask about lower/upper halves repeatedly
    # After each question, we can eliminate some possibilities
    # The strategy is to ask log2(N/2^k) * (k+1) questions to be safe

    current_range = set(range(1, N + 1))
    questions_to_ask = []

    # Simple halving: split the range in half repeatedly
    while len(current_range) > 1:
        sorted_range = sorted(current_range)
        mid = len(sorted_range) // 2
        lower_half = set(sorted_range[:mid])

        if lower_half:
            questions_to_ask.append(lower_half)
            current_range = lower_half if len(lower_half) > 1 else set(sorted_range[mid:])
        else:
            break

    return questions_to_ask, 2 ** k


def determine_possible_values(k: int, N: int, questions: List[Set[int]], answers: List[bool]) -> Set[int]:
    """
    Given a sequence of questions and answers, determine which values of x are consistent
    with the lying constraint (at least 1 truth in every k+1 consecutive answers).

    Returns:
        Set of possible values for x
    """
    possible = set()

    for x_candidate in range(1, N + 1):
        # Check if this candidate is consistent with all answers
        is_consistent = True

        # Check every window of k+1 consecutive answers
        for start_idx in range(len(answers) - k):
            window_answers = answers[start_idx:start_idx + k + 1]
            window_questions = questions[start_idx:start_idx + k + 1]

            # Count truths in this window for this candidate
            truths = 0
            for i in range(len(window_answers)):
                true_answer = x_candidate in window_questions[i]
                if window_answers[i] == true_answer:
                    truths += 1

            if truths == 0:
                # This window has no truths, violates constraint
                is_consistent = False
                break

        if is_consistent:
            possible.add(x_candidate)

    return possible


def optimal_strategy_b(k: int, N: int) -> Tuple[List[Set[int]], int]:
    """
    Implement an optimal strategy for player B.

    The key insight for part (a): B can use a strategy where each question
    reduces the number of possible values. After asking enough questions,
    B can narrow down to at most 2^k possibilities.

    Strategy: Ask questions about carefully chosen sets to maximize information gain
    even with lies.
    """
    questions = []

    # We'll use a base-2 representation strategy
    # Ask k+1 questions for each bit position to ensure we get at least one truth

    num_bits = 0
    temp = N - 1
    while temp > 0:
        num_bits += 1
        temp //= 2

    # For each bit position, ask k+1 times about numbers with that bit set
    for bit in range(num_bits):
        for _ in range(k + 1):
            # Create set of numbers with this bit set in their binary representation
            S = set()
            for num in range(1, N + 1):
                if (num - 1) & (1 << bit):
                    S.add(num)
            if S:
                questions.append(S)

    return questions, 2 ** k


def simulate_game(k: int, N: int, x: int, strategy: str = "binary") -> Dict[str, Any]:
    """
    Simulate a complete game.

    Args:
        k: Lying parameter
        N: Upper bound
        x: Secret number
        strategy: Which strategy B should use ("binary" or "optimal")

    Returns:
        Dictionary with simulation results
    """
    game = LiarsGame(k, N, x)

    if strategy == "binary":
        questions, max_final_size = binary_search_strategy(k, N)
    else:
        questions, max_final_size = optimal_strategy_b(k, N)

    answers = []
    for S in questions:
        answer = game.ask_question(S)
        answers.append(answer)

    # Determine possible values based on answers
    possible = determine_possible_values(k, N, questions, answers)

    b_wins = len(possible) <= max_final_size and x in possible

    return {
        "k": k,
        "N": N,
        "x": x,
        "num_questions": len(questions),
        "questions": [list(S) for S in questions],
        "answers": answers,
        "possible_values": sorted(list(possible)),
        "final_set_size": len(possible),
        "max_allowed_size": max_final_size,
        "b_wins": b_wins,
        "strategy": strategy
    }


def analyze_part_a(k_values: List[int] = None) -> List[Dict[str, Any]]:
    """
    Analyze part (a): If n ≥ 2^k, then B can guarantee a win.

    Test for various small values of k and N.
    """
    if k_values is None:
        k_values = [1, 2, 3, 4]

    results = []

    for k in k_values:
        n_threshold = 2 ** k

        # Test with N values around the threshold
        test_N_values = [n_threshold, n_threshold + 1, n_threshold + 5]

        for N in test_N_values:
            if N > 100:  # Skip very large values for simulation
                continue

            # Test with several x values
            test_x_values = [1, N // 2, N] if N > 1 else [1]

            for x in test_x_values:
                result = simulate_game(k, N, x, strategy="optimal")
                results.append(result)

                # For small k, verify all possible x values
                if k <= 3 and N <= 20:
                    wins = 0
                    for test_x in range(1, N + 1):
                        test_result = simulate_game(k, N, test_x, strategy="optimal")
                        if test_result["b_wins"]:
                            wins += 1

                    result["all_x_tested"] = True
                    result["total_wins"] = wins
                    result["total_tests"] = N
                    result["win_rate"] = wins / N

    return results


def analyze_part_b(k_values: List[int] = None) -> List[Dict[str, Any]]:
    """
    Analyze part (b): For large k, exists n ≥ (1.99)^k where B cannot guarantee a win.

    This is harder to simulate but we can explore small cases.
    """
    if k_values is None:
        k_values = [2, 3, 4, 5]

    results = []

    for k in k_values:
        lower_bound = int((1.99) ** k)
        n_threshold = 2 ** k

        # For part b, we want to find n values between lower_bound and n_threshold
        # where B might not be able to win

        result = {
            "k": k,
            "lower_bound_1.99^k": lower_bound,
            "threshold_2^k": n_threshold,
            "gap": n_threshold - lower_bound,
            "ratio": n_threshold / lower_bound if lower_bound > 0 else float('inf')
        }

        results.append(result)

    return results


def generate_ground_truth_data() -> Dict[str, Any]:
    """
    Generate comprehensive ground truth data for the problem.
    """
    print("Generating ground truth data for IMO 2012 Problem 3...")

    # Part (a) analysis
    print("\nAnalyzing part (a)...")
    part_a_results = analyze_part_a([1, 2, 3, 4])

    # Part (b) analysis
    print("Analyzing part (b)...")
    part_b_results = analyze_part_b([2, 3, 4, 5, 6, 7, 8])

    # Small examples for visualization
    print("\nGenerating visualization examples...")
    examples = []

    # Example 1: k=1, N=4, showing B can win with n=2
    for x in range(1, 5):
        examples.append(simulate_game(k=1, N=4, x=x, strategy="optimal"))

    # Example 2: k=2, N=8
    for x in [1, 4, 8]:
        examples.append(simulate_game(k=2, N=8, x=x, strategy="optimal"))

    # Example 3: k=3, N=16
    examples.append(simulate_game(k=3, N=16, x=10, strategy="optimal"))

    # Compute bounds for various k
    print("\nComputing theoretical bounds...")
    bounds = []
    for k in range(1, 11):
        bounds.append({
            "k": k,
            "2^k": 2 ** k,
            "1.99^k": (1.99) ** k,
            "gap": 2 ** k - (1.99) ** k
        })

    ground_truth = {
        "problem": "IMO 2012 Problem 3: The Liar's Guessing Game",
        "part_a": {
            "statement": "If n >= 2^k, then B can guarantee a win",
            "results": part_a_results
        },
        "part_b": {
            "statement": "For large k, exists n >= (1.99)^k where B cannot guarantee a win",
            "results": part_b_results
        },
        "examples": examples,
        "theoretical_bounds": bounds,
        "insights": [
            "Part (a) shows that 2^k possibilities can always be distinguished with the lying constraint",
            "The key is that in k+1 questions, at least one answer is truthful",
            "Part (b) shows there's a gap between (1.99)^k and 2^k where the problem is non-trivial",
            "For small k, 2^k grows faster than 1.99^k, creating a significant gap"
        ]
    }

    return ground_truth


def main():
    """Main function to run the simulation and generate results."""
    ground_truth = generate_ground_truth_data()

    # Save to JSON
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nTheoretical Bounds (first 10 values of k):")
    print(f"{'k':<5} {'2^k':<10} {'1.99^k':<15} {'Gap':<15}")
    print("-" * 50)
    for bound in ground_truth["theoretical_bounds"]:
        print(f"{bound['k']:<5} {bound['2^k']:<10} {bound['1.99^k']:<15.2f} {bound['gap']:<15.2f}")

    print("\nPart (a) Sample Results:")
    for result in ground_truth["part_a"]["results"][:5]:
        win_status = "WIN" if result["b_wins"] else "LOSS"
        print(f"k={result['k']}, N={result['N']}, x={result['x']}: "
              f"{result['final_set_size']} possibilities (max allowed: {result['max_allowed_size']}) - {win_status}")

    print("\nPart (b) Analysis:")
    for result in ground_truth["part_b"]["results"][:5]:
        print(f"k={result['k']}: (1.99)^k ≈ {result['lower_bound_1.99^k']:.0f}, "
              f"2^k = {result['threshold_2^k']}, gap = {result['gap']:.0f}")

    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
