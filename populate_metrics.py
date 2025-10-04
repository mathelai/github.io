#!/usr/bin/env python3
"""
Systematically populate metrics in all results.json files.

This script:
1. Runs each simulation.py with timing
2. Extracts key metrics from the simulation
3. Updates results.json with populated metrics
"""

import json
import subprocess
import time
from pathlib import Path
import sys

def run_simulation_with_metrics(problem_dir: Path):
    """Run a simulation and extract metrics."""
    sim_file = problem_dir / "simulation.py"
    results_file = problem_dir / "results.json"

    if not sim_file.exists():
        return None, "No simulation.py"

    if not results_file.exists():
        return None, "No results.json"

    print(f"\n{'='*60}")
    print(f"Processing: {problem_dir.name}")
    print(f"{'='*60}")

    # Read existing results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        return None, f"Failed to read results.json: {e}"

    # Check if metrics already populated
    metrics = results.get('metrics', {})
    if metrics and len(metrics) > 1:  # More than just empty dict
        print(f"  ✓ Already has metrics: {list(metrics.keys())}")
        return results, "Already populated"

    # Run the simulation with timing
    print(f"  Running simulation...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(sim_file)],
            cwd=str(problem_dir),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if result.returncode != 0:
            print(f"  ⚠ Simulation failed with code {result.returncode}")
            print(f"  stderr: {result.stderr[:200]}")
            return None, f"Failed with code {result.returncode}"

        print(f"  ✓ Completed in {elapsed_ms:.0f}ms")

        # Re-read results after simulation
        with open(results_file, 'r') as f:
            updated_results = json.load(f)

        # Extract metrics from the output and results
        metrics = extract_metrics(updated_results, result.stdout, elapsed_ms)

        # Update results with metrics
        updated_results['metrics'] = metrics

        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(updated_results, f, indent=2)

        print(f"  ✓ Metrics populated: {list(metrics.keys())}")
        return updated_results, "Success"

    except subprocess.TimeoutExpired:
        print(f"  ⏱ Timeout after 120s")
        return None, "Timeout"
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, str(e)


def extract_metrics(results: dict, stdout: str, elapsed_ms: float) -> dict:
    """Extract metrics from simulation results and output."""
    metrics = {
        'runtime_ms': round(elapsed_ms, 2),
        'status': results.get('status', 'success'),
    }

    # Try to extract runs/trials from ground_truth or analysis
    ground_truth = results.get('ground_truth', {})
    analysis = results.get('analysis', {})
    params = results.get('params', {})

    # Look for test cases, runs, trials
    runs = []

    # Check for test_results or similar structures
    if isinstance(ground_truth, dict):
        for key in ['test_results', 'cases', 'trials', 'runs']:
            if key in ground_truth:
                data = ground_truth[key]
                if isinstance(data, list):
                    runs.extend(data)

    # Check for n_range or parameter ranges
    coverage = {}
    if 'n_range' in params:
        n_range = params['n_range']
        if isinstance(n_range, list) and len(n_range) >= 2:
            coverage['n'] = {'min': n_range[0], 'max': n_range[-1]}

    # Count successful runs
    if runs:
        metrics['runs_count'] = len(runs)
        failures = sum(1 for r in runs if isinstance(r, dict) and not r.get('success', True))
        metrics['failures'] = failures

    # Add coverage if found
    if coverage:
        metrics['coverage'] = coverage

    # Extract problem-specific metrics
    if 'pattern_verified' in ground_truth:
        metrics['pattern_verified'] = ground_truth['pattern_verified']

    if 'formula' in ground_truth:
        metrics['formula'] = ground_truth['formula']

    # Look for key statistics in analysis
    if isinstance(analysis, dict):
        for key in ['key_insight', 'observation', 'conclusion']:
            if key in analysis and isinstance(analysis[key], str):
                # Don't duplicate large text, just note it exists
                metrics[f'has_{key}'] = True

    # Parse stdout for additional metrics
    lines = stdout.split('\n')
    for line in lines:
        if 'tested' in line.lower() or 'cases' in line.lower():
            # Try to extract numbers
            words = line.split()
            for i, word in enumerate(words):
                if word.isdigit() and i > 0:
                    if 'case' in words[i-1].lower() or 'test' in words[i-1].lower():
                        metrics['cases_tested'] = int(word)

    return metrics


def main():
    """Process all problem directories."""
    app_dir = Path("/Users/idrori/develop/MathEL/app")

    results_summary = {
        'success': [],
        'already_populated': [],
        'failed': [],
        'timeout': [],
    }

    # Process all imo* directories
    for problem_dir in sorted(app_dir.glob("imo*")):
        if not problem_dir.is_dir():
            continue

        result, status = run_simulation_with_metrics(problem_dir)

        if status == "Already populated":
            results_summary['already_populated'].append(problem_dir.name)
        elif status == "Success":
            results_summary['success'].append(problem_dir.name)
        elif status == "Timeout":
            results_summary['timeout'].append(problem_dir.name)
        else:
            results_summary['failed'].append((problem_dir.name, status))

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Already populated: {len(results_summary['already_populated'])}")
    print(f"Successfully populated: {len(results_summary['success'])}")
    print(f"Timeouts: {len(results_summary['timeout'])}")
    print(f"Failed: {len(results_summary['failed'])}")

    if results_summary['success']:
        print(f"\nSuccessfully populated:")
        for name in results_summary['success']:
            print(f"  ✓ {name}")

    if results_summary['timeout']:
        print(f"\nTimeouts:")
        for name in results_summary['timeout']:
            print(f"  ⏱ {name}")

    if results_summary['failed']:
        print(f"\nFailed:")
        for name, reason in results_summary['failed']:
            print(f"  ✗ {name}: {reason}")


if __name__ == "__main__":
    main()
