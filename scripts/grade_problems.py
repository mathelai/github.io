#!/usr/bin/env python3
"""
Grade IMO problem proofs using OpenAI GPT-5 Pro or GPT-5.

This script:
1. Reads the grading prompt from prompts/grading.txt
2. Iterates through all imo* problem directories
3. For each problem, reads the required files and calls the OpenAI API
4. Saves grading results to grade.txt in each problem directory
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    exit(1)


def read_file(filepath: Path) -> Optional[str]:
    """Read a file and return its contents, or None if file doesn't exist or is empty."""
    try:
        if not filepath.exists():
            return None
        content = filepath.read_text(encoding='utf-8')
        return content if content.strip() else None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def grade_problem(client: OpenAI, grading_prompt: str, problem_dir: Path, model: str) -> Optional[Dict]:
    """Grade a single problem using the OpenAI API."""
    problem_id = problem_dir.name
    print(f"\nGrading {problem_id} with {model}...")

    # Read required files
    files = {
        'problem.txt': read_file(problem_dir / 'problem.txt'),
        'proof-shortlist.txt': read_file(problem_dir / 'proof-shortlist.txt'),
        'proof-gpt5pro.txt': read_file(problem_dir / 'proof-gpt5pro.txt'),
        'proof-deepthink.txt': read_file(problem_dir / 'proof-deepthink.txt')
    }

    # Check for missing files
    missing_files = [name for name, content in files.items() if content is None]
    if missing_files:
        print(f"  Warning: Missing or empty files: {', '.join(missing_files)}")

    # Construct the prompt with all file contents
    full_prompt = f"{grading_prompt}\n\n"
    full_prompt += "=" * 80 + "\n\n"

    for filename, content in files.items():
        full_prompt += f"# {filename}\n"
        if content is None:
            full_prompt += "[FILE MISSING OR EMPTY]\n"
        else:
            full_prompt += f"{content}\n"
        full_prompt += "\n" + "=" * 80 + "\n\n"

    try:
        # Call OpenAI API using the responses endpoint
        system_message = "You are an expert IMO problem grader. Evaluate proofs carefully and return valid JSON only."
        full_input = f"{system_message}\n\n{full_prompt}"

        response = client.responses.create(
            model=model,
            input=full_input
        )

        # Parse the response
        result = json.loads(response.output.content)
        print(f"  ✓ Grading complete")
        return result

    except Exception as e:
        print(f"  ✗ Error grading: {e}")
        return None


def main():
    """Main function to grade all problems."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Grade IMO problem proofs using OpenAI GPT-5.')
    parser.add_argument('-gpt5', action='store_true',
                        help='Use GPT-5 model instead of GPT-5 Pro (default is GPT-5 Pro)')
    args = parser.parse_args()

    # Determine which model to use
    model = "gpt-5" if args.gpt5 else "gpt-5-pro"
    print(f"Using model: {model}")

    # Setup paths
    app_dir = Path(__file__).parent.parent
    prompts_dir = app_dir / 'prompts'
    grading_prompt_file = prompts_dir / 'grading.txt'

    # Read grading prompt
    if not grading_prompt_file.exists():
        print(f"Error: Grading prompt not found at {grading_prompt_file}")
        return

    grading_prompt = grading_prompt_file.read_text(encoding='utf-8')
    print(f"Loaded grading prompt from {grading_prompt_file}")

    # Initialize OpenAI client
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    # Find all problem directories
    problem_dirs = sorted(glob.glob(str(app_dir / 'imo*')))
    problem_dirs = [Path(d) for d in problem_dirs if Path(d).is_dir()]

    print(f"\nFound {len(problem_dirs)} problem directories")

    # Grade each problem
    results_summary = {}
    for problem_dir in problem_dirs:
        result = grade_problem(client, grading_prompt, problem_dir, model)

        if result:
            # Save result to problem directory
            output_file = problem_dir / 'grade.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved results to {output_file}")

            # Track summary
            results_summary[problem_dir.name] = {
                'gpt5pro_grade': result.get('gpt5pro', {}).get('grade'),
                'gpt5pro_comment': result.get('gpt5pro', {}).get('comment'),
                'deepthink_grade': result.get('deepthink', {}).get('grade'),
                'deepthink_comment': result.get('deepthink', {}).get('comment')
            }

    # Save summary
    summary_file = app_dir / 'grade-summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Grading complete! Summary saved to {summary_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
