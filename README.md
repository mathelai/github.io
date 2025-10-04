# DRAFT TEMPLATE ONLY
# MathEL: Autonomous Self-Improving AI for IMO Combinatorics

A comprehensive collection of simulations, visualizations, and analyses for 39 International Mathematical Olympiad (IMO) combinatorics problems spanning 2000-2025.

## Overview

This repository provides:
- **Python simulations** that explore small instances, find patterns, and generate ground truth data
- **Interactive web visualizations** for each problem (work offline, no server required)
- **Proofs and explanations** for each problem
- **Unified results schema** for reproducible research

## Project Structure

```
app/
├── index.html              # Main problem browser UI
├── dataset.js              # Problem metadata (308 KB)
├── README.md               # This file
├── LICENSE                 # MIT License
├── CITATION.cff            # Citation metadata
├── pyproject.toml          # Python dependencies
│
├── schemas/
│   └── results.schema.json # Unified results schema
│
└── imo{YYYY}p{N}/          # 39 problem directories
    ├── problem.txt         # Problem statement (LaTeX)
    ├── index.html          # Interactive visualization
    ├── simulation.py       # Python simulation code
    ├── results.json        # Simulation results
    ├── proof.txt           # Proof/explanation
    └── grade.txt           # Grading information
```

## Quick Start

### Browse Problems

Simply open `index.html` in your web browser:
- Browse all 39 problems with search and filters
- View problem metadata (year, difficulty, category)
- Click any problem card to open its interactive visualization

### View Individual Problems

Each problem directory contains:
- `index.html` - Interactive visualization (open directly in browser)
- `problem.txt` - Problem statement
- `proof.txt` - Proof/explanation
- `simulation.py` - Python simulation code
- `results.json` - Pre-computed results

### Run Simulations (Optional)

If you want to run simulations:

```bash
# Install dependencies
pip install numpy pydantic

# Run a simulation
cd imo2025p1
python simulation.py
```

No server required - all visualizations work offline in the browser.

## Problem Coverage

### By Year
- **2025**: 2 problems
- **2024**: 2 problems
- **2023**: 1 problem
- **2022**: 2 problems (P1, P6)
- **2021**: 1 problem
- **2020**: 2 problems
- **2019**: 2 problems
- **2018**: 2 problems
- **2017**: 2 problems
- **2016**: 2 problems
- **2015**: 2 problems
- **2014**: 2 problems
- **2013**: 2 problems
- **2012**: 1 problem
- **2011**: 2 problems
- **2010**: 1 problem
- **2009**: 1 problem
- **2008**: 1 problem
- **2007**: 1 problem
- **2006**: 1 problem
- **2005**: 1 problem
- **2004**: 1 problem
- **2001**: 2 problems
- **2000**: 1 problem

### By Category
All problems are **combinatorics** with subcategories including:
- Combinatorial Geometry
- Graph Theory
- Combinatorial Game Theory
- Extremal Combinatorics
- Combinatorial Number Theory

## Proof Status

**Important**: This repository contains a mix of:
- ✅ **Experimental evidence** - Verified for small cases (labeled as "Experimental")
- 🟡 **Sketches** - Partial proofs with key ideas (labeled as "Sketch")
- ⚠️ **Unverified claims** - Require mathematical review

### Recent Corrections
- **IMO 2025 P1**: Corrected answer is k ∈ {0, 1, n-2} (previously incorrectly claimed as k ∈ {0, 1}). Simulation and proof have been updated to reflect the correct triangular pattern construction.

## Results Schema

All `results.json` files conform to `schemas/results.schema.json` with standardized fields:
- `problem_id` - Problem identifier (e.g., "IMO-2022-P1")
- `params` - Input configuration
- `method` - Algorithm name/version
- `metrics` - Problem-specific outputs
- `seed` - Random seed for reproducibility
- `environment` - Python version, platform, timestamp
- `status` - Success/timeout/failed/partial

## Citation

If you use this software in your research, please cite:

```bibtex
@software{mathel2025,
  title = {MathEL: Autonomous Self-Improving AI for IMO Combinatorics},
  author = {{MathEL Project}},
  year = {2025},
  url = {https://github.com/mathel/mathel},
  version = {0.1.0}
}
```

See `CITATION.cff` for machine-readable citation metadata.

## License

MIT License - see `LICENSE` file for details.

## Acknowledgments

Problem statements are from the International Mathematical Olympiad (IMO).
Original problems © International Mathematical Union.
