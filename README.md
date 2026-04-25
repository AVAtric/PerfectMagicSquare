# Memetic Algorithm: Solving Magic Squares

A high-performance C++ implementation for generating magic squares of sizes 3 to 9.
Uses a memetic algorithm (genetic algorithm + steepest-descent local search) with OpenMP parallelization to find solutions fast.

## Build

Requires a C++20 compiler and OpenMP support. No third-party headers — the colored grid is rendered with raw ANSI escape codes.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

On macOS with Homebrew:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

## Usage

```bash
./bin/PerfectMagicSquare -d <dimension> -p <population> -i <iterations> [-o <output>] [-v] [-s]
```

| Flag | Description | Range |
|------|-------------|-------|
| `-d` | Square dimension | 3 - 9 |
| `-p` | Population size | 1000 - 10000 |
| `-i` | Max iterations (-1 = infinite) | 1000 - 100000 |
| `-o` | Output CSV file name (without extension) | - |
| `-v` | Verbose mode | - |
| `-s` | Silent mode | - |
| `-h` | Help | - |

Example:
```bash
./bin/PerfectMagicSquare -d 5 -p 5000 -i 10000
```

## Tests

Run all tests via CTest:
```bash
cd build
ctest --output-on-failure
```

Two binaries exist per square size (3-9): a correctness test and a live visual test. Each is a one-line stub that calls into `tests/test_helpers.h`.
```bash
./bin/PerfectMagicSquareTestsThree         # correctness, validates 3x3 solution
./bin/PerfectMagicSquareTestsFiveVisual    # live animated 5x5 run
```

The visual tests redraw the current best candidate in place every ~100 ms with per-cell coloring (green = the cell's row/column/diagonal sums are correct, red = at least one is off), plus generation, fitness, mutation rate and stagnation counters. Run them in a real terminal — piping the output strips the terminal-control sequences. Larger sizes (7-9) make the convergence pattern most visible.

## Algorithm

The solver is a **memetic algorithm** combining a genetic algorithm for global exploration with steepest-descent local search for fast local convergence.

### Key components

- **Representation**: Each candidate is a permutation of 1..n*n arranged in an n x n grid.
- **Fitness**: Sum of absolute deviations from the magic sum across all rows, columns, and both diagonals. A fitness of 0 means a valid magic square.
- **Selection**: Tournament selection (size 2) for parent picking in crossover.
- **Crossover**: Cell-wise selection from the fitter parent (per row/column fitness), with deterministic fill of missing values.
- **Mutation**: Random swap of two cell values with adaptive probability that increases on stagnation.
- **Local search**: Exhaustive steepest-descent that evaluates all C(n*n, 2) possible swaps per round using O(1) delta fitness computation. Self-terminates at local optima.
- **Elitism**: Top 10% of the population is preserved across generations.
- **Catastrophic restart**: After 30 generations of stagnation, the population is regenerated (keeping elite) to escape local optima.

### Performance optimizations

- **Flat array storage** (row-major `vector<int>`) instead of `vector<vector<int>>` for cache-friendly access.
- **Cached row/column/diagonal sums** enable O(1) fitness queries and incremental updates after swaps.
- **O(1) swap evaluation** in local search via delta computation on cached sums (no full fitness recompute).
- **O(1) value existence check** in crossover using a boolean array instead of O(n*n) linear scan.
- **No expensive deduplication**: Removed all `std::find()` calls over population vectors (was O(pop * n*n) per call).
- **Hoisted parent fitness in crossover**: per-row/column fitness values are precomputed once per parent instead of n*n times per pair.
- **OpenMP parallelization** of crossover, mutation, and local search phases.

### Benchmark results

Measured on Apple M1 Max (Release build):

| Size | Population | Iterations | Time |
|------|-----------|------------|------|
| 3x3 | 1,000 | 1,000 | < 1 ms |
| 4x4 | 10,000 | 10,000 | ~3 ms |
| 5x5 | 10,000 | 10,000 | ~1.4 s |
| 6x6 | 10,000 | 100,000 | ~0.3 s |
| 7x7 | 10,000 | 100,000 | ~3 s |
| 8x8 | 10,000 | 100,000 | ~1.5 s |
| 9x9 | 10,000 | 100,000 | ~6 s |

Times vary between runs due to the stochastic nature of the algorithm.

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── magic_square.h       # MagicSquare class and solver declarations
│   └── program_options.h    # CLI argument parsing
├── src/
│   ├── magic_square.cpp     # Core algorithm + ANSI grid renderer
│   ├── program_options.cpp  # CLI argument parsing
│   └── main.cpp             # Entry point
└── tests/
    ├── CMakeLists.txt
    ├── test_helpers.h                                                 # shared validate/runCorrectnessTest/runVisualTest
    ├── square_three_test.cpp  ... square_nine_test.cpp                # correctness tests (1-line stubs)
    └── square_three_visual_test.cpp  ... square_nine_visual_test.cpp  # live animated runs (1-line stubs)
```

## License

Released under the [MIT License](LICENSE).
