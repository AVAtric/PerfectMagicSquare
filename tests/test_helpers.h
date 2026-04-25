//
// Shared helpers for the per-size correctness and visual tests.
//

#ifndef PERFECT_MAGIC_SQUARE_TEST_HELPERS_H
#define PERFECT_MAGIC_SQUARE_TEST_HELPERS_H

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "magic_square.h"
#include "program_options.h"

namespace test_helpers {

inline bool validate(const MagicSquare &square, int size) {
    const int expected_sum = magic_sum(size);

    std::vector<int> values;
    values.reserve(size * size);
    for (int r = 0; r < size; ++r)
        for (int c = 0; c < size; ++c)
            values.push_back(square.getValue(r, c));
    std::sort(values.begin(), values.end());

    std::vector<int> expected(size * size);
    std::iota(expected.begin(), expected.end(), 1);
    if (values != expected) {
        std::cout << "Validation FAILED: values are not 1.." << size * size << '\n';
        return false;
    }

    if (square.fitnessRows() != 0 || square.fitnessColumns() != 0 ||
        square.fitnessDiagonal1() != 0 || square.fitnessDiagonal2() != 0) {
        std::cout << "Validation FAILED: sums do not match " << expected_sum << '\n';
        return false;
    }

    std::cout << "Validation PASSED (magic sum = " << expected_sum << ")\n";
    return true;
}

inline std::vector<MagicSquare> seedPopulation(int size, int population) {
    std::vector<MagicSquare> pop;
    pop.reserve(population);
    for (int i = 0; i < population; i++) pop.emplace_back(size);
    return pop;
}

inline int runCorrectnessTest(int size, int population, int iterations, int argc, char **argv) {
    const std::vector<std::string_view> args(argv, argv + argc);
    const bool verbose = program_options::has(args, "-v");

    auto pop = seedPopulation(size, population);

    const auto start = std::chrono::high_resolution_clock::now();
    auto square = solve(pop, size, iterations, verbose);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed: " << elapsed.count() << " ms\n";

    if (square.getFitness() != 0) {
        std::cout << "No solution found!\n";
        return EXIT_FAILURE;
    }

    const std::string title = std::to_string(size) + "x" + std::to_string(size) + " Solution";
    square.print(title, true, false);
    square.write("result_" + std::to_string(size) + ".csv");

    return validate(square, size) ? EXIT_SUCCESS : EXIT_FAILURE;
}

inline int runVisualTest(int size, int population, int iterations) {
    std::cout << "=== Visual Test: " << size << "x" << size << " Magic Square (verbose) ===\n";
    std::cout << "Population: " << population << ", Iterations: " << iterations << '\n';
    std::cout << "Expected magic sum: " << magic_sum(size) << "\n\n";

    auto pop = seedPopulation(size, population);

    const auto start = std::chrono::high_resolution_clock::now();
    auto square = solve(pop, size, iterations, false, true);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "========================================\n";

    if (square.getFitness() == 0) {
        square.print("Final Solution", true, false);
        std::cout << "Solved in " << elapsed.count() << " ms\n";
        return EXIT_SUCCESS;
    }

    std::cout << "Best attempt (fitness = " << square.getFitness() << "):\n";
    square.print("Best Attempt", true, true);
    std::cout << "Time: " << elapsed.count() << " ms\n";
    return EXIT_FAILURE;
}

} // namespace test_helpers

#endif //PERFECT_MAGIC_SQUARE_TEST_HELPERS_H
