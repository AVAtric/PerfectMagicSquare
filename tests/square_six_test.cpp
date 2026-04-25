//
// Created by Adnan Vatric on 17.02.23.
//

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

#include "magic_square.h"
#include "program_options.h"

const int POPULATION = 10000;
const int SIZE = 6;
const int ITERATIONS = 100000;

bool validate(MagicSquare &square, int size) {
    int expected_sum = magic_sum(size);

    std::vector<int> values;
    for (int r = 0; r < size; ++r)
        for (int c = 0; c < size; ++c)
            values.push_back(square.getValue(r, c));
    std::sort(values.begin(), values.end());
    std::vector<int> expected(size * size);
    std::iota(expected.begin(), expected.end(), 1);
    if (values != expected) {
        std::cout << "Validation FAILED: values are not 1.." << size * size << std::endl;
        return false;
    }

    if (square.fitnessRows() != 0 || square.fitnessColumns() != 0 ||
        square.fitnessDiagonal1() != 0 || square.fitnessDiagonal2() != 0) {
        std::cout << "Validation FAILED: sums do not match " << expected_sum << std::endl;
        return false;
    }

    std::cout << "Validation PASSED (magic sum = " << expected_sum << ")" << std::endl;
    return true;
}

int main(int argc, char **argv) {
    const std::vector<std::string_view> args(argv, argv + argc);
    bool verbose = program_options::has(args, "-v");
    std::vector<MagicSquare> population;
    std::string name("result_6.csv");

    for (int i = 0; i < POPULATION; i++) population.emplace_back(SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    auto square = solve(population, SIZE, ITERATIONS, verbose);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed: " << elapsed.count() << " ms" << std::endl;

    if (square.getFitness() == 0) {
        square.print("6x6 Solution", true, false);
        square.write(name);

        if (!validate(square, SIZE))
            return EXIT_FAILURE;

        return EXIT_SUCCESS;
    }

    std::cout << "No solution found!" << std::endl;

    return EXIT_FAILURE;
}