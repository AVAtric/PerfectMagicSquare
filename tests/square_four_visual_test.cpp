//
// Visual test: runs a 4x4 solve in verbose mode to show the genetic algorithm
// evolving in real time.
//

#include <iostream>
#include <vector>
#include <chrono>

#include "magic_square.h"

const int POPULATION = 10000;
const int SIZE = 4;
const int ITERATIONS = 10000;

int main() {
    std::cout << "=== Visual Test: 4x4 Magic Square (verbose) ===" << std::endl;
    std::cout << "Population: " << POPULATION << ", Iterations: " << ITERATIONS << std::endl;
    std::cout << "Expected magic sum: " << magic_sum(SIZE) << std::endl << std::endl;

    std::vector<MagicSquare> population;
    for (int i = 0; i < POPULATION; i++) population.emplace_back(SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    auto square = solve(population, SIZE, ITERATIONS, true);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "========================================" << std::endl;

    if (square.getFitness() == 0) {
        square.print("Final Solution", true, false);
        std::cout << "Solved in " << elapsed.count() << " ms" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Best attempt (fitness = " << square.getFitness() << "):" << std::endl;
    square.print("Best Attempt", true, true);
    std::cout << "Time: " << elapsed.count() << " ms" << std::endl;

    return EXIT_FAILURE;
}