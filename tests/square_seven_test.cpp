//
// Created by Adnan Vatric on 17.02.23.
//

#include <iostream>
#include <vector>

#include "magic_square.h"
#include "program_options.h"

const int POPULATION = 10000;
const int SIZE = 7;
const int ITERATIONS = 100000;

int main(int argc, char **argv) {
    const std::vector<std::string_view> args(argv, argv + argc);
    bool verbose = program_options::has(args, "-v");
    std::vector<MagicSquare> population;
    std::string name("result_7.csv");

    for (int i = 0; i < POPULATION; i++) population.emplace_back(SIZE);

    auto square = solve(population, SIZE, ITERATIONS, verbose);

    if (square.getFitness() == 0) {
        square.print("7x7 Solution", true, false);
        square.write(name);

        return EXIT_SUCCESS;
    }

    std::cout << "No solution found!" << std::endl;

    return EXIT_FAILURE;
}