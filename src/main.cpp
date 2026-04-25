//
// Created by Adnan Vatric on 17.02.23.
//

#include "magic_square.h"

#include <iostream>
#include <vector>

#include "program_options.h"

/**
 * Entry point
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    const std::vector<std::string_view> args(argv, argv + argc);

    bool fail = false;
    bool verbose = program_options::has(args, "-v");
    bool help = program_options::has(args, "-h");
    bool silent = program_options::has(args, "-s");

    int size = 0;
    int populationSize = 0;
    int iterations = 0;

    std::string name;

    if (help) {
        program_options::description();
        return EXIT_SUCCESS;
    }

    if (program_options::has(args, "-d"))
        size = std::stoi(program_options::get(args, "-d").begin());

    if (program_options::has(args, "-p"))
        populationSize = std::stoi(program_options::get(args, "-p").begin());

    if (program_options::has(args, "-i"))
        iterations = std::stoi(program_options::get(args, "-i").begin());

    if (program_options::has(args, "-o"))
        name = program_options::get(args, "-o").begin();

    if (silent && verbose) {
        std::cout << "Can't combine verbose and silent mode!" << std::endl << std::endl;
        fail = true;
    }

    if (size < 3 || size > 9) {
        std::cout << "Wrong square dimension!" << std::endl << std::endl;
        fail = true;
    }

    if ((populationSize < 1000 || populationSize > 10000) && !fail) {
        std::cout << "Wrong population size!" << std::endl << std::endl;
        fail = true;
    }

    if ((iterations < 1000 || iterations > 100000) && (iterations != -1) && !fail) {
        std::cout << "Wrong iterations count!" << std::endl << std::endl;
        fail = true;
    }

    if (fail) {
        program_options::description();
        return EXIT_FAILURE;
    }

    std::vector<MagicSquare> population;

    for (int i = 0; i < populationSize; i++) population.emplace_back(size);

    auto square = solve(population, size, iterations, verbose);

    if (square.getFitness() == 0) {
        if (!silent) {
            square.print("Solution", true, false);
        }

        if (!name.empty())
            square.write(name.append(".csv"));

        return EXIT_SUCCESS;
    }

    if (!silent)
        std::cout << "No solution found!" << std::endl;

    return EXIT_SUCCESS;
}
