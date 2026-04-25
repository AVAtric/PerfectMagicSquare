//
// Created by Adnan Vatric on 17.02.23.
//

#include "magic_square.h"

#include <iostream>
#include <vector>

#include "program_options.h"

int main(int argc, char **argv) {
    const std::vector<std::string_view> args(argv, argv + argc);

    const bool verbose = program_options::has(args, "-v");
    const bool silent = program_options::has(args, "-s");

    if (program_options::has(args, "-h")) {
        program_options::description();
        return EXIT_SUCCESS;
    }

    int size = 0;
    int populationSize = 0;
    int iterations = 0;
    std::string name;

    if (program_options::has(args, "-d"))
        size = std::stoi(std::string(program_options::get(args, "-d")));

    if (program_options::has(args, "-p"))
        populationSize = std::stoi(std::string(program_options::get(args, "-p")));

    if (program_options::has(args, "-i"))
        iterations = std::stoi(std::string(program_options::get(args, "-i")));

    if (program_options::has(args, "-o"))
        name = std::string(program_options::get(args, "-o"));

    const char *error = nullptr;
    if (silent && verbose)                                error = "Can't combine verbose and silent mode!";
    else if (size < MIN_DIMENSION || size > MAX_DIMENSION) error = "Wrong square dimension!";
    else if (populationSize < 1000 || populationSize > 10000) error = "Wrong population size!";
    else if ((iterations < 1000 || iterations > 100000) && iterations != -1) error = "Wrong iterations count!";

    if (error) {
        std::cout << error << "\n\n";
        program_options::description();
        return EXIT_FAILURE;
    }

    std::vector<MagicSquare> population;
    population.reserve(populationSize);
    for (int i = 0; i < populationSize; i++) population.emplace_back(size);

    auto square = solve(population, size, iterations, verbose);

    if (square.getFitness() == 0) {
        if (!silent) square.print("Solution", true, false);
        if (!name.empty()) square.write(name + ".csv");
        return EXIT_SUCCESS;
    }

    if (!silent) std::cout << "No solution found!\n";

    return EXIT_SUCCESS;
}
