//
// Created by Adnan Vatric on 17.02.23.
//

#include "magic_square.h"

#include <iostream>
#include <random>
#include <fstream>
#include <algorithm>
#include <numeric>

#include <omp.h>

#include "tabulate.hpp"

/**
 * Create a new magic square with given size.
 *
 * @param size
 * @param randomize
 */
MagicSquare::MagicSquare(int size, bool randomize)
        : values(size * size, 0), dimension(size), fitness(0), sum(magic_sum(size)),
          row_sums(size, 0), col_sums(size, 0), diag1_sum(0), diag2_sum(0) {
    if (randomize) this->randomize();
    else this->evaluate();
}

/**
 * Init square with 0 values.
 */
void MagicSquare::init() {
    std::fill(values.begin(), values.end(), 0);
    std::fill(row_sums.begin(), row_sums.end(), 0);
    std::fill(col_sums.begin(), col_sums.end(), 0);
    diag1_sum = 0;
    diag2_sum = 0;
    computeFitnessFromSums();
}

/**
 * Generate random numbers for magic square.
 */
void MagicSquare::randomize() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::iota(values.begin(), values.end(), 1);
    std::shuffle(values.begin(), values.end(), rng);
    evaluate();
}

/**
 * Recompute all cached sums from the values array.
 */
void MagicSquare::recomputeSums() {
    std::fill(row_sums.begin(), row_sums.end(), 0);
    std::fill(col_sums.begin(), col_sums.end(), 0);
    diag1_sum = 0;
    diag2_sum = 0;

    for (int r = 0; r < dimension; r++) {
        for (int c = 0; c < dimension; c++) {
            int v = values[r * dimension + c];
            row_sums[r] += v;
            col_sums[c] += v;
            if (r == c) diag1_sum += v;
            if (r + c == dimension - 1) diag2_sum += v;
        }
    }
}

/**
 * Compute fitness from cached sums (O(n) instead of O(n^2)).
 */
void MagicSquare::computeFitnessFromSums() {
    fitness = 0;
    for (int i = 0; i < dimension; i++) {
        fitness += std::abs(row_sums[i] - sum);
        fitness += std::abs(col_sums[i] - sum);
    }
    fitness += std::abs(diag1_sum - sum);
    fitness += std::abs(diag2_sum - sum);
}

/**
 * Evaluate the fitness of a square solution.
 */
void MagicSquare::evaluate() {
    recomputeSums();
    computeFitnessFromSums();
}

/**
 * Change position of two random numbers with incremental fitness update.
 */
void MagicSquare::swap() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, dimension - 1);

    int r1, c1, r2, c2;
    do {
        r1 = dist(rng);
        c1 = dist(rng);
        r2 = dist(rng);
        c2 = dist(rng);
    } while (r1 == r2 && c1 == c2);

    int idx1 = r1 * dimension + c1;
    int idx2 = r2 * dimension + c2;
    int diff = values[idx1] - values[idx2];

    // Incremental sum updates
    if (r1 != r2) {
        row_sums[r1] -= diff;
        row_sums[r2] += diff;
    }
    if (c1 != c2) {
        col_sums[c1] -= diff;
        col_sums[c2] += diff;
    }
    if (r1 == c1) diag1_sum -= diff;
    if (r2 == c2) diag1_sum += diff;
    if (r1 + c1 == dimension - 1) diag2_sum -= diff;
    if (r2 + c2 == dimension - 1) diag2_sum += diff;

    std::swap(values[idx1], values[idx2]);
    computeFitnessFromSums();
}

/**
 * Steepest-descent local search: try all possible swaps per round,
 * apply the single best-improving swap. O(1) per swap evaluation
 * using cached sums. Self-terminates at local optimum.
 *
 * @param rounds maximum number of improvement rounds
 */
void MagicSquare::localSearch(int rounds) {
    int n2 = dimension * dimension;

    for (int r = 0; r < rounds && fitness > 0; r++) {
        int bestIdx1 = -1, bestIdx2 = -1;
        int bestFitness = fitness;

        for (int idx1 = 0; idx1 < n2 - 1; idx1++) {
            int r1 = idx1 / dimension, c1 = idx1 % dimension;

            for (int idx2 = idx1 + 1; idx2 < n2; idx2++) {
                int r2 = idx2 / dimension, c2 = idx2 % dimension;
                int diff = values[idx1] - values[idx2];
                if (diff == 0) continue;

                // O(1) trial fitness via delta computation
                int trialFitness = fitness;

                if (r1 != r2) {
                    trialFitness -= std::abs(row_sums[r1] - sum);
                    trialFitness += std::abs(row_sums[r1] - diff - sum);
                    trialFitness -= std::abs(row_sums[r2] - sum);
                    trialFitness += std::abs(row_sums[r2] + diff - sum);
                }

                if (c1 != c2) {
                    trialFitness -= std::abs(col_sums[c1] - sum);
                    trialFitness += std::abs(col_sums[c1] - diff - sum);
                    trialFitness -= std::abs(col_sums[c2] - sum);
                    trialFitness += std::abs(col_sums[c2] + diff - sum);
                }

                bool on_d1_1 = (r1 == c1), on_d1_2 = (r2 == c2);
                if (on_d1_1 || on_d1_2) {
                    int new_d1 = diag1_sum;
                    if (on_d1_1) new_d1 -= diff;
                    if (on_d1_2) new_d1 += diff;
                    trialFitness -= std::abs(diag1_sum - sum);
                    trialFitness += std::abs(new_d1 - sum);
                }

                bool on_d2_1 = (r1 + c1 == dimension - 1);
                bool on_d2_2 = (r2 + c2 == dimension - 1);
                if (on_d2_1 || on_d2_2) {
                    int new_d2 = diag2_sum;
                    if (on_d2_1) new_d2 -= diff;
                    if (on_d2_2) new_d2 += diff;
                    trialFitness -= std::abs(diag2_sum - sum);
                    trialFitness += std::abs(new_d2 - sum);
                }

                if (trialFitness < bestFitness) {
                    bestFitness = trialFitness;
                    bestIdx1 = idx1;
                    bestIdx2 = idx2;
                    if (bestFitness == 0) goto apply_swap;
                }
            }
        }

        apply_swap:
        if (bestIdx1 != -1 && bestFitness < fitness) {
            int r1 = bestIdx1 / dimension, c1 = bestIdx1 % dimension;
            int r2 = bestIdx2 / dimension, c2 = bestIdx2 % dimension;
            int diff = values[bestIdx1] - values[bestIdx2];

            std::swap(values[bestIdx1], values[bestIdx2]);

            if (r1 != r2) {
                row_sums[r1] -= diff;
                row_sums[r2] += diff;
            }
            if (c1 != c2) {
                col_sums[c1] -= diff;
                col_sums[c2] += diff;
            }
            if (r1 == c1) diag1_sum -= diff;
            if (r2 == c2) diag1_sum += diff;
            if (r1 + c1 == dimension - 1) diag2_sum -= diff;
            if (r2 + c2 == dimension - 1) diag2_sum += diff;

            fitness = bestFitness;
        } else {
            break; // local optimum reached, no improving swap exists
        }
    }
}

/**
 * Print square to commandline using tabulate library
 */
void MagicSquare::print(const std::string &title, bool show_details, bool show_fitness) {
    std::cout << title << ":" << std::endl << std::endl;

    tabulate::Table square_table;

    square_table.format()
            .font_style({tabulate::FontStyle::bold})
            .font_align(tabulate::FontAlign::center)
            .column_separator("")
            .corner("")
            .border_top("")
            .border_bottom("")
            .border_left("")
            .border_right("");

    for (int i = 0; i < dimension; i++) {
        tabulate::Table::Row_t table_row;
        for (int j = 0; j < dimension; j++) {
            table_row.push_back(std::to_string(values[i * dimension + j]));
        }
        square_table.add_row(table_row);
    }

    if (show_details) {
        std::vector<int> rows_fitness(dimension);
        std::vector<int> cols_fitness(dimension);
        for (int i = 0; i < dimension; ++i) {
            rows_fitness[i] = fitnessRows(i);
            cols_fitness[i] = fitnessColumns(i);
        }

        int diag1_fitness = fitnessDiagonal1();
        int diag2_fitness = fitnessDiagonal2();

        for (int i = 0; i < dimension; ++i) {
            for (int j = 0; j < dimension; ++j) {
                bool row_correct = (rows_fitness[i] == 0);
                bool col_correct = (cols_fitness[j] == 0);
                bool diag_correct = true;

                if (i == j && diag1_fitness != 0)
                    diag_correct = false;

                if (i + j == dimension - 1 && diag2_fitness != 0)
                    diag_correct = false;

                bool cell_correct = row_correct && col_correct && diag_correct;

                if (cell_correct) {
                    square_table[i][j].format()
                            .font_background_color(tabulate::Color::green)
                            .font_color(tabulate::Color::grey);
                } else {
                    square_table[i][j].format()
                            .font_background_color(tabulate::Color::red)
                            .font_color(tabulate::Color::grey);
                }
            }
        }
    }

    std::cout << square_table << std::endl;

    if (show_fitness)
        std::cout << std::endl << "Fitness: " << this->getFitness() << std::endl;

    std::cout << std::endl << std::endl;
}

/**
 * Output square to a csv file.
 */
void MagicSquare::write(const std::string &name) {
    std::ofstream outputFile(name, std::ios::trunc);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++)
            outputFile << values[i * dimension + j] << ';';
        outputFile << std::endl;
    }

    outputFile.close();
}

/**
 * Calculate fitness for rows using cached sums (O(1) per row).
 */
int MagicSquare::fitnessRows(int row_index) const {
    if (row_index != -1) {
        return std::abs(row_sums[row_index] - sum);
    }
    int fit = 0;
    for (int i = 0; i < dimension; i++)
        fit += std::abs(row_sums[i] - sum);
    return fit;
}

/**
 * Calculate fitness for columns using cached sums (O(1) per column).
 */
int MagicSquare::fitnessColumns(int col_index) const {
    if (col_index != -1) {
        return std::abs(col_sums[col_index] - sum);
    }
    int fit = 0;
    for (int i = 0; i < dimension; i++)
        fit += std::abs(col_sums[i] - sum);
    return fit;
}

/**
 * Calculate fitness of diagonal (left top to right bottom).
 */
int MagicSquare::fitnessDiagonal1() const {
    return std::abs(diag1_sum - sum);
}

/**
 * Calculate fitness of diagonal (right top to left bottom).
 */
int MagicSquare::fitnessDiagonal2() const {
    return std::abs(diag2_sum - sum);
}

/**
 * Checks if given value exists in square.
 */
bool MagicSquare::valueExist(int value) const {
    for (int i = 0; i < dimension * dimension; i++)
        if (values[i] == value)
            return true;
    return false;
}

/**
 * Copy assign magic square (copies cached sums, avoids recomputation).
 */
MagicSquare &MagicSquare::operator=(const MagicSquare &other) {
    if (this != &other) {
        values = other.values;
        fitness = other.fitness;
        row_sums = other.row_sums;
        col_sums = other.col_sums;
        diag1_sum = other.diag1_sum;
        diag2_sum = other.diag2_sum;
    }
    return *this;
}

/**
 * Custom operator comparing squares.
 */
bool operator==(const MagicSquare &a, const MagicSquare &b) {
    if (&a == &b) return true;
    return a.values == b.values;
}

/**
 * Custom operator comparing squares.
 */
bool operator!=(const MagicSquare &a, const MagicSquare &b) {
    return !(a == b);
}

/**
 * Sort squares by fitness.
 */
void sort(std::vector<MagicSquare> &population) {
    std::sort(population.begin(),
              population.end(), [](const MagicSquare &a, const MagicSquare &b) {
                return a.getFitness() < b.getFitness();
            });
}

/**
 * Select the best third of the population (simplified, no expensive dedup).
 */
void selection(std::vector<MagicSquare> &population, std::vector<MagicSquare> &selected) {
    sort(population);
    int count = static_cast<int>(population.size()) / 3;
    selected.reserve(count);
    for (int i = 0; i < count; i++)
        selected.push_back(population[i]);
}

/**
 * Combine squares from population using tournament selection,
 * O(1) value existence check via bitset, and deterministic fill.
 */
void crossover(std::vector<MagicSquare> &population, std::vector<MagicSquare> &offspring, int size) {
    int numOffspring = static_cast<int>(population.size()) / 3;
    int n2 = size * size;
    int popSize = static_cast<int>(population.size());

#pragma omp parallel default(none) shared(population, offspring, size, numOffspring, n2, popSize)
    {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> distParent(0, popSize - 1);
        std::vector<MagicSquare> localOffspring;
        localOffspring.reserve(numOffspring / omp_get_num_threads() + 1);

#pragma omp for nowait
        for (int i = 0; i < numOffspring; i++) {
            MagicSquare child(size, false);
            child.init();

            // Tournament selection: pick 2 candidates per parent, take the fitter one
            int p1 = distParent(rng);
            {
                int p1b = distParent(rng);
                if (population[p1b].getFitness() < population[p1].getFitness()) p1 = p1b;
            }
            int p2 = distParent(rng);
            {
                int p2b = distParent(rng);
                if (population[p2b].getFitness() < population[p2].getFitness()) p2 = p2b;
            }
            while (p1 == p2) p2 = distParent(rng);

            const MagicSquare &parent1 = population[p1];
            const MagicSquare &parent2 = population[p2];

            // O(1) value existence check via boolean array (replaces O(n^2) valueExist)
            bool used[82] = {};

            for (int row = 0; row < size; ++row) {
                for (int col = 0; col < size; ++col) {
                    int val1 = parent1.getValue(row, col);
                    int val2 = parent2.getValue(row, col);
                    int chosenValue = ((parent1.fitnessRows(row) + parent1.fitnessColumns(col)) <
                                       (parent2.fitnessRows(row) + parent2.fitnessColumns(col))) ? val1 : val2;
                    if (!used[chosenValue]) {
                        child.setValue(row, col, chosenValue);
                        used[chosenValue] = true;
                    }
                }
            }

            // Deterministic fill: collect missing values and shuffle once
            std::vector<int> missing;
            missing.reserve(n2);
            for (int v = 1; v <= n2; v++)
                if (!used[v]) missing.push_back(v);
            std::shuffle(missing.begin(), missing.end(), rng);

            int mi = 0;
            for (int row = 0; row < size; ++row)
                for (int col = 0; col < size; ++col)
                    if (child.getValue(row, col) == 0)
                        child.setValue(row, col, missing[mi++]);

            child.evaluate();
            localOffspring.push_back(std::move(child));
        }

#pragma omp critical
        {
            offspring.insert(offspring.end(),
                             std::make_move_iterator(localOffspring.begin()),
                             std::make_move_iterator(localOffspring.end()));
        }
    }
}

/**
 * Change position of two numbers in a square by a given probability.
 */
void mutate(std::vector<MagicSquare> &population, double probability) {
#pragma omp parallel default(none) shared(population, probability)
    {
        thread_local std::mt19937 rng(std::random_device{}());
        thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for nowait
        for (size_t i = 0; i < population.size(); ++i) {
            if (dist(rng) < probability)
                population[i].swap();
        }
    }
}

/**
 * Solve a magic square using a memetic algorithm (GA + local search).
 * Key optimizations:
 * - No std::find() deduplication (was O(pop * n^2) per call)
 * - Elitism preserves best solutions
 * - Local search (steepest descent) on top candidates each generation
 * - Catastrophic restart on prolonged stagnation
 */
MagicSquare solve(std::vector<MagicSquare> &population, int size, int iterations, bool verbose) {
    int lastFitness = -1;
    int unchanged = 0;
    double probability = BASE_MUTATION;
    bool infinite = (iterations == -1);
    int popSize = static_cast<int>(population.size());
    int eliteCount = std::max(2, popSize / 10);
    int lsRounds = size * size;

    // Apply initial local search to best candidates
    sort(population);
    if (population.front().getFitness() == 0)
        return population.front();

    int initLS = std::min(20, popSize);
#pragma omp parallel for default(none) shared(population, initLS, lsRounds)
    for (int i = 0; i < initLS; i++) {
        population[i].localSearch(lsRounds);
    }
    sort(population);
    if (population.front().getFitness() == 0)
        return population.front();

    for (int it = 0; (it < iterations) || infinite; it++) {
        std::vector<MagicSquare> offspring;
        offspring.reserve(popSize / 3);

        // Crossover with tournament selection
        crossover(population, offspring, size);

        // Check offspring for solution
        for (auto &o : offspring) {
            if (o.getFitness() == 0) return o;
        }

        // Mutate offspring
        mutate(offspring, probability);

        // Local search on top offspring (parallel)
        sort(offspring);
        int lsCount = std::min(10, static_cast<int>(offspring.size()));
#pragma omp parallel for default(none) shared(offspring, lsCount, lsRounds)
        for (int i = 0; i < lsCount; i++) {
            offspring[i].localSearch(lsRounds);
        }

        // Check for solution after local search
        for (int i = 0; i < lsCount; i++) {
            if (offspring[i].getFitness() == 0) return offspring[i];
        }

        // Re-sort after local search changed fitness values
        sort(offspring);

        if (verbose) {
            std::cout << "Gen " << it << " best fitness: " << population[0].getFitness()
                      << " offspring best: " << offspring[0].getFitness() << std::endl;
        }

        // Build next generation: elite + offspring + random

        // Keep elite from current population (indices 0..eliteCount-1 stay)
        size_t i = eliteCount;

        // Insert best offspring
        for (size_t j = 0; j < offspring.size() && i < static_cast<size_t>(popSize); j++) {
            population[i++] = std::move(offspring[j]);
        }

        // Fill rest with new random squares
        while (i < static_cast<size_t>(popSize)) {
            population[i] = MagicSquare(size);
            i++;
        }

        sort(population);

        // Adaptive mutation + stagnation handling
        if (population.front().getFitness() == lastFitness) {
            unchanged++;
            if (probability < 1.0 && unchanged >= BASE_CHANGE_COUNT)
                probability += 0.1;

            // Catastrophic restart after prolonged stagnation
            if (unchanged > 30) {
                // Keep the very best, regenerate everything else
                for (int j = eliteCount; j < popSize; j++) {
                    population[j] = MagicSquare(size);
                }
                // Apply local search to new random squares
                int restartLS = std::min(20, popSize - eliteCount);
#pragma omp parallel for default(none) shared(population, eliteCount, restartLS, lsRounds)
                for (int j = 0; j < restartLS; j++) {
                    population[eliteCount + j].localSearch(lsRounds);
                }
                sort(population);
                if (population.front().getFitness() == 0)
                    return population.front();
                unchanged = 0;
                probability = BASE_MUTATION;
            }
        } else {
            unchanged = 0;
            probability = BASE_MUTATION;
        }

        lastFitness = population.front().getFitness();
    }

    sort(population);
    return population.front();
}