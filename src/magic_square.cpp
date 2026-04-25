//
// Created by Adnan Vatric on 17.02.23.
//

#include "magic_square.h"

#include <iostream>
#include <random>
#include <iomanip>
#include <fstream>

#include <omp.h>

#include "tabulate.hpp"

/**
 * Create a new magic square with given size.
 *
 * @param size
 * @param randomize
 */
MagicSquare::MagicSquare(int size, bool randomize) : dimension(size), fitness(0) {
    this->values.resize(size, std::vector<int>(size));
    this->sum = MAGIC_SUM(size);

    if (randomize) this->randomize();

    this->evaluate();
}

/**
 * Init square with 0 values.
 */
void MagicSquare::init() {
    std::fill(this->values.begin(), this->values.end(), std::vector<int>(this->dimension, 0));
    this->evaluate();
}

/**
 * Generate random numbers for magic square.
 */
void MagicSquare::randomize() {
    thread_local std::mt19937 rng(std::random_device{}());
    std::vector<int> numbers(this->dimension * this->dimension);
    std::iota(numbers.begin(), numbers.end(), 1); // Fill numbers from 1 to n*n
    std::shuffle(numbers.begin(), numbers.end(), rng); // Shuffle numbers

    // Fill the square with shuffled numbers
    for (int i = 0; i < this->dimension; ++i)
        for (int j = 0; j < this->dimension; ++j)
            this->values[i][j] = numbers[i * this->dimension + j];

    evaluate(); // Evaluate the new configuration
}

/**
 * Evaluate the fitness of a square solution.
 *
 */
void MagicSquare::evaluate() {
    this->fitness = 0;

    this->fitness += this->fitnessRows();
    this->fitness += this->fitnessColumns();
    this->fitness += this->fitnessDiagonal1();
    this->fitness += this->fitnessDiagonal2();
}

/**
 * Change position of two random numbers.
 *
 */
void MagicSquare::swap() {
    int fromRow, toRow, fromCol, toCol;
    thread_local std::mt19937 rng(std::random_device{}());
    thread_local std::uniform_int_distribution<int> dist(0, this->dimension - 1);

    do {
        fromRow = dist(rng);
        toRow = dist(rng);
        fromCol = dist(rng);
        toCol = dist(rng);
    } while ((fromRow == toRow) && (fromCol == toCol));

    std::swap(this->values[fromRow][fromCol], this->values[toRow][toCol]);
    this->evaluate();
}

/**
 * Print square to commandline using tabulate library
 *
 * @param show_details
 */
void MagicSquare::print(const std::string &title, bool show_details, bool show_fitness) {
    // Print the title
    std::cout << title << ":" << std::endl << std::endl;

    // Square table
    tabulate::Table square_table;

    // Initial table formatting for better readability
    square_table.format()
            .font_style({tabulate::FontStyle::bold})
            .font_align(tabulate::FontAlign::center)
            .column_separator("")
            .corner("")
            .border_top("")
            .border_bottom("")
            .border_left("")
            .border_right("");

    // Adding rows
    for (const auto &row : this->values) {
        tabulate::Table::Row_t table_row;
        for (int value : row) {
            table_row.push_back(std::to_string(value));
        }
        square_table.add_row(table_row);
    }

    if (show_details) {
        // Precompute fitness for rows and columns
        std::vector<int> rows_fitness(this->dimension);
        std::vector<int> cols_fitness(this->dimension);
        for (int i = 0; i < this->dimension; ++i) {
            rows_fitness[i] = fitnessRows(i);
            cols_fitness[i] = fitnessColumns(i);
        }

        // Precompute fitness for diagonals
        int diag1_fitness = fitnessDiagonal1();
        int diag2_fitness = fitnessDiagonal2();

        // Loop through each cell and apply coloring
        for (size_t i = 0; i < this->values.size(); ++i) {
            for (size_t j = 0; j < this->values[i].size(); ++j) {
                bool row_correct = (rows_fitness[i] == 0);
                bool col_correct = (cols_fitness[j] == 0);
                bool diag_correct = true;

                // Check if the cell is on the first diagonal
                bool is_on_diag1 = (i == j);
                if (is_on_diag1 && diag1_fitness != 0) {
                    diag_correct = false;
                }

                // Check if the cell is on the second diagonal
                bool is_on_diag2 = (i + j == this->dimension - 1);
                if (is_on_diag2 && diag2_fitness != 0) {
                    diag_correct = false;
                }

                // The cell is correct if row, column, and diagonals (if applicable) are correct
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

    // Print the square table
    std::cout << square_table << std::endl;

    // Print fitness information if requested
    if (show_fitness)
        std::cout << std::endl << "Fitness: " << this->getFitness() << std::endl;

    std::cout << std::endl << std::endl;
}

/**
 * Output square to a csv file.
 *
 * @param name
 */
void MagicSquare::write(std::string &name) {
    std::ofstream outputFile(name, std::ios::trunc);

    for (const auto &row: this->values) {
        for (const auto &num: row)
            outputFile << num << ';';

        outputFile << std::endl;
    }

    outputFile.close();
}

/**
 * Calculate fitness for all rows of a square.
 *
 * @return
 */
int MagicSquare::fitnessRows(int row_index) {
    if (row_index != -1) {
        // Calculate fitness for a single row
        int sum_row = 0;
        for (int col = 0; col < this->dimension; ++col)
            sum_row += this->values[row_index][col];
        return std::abs(sum_row - this->sum);
    } else {
        // Calculate fitness for all rows
        int fit = 0;
        for (int row = 0; row < this->dimension; ++row) {
            int sum_row = 0;
            for (int col = 0; col < this->dimension; ++col)
                sum_row += this->values[row][col];
            fit += std::abs(sum_row - this->sum);
        }
        return fit;
    }
}

/**
 * Calculate fitness for all columns of a square.
 *
 * @return
 */
int MagicSquare::fitnessColumns(int col_index) {
    if (col_index != -1) {
        // Calculate fitness for a single column
        int sum_col = 0;
        for (int row = 0; row < this->dimension; ++row)
            sum_col += this->values[row][col_index];
        return std::abs(sum_col - this->sum);
    } else {
        // Calculate fitness for all columns
        int fit = 0;
        for (int col = 0; col < this->dimension; ++col) {
            int sum_col = 0;
            for (int row = 0; row < this->dimension; ++row)
                sum_col += this->values[row][col];
            fit += std::abs(sum_col - this->sum);
        }
        return fit;
    }
}

/**
 * Calculate fitness of diagonal (left top to right bottom)
 *
 * @return
 */
int MagicSquare::fitnessDiagonal1() {
    int s = 0;

    for (int i = 0; i < this->dimension; i++)
        s += this->values[i][i];

    return std::abs(s - this->sum);
}

/**
 * Calculate fitness of diagonal (right top to left bottom)
 *
 * @return
 */
int MagicSquare::fitnessDiagonal2() {
    int s = 0;

    for (int i = 0; i < this->dimension; i++)
        s += this->values[i][this->dimension - i - 1];

    return std::abs(s - this->sum);
}

/**
 * Checks if given value exists in square.
 *
 * @param value
 * @return
 */
bool MagicSquare::valueExist(int value) {
    for (auto &row: this->values)
        for (auto col: row)
            if (col == value)
                return true;

    return false;
}

/**
 * Copy assign magic square.
 *
 * @param other
 * @return
 */
MagicSquare &MagicSquare::operator=(const MagicSquare &other) {
    if (this != &other)
        for (int row = 0; row < this->dimension; row++)
            for (int col = 0; col < this->dimension; col++)
                this->values[row][col] = other.getValue(row, col);

    this->evaluate();

    return *this;
}

/**
 * Custom operator comparing squares.
 *
 * @param a
 * @param b
 * @return
 */
bool operator==(const MagicSquare &a, const MagicSquare &b) {
    if (&a == &b)
        return true;

    for (int row = 0; row < a.dimension; row++)
        for (int col = 0; col < a.dimension; col++)
            if (a.getValue(row, col) != b.getValue(row, col))
                return false;

    return true;
}

/**
 * Custom operator comparing squares.
 *
 * @param a
 * @param b
 * @return
 */
bool operator!=(const MagicSquare &a, const MagicSquare &b) {
    return !(a == b);
}

/**
 * Sort squares by fitness.
 *
 * @param population
 */
void sort(std::vector<MagicSquare> &population) {
    std::sort(population.begin(),
              population.end(), [](const MagicSquare &a, const MagicSquare &b) {
                return a.getFitness() < b.getFitness();
            });
}

/**
 * Select the best third of the population.
 *
 * @param population
 * @param selected
 * @return
 */
void selection(std::vector<MagicSquare> &population, std::vector<MagicSquare> &selected) {
    sort(population);

    std::vector<MagicSquare> local_selected;

    for (int i = 0; i < population.size() / 3; i++)
        if (std::find(selected.begin(), selected.end(), population[i]) == selected.end())
            local_selected.push_back(population[i]);

    // Merge local vectors into the global vector
    selected.insert(selected.end(), local_selected.begin(), local_selected.end());
}

/**
 * Combine random squares from population.
 *
 * @param offspring
 * @param size
 * @param population
 * @return
 */
void crossover(std::vector<MagicSquare> &population, std::vector<MagicSquare> &offspring, int size) {
#pragma omp parallel default(none) shared(population, offspring, size)
    {
        thread_local std::mt19937 rng1(std::random_device{}());
        thread_local std::mt19937 rng2(std::random_device{}());
        thread_local std::mt19937 rng3(std::random_device{}());
        thread_local std::uniform_int_distribution<int> distParent(0, population.size() - 1);
        thread_local std::uniform_int_distribution<int> distFill(1, size * size);
        std::vector<MagicSquare> localOffspring;  // Local storage for each thread's offspring

#pragma omp for nowait  // Distribute loop iterations across threads without waiting
        for (int i = 0; i < population.size() / 3; i++) {
            MagicSquare child(size, false);

            child.init();

            MagicSquare parent1 = population[distParent(rng1)];
            MagicSquare parent2 = population[distParent(rng2)];

            while (parent1 == parent2) parent2 = population[distParent(rng2)];

            // Select values based on fitness
            for (int row = 0; row < size; ++row) {
                for (int col = 0; col < size; ++col) {
                    int val1 = parent1.getValue(row, col);
                    int val2 = parent2.getValue(row, col);
                    int chosenValue = ((parent1.fitnessRows(row) + parent1.fitnessColumns(col)) <
                                       (parent2.fitnessRows(row) + parent2.fitnessColumns(col))) ? val1 : val2;
                    if (!child.valueExist(chosenValue)) {
                        child.setValue(row, col, chosenValue);
                    }
                }
            }

            // Fill the rest with random values
            for (int row = 0; row < size; ++row) {
                for (int col = 0; col < size; ++col) {
                    if (child.getValue(row, col) == 0) {
                        int newValue;
                        do {
                            newValue = distFill(rng3);
                        } while (child.valueExist(newValue));
                        child.setValue(row, col, newValue);
                    }
                }
            }

            child.evaluate();
            localOffspring.push_back(child);
        }

#pragma omp critical  // Use critical section to safely merge results from each thread
        {
            offspring.insert(offspring.end(), localOffspring.begin(), localOffspring.end());
        }
    }
}


/**
 * Change position of two numbers in a square by a given probability.
 *
 * @param population
 * @param probability
 */
void mutate(std::vector<MagicSquare> &population, double probability) {
#pragma omp parallel default(none) shared(population, probability)
    {
        thread_local std::mt19937 rng(std::random_device{}());
        thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        // Mutate each square with a certain probability
#pragma omp for nowait
        for (size_t i = 0; i < population.size(); ++i) {
            if (dist(rng) < probability)
                population[i].swap();
        }
    }
}

/**
 * Solve a magic square using given parameters.
 *
 * @param population
 * @param size
 * @param iterations
 * @param verbose
 * @return
 */
MagicSquare solve(std::vector<MagicSquare> &population, int size, int iterations, bool verbose) {
    int lastFitness = -1;
    int unchanged = 0;
    double probability = BASE_MUTATION;
    bool infinite = false;

    if (iterations == -1) infinite = true;

    for (int it = 0; (it < iterations) || infinite; it++) {
        std::vector<MagicSquare> selected;
        std::vector<MagicSquare> offspring;
        std::vector<MagicSquare> check;
        selection(population, selected);

        if (selected.front().getFitness() == 0)
            return selected.front();

        if (verbose) {
            int count = 5;

            std::cout << "Current top 5:" << std::endl << std::endl;

            for (int i = 0; i < std::min(count, (int) selected.size()); ++i)
                selected[i].print("Round #" + std::to_string(i + 1));
        }

        crossover(population, offspring, size);
        sort(offspring);

        if (offspring.front().getFitness() == 0)
            return offspring.front();

        if (population.front().getFitness() == lastFitness) {
            unchanged++;

            if (probability < 1 && unchanged >= BASE_CHANGE_COUNT) probability += 0.1;
        } else {
            unchanged = 0;
            probability = BASE_MUTATION;
        }

        mutate(offspring, probability);

        lastFitness = population.front().getFitness();

        size_t i = 0;
        // Update population with selected individuals
        for (; i < selected.size() && i < population.size(); ++i) {
            population[i] = selected[i];
        }

        // Add offspring to the population
        for (size_t j = 0; j < offspring.size() && i < population.size(); ++j) {
            if (std::find(selected.begin(), selected.end(), offspring[j]) == selected.end()) {
                population[i] = offspring[j];
                i++;
            }
        }

        // Fill the rest with new random squares
        while (i < population.size()) {
            MagicSquare tmpSquare = MagicSquare(size);

            if ((std::find(selected.begin(), selected.end(), tmpSquare) == selected.end()) &&
                (std::find(offspring.begin(), offspring.end(), tmpSquare) == offspring.end()) &&
                (std::find(check.begin(), check.end(), tmpSquare) == check.end())) {
                population[i] = tmpSquare;
                check.push_back(tmpSquare);
                i++;
            }
        }

        lastFitness = population.front().getFitness();
    }

    sort(population);
    return population.front();
}
