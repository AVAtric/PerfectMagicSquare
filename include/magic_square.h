//
// Created by Adnan Vatric on 17.02.23.
//

#ifndef PERFECT_MAGIC_SQUARE_MAGIC_SQUARE_H
#define PERFECT_MAGIC_SQUARE_MAGIC_SQUARE_H

#include <string>
#include <vector>

// Calculate the magic sum of a square of given size
constexpr int magic_sum(int n) { return n * (n * n + 1) / 2; }

// Define the mutation rate and the number of changes needed to increase the mutation rate
const double BASE_MUTATION = 0.1;
const double BASE_CHANGE_COUNT = 3;

/**
 * Base structure of a single magic square.
 * The size is is passed to constructor.
 *
 */
class MagicSquare {
public:
    explicit MagicSquare(int, bool = true);

    void init();

    void randomize();

    void evaluate();

    void swap();

    void localSearch(int rounds);

    void print(const std::string &title = "", bool show_details = true, bool show_fitness = true);

    void write(const std::string &);

    int fitnessRows(int row_index = -1) const;

    int fitnessColumns(int col_index = -1) const;

    int fitnessDiagonal1() const;

    int fitnessDiagonal2() const;

    [[nodiscard]] auto getFitness() const { return this->fitness; }

    [[nodiscard]] auto getSum() const { return this->sum; }

    [[nodiscard]] auto getValue(int row, int col) const { return this->values[row * dimension + col]; }

    auto &getValues() { return this->values; }

    void setValue(int row, int col, int value) { this->values[row * dimension + col] = value; }

    bool valueExist(int) const;

    MagicSquare &operator=(const MagicSquare &);

    friend bool operator==(const MagicSquare &, const MagicSquare &);

    friend bool operator!=(const MagicSquare &, const MagicSquare &);

private:
    std::vector<int> values;    // flat row-major array (was vector<vector<int>>)
    int dimension;
    int fitness;
    int sum;

    // Cached sums for O(1) fitness queries and incremental swap updates
    std::vector<int> row_sums;
    std::vector<int> col_sums;
    int diag1_sum;
    int diag2_sum;

    void recomputeSums();

    void computeFitnessFromSums();
};

bool operator==(const MagicSquare &, const MagicSquare &);

bool operator!=(const MagicSquare &, const MagicSquare &);

void sort(std::vector<MagicSquare> &);

void selection(std::vector<MagicSquare> &, std::vector<MagicSquare> &);

void crossover(std::vector<MagicSquare> &, std::vector<MagicSquare> &, int);

void mutate(std::vector<MagicSquare> &population, double probability);

MagicSquare solve(std::vector<MagicSquare> &, int, int, bool = false);

#endif //PERFECT_MAGIC_SQUARE_MAGIC_SQUARE_H