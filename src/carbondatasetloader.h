#pragma on

// turn off warnings for this
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "binpack/nnue_data_binpack_format.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>

namespace carbon {

template<typename Duration = std::chrono::milliseconds>
inline double tick() {

    return (double) std::chrono::duration_cast<Duration>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

using DataEntry = binpack::binpack::TrainingDataEntry;
using DataSet   = std::vector<DataEntry>;
using binpack::binpack::CompressedTrainingDataEntryParallelReader;

std::function<bool(const DataEntry&)> skipPredicate = [](const DataEntry& entry) {
    static constexpr int    VALUE_NONE                      = 32002;

    static constexpr double desired_piece_count_weights[33] = {
        1.000000, 1.121094, 1.234375, 1.339844, 1.437500, 1.527344, 1.609375, 1.683594, 1.750000,
        1.808594, 1.859375, 1.902344, 1.937500, 1.964844, 1.984375, 1.996094, 2.000000, 1.996094,
        1.984375, 1.964844, 1.937500, 1.902344, 1.859375, 1.808594, 1.750000, 1.683594, 1.609375,
        1.527344, 1.437500, 1.339844, 1.234375, 1.121094, 1.000000};

    static constexpr double desired_piece_count_weights_total = []() {
        double tot = 0;
        for (auto w : desired_piece_count_weights)
            tot += w;
        return tot;
    }();

    static thread_local std::mt19937 gen(std::random_device {}());

    // keep stats on passing pieces
    static thread_local double alpha                            = 1;
    static thread_local double piece_count_history_all[33]      = {0};
    static thread_local double piece_count_history_passed[33]   = {0};
    static thread_local double piece_count_history_all_total    = 0;
    static thread_local double piece_count_history_passed_total = 0;

    // max skipping rate
    static constexpr double max_skipping_rate = 10.0;

    auto                    do_wld_skip       = [&entry]() {
        auto&                       prng = rng::get_thread_local_rng();

        std::bernoulli_distribution distrib(1.0
                                            - entry.score_result_prob() * entry.score_result_prob());
        return distrib(prng);
    };

    // Skip if the entry score is none
    if (entry.score == VALUE_NONE) {
        return true;
    }

    // Skip if the entry is too early
    if (entry.ply <= 16) {
        return true;
    }

    // Skip if the entry is a capturing move
    if ((entry.isCapturingMove() && (entry.score == 0 || entry.seeGE(0))) || entry.isInCheck()) {
        return true;
    }

    if (do_wld_skip()) {
        return true;
    }

    // const int pc = entry.pos.piecesBB().count();
    // piece_count_history_all[pc] += 1;
    // piece_count_history_all_total += 1;

    // // update alpha, which scales the filtering probability, to a maximum rate.
    // if (uint64_t(piece_count_history_all_total) % 10000 == 0) {
    //     double pass = piece_count_history_all_total * desired_piece_count_weights_total;
    //     for (int i = 0; i < 33; ++i) {
    //         if (desired_piece_count_weights[pc] > 0) {
    //             double tmp = piece_count_history_all_total * desired_piece_count_weights[pc]
    //                          / (desired_piece_count_weights_total * piece_count_history_all[pc]);
    //             if (tmp < pass)
    //                 pass = tmp;
    //         }
    //     }
    //     alpha = 1.0 / (pass * max_skipping_rate);
    // }

    // double tmp = alpha * piece_count_history_all_total * desired_piece_count_weights[pc]
    //              / (desired_piece_count_weights_total * piece_count_history_all[pc]);
    // tmp = std::min(1.0, tmp);
    // std::bernoulli_distribution distrib(1.0 - tmp);
    // auto&                       prng = rng::get_thread_local_rng();
    // if (distrib(prng))
    //     return true;

    // piece_count_history_passed[pc] += 1;
    // piece_count_history_passed_total += 1;

    return false;
};

struct DataLoader {

    static constexpr std::size_t                               ChunkSize = (1 << 20);

    std::vector<std::string>                                   paths;
    std::unique_ptr<CompressedTrainingDataEntryParallelReader> reader;

    std::vector<std::size_t>                                   permute_shuffle;
    DataSet                                                    buffer;
    DataSet                                                    active_buffer;
    DataSet                                                    active_batch;

    std::thread                                                readingThread;
    int                                                        batch_size;
    int                                                        current_batch_index  = 0;
    size_t                                                     total_positions_read = 0;
    int                                                        concurrency          = 8;

    static constexpr auto openmode = std::ios::in | std::ios::binary;

    DataLoader(const std::vector<std::string>& filename, int batch_size, int concurrency)
        : reader(std::make_unique<CompressedTrainingDataEntryParallelReader>(concurrency,
                                                                             filename,
                                                                             openmode,
                                                                             false,
                                                                             skipPredicate))
        , batch_size(batch_size)
        , paths(filename)
        , concurrency(concurrency) {
        buffer.reserve(ChunkSize);
        active_buffer.reserve(ChunkSize);
        permute_shuffle.resize(ChunkSize);
        active_batch.reserve(batch_size);
    }

    void start() {

        current_batch_index = 0;

        shuffle();
        loadNext();
        loadToActiveBuffer();
        readingThread = std::thread(&DataLoader::loadNext, this);
    }

    void loadToActiveBuffer() {
        active_buffer.clear();
        for (int i = 0; i < buffer.size(); i++) {
            active_buffer.push_back(buffer[i]);
        }
    }

    void loadNext() {
        buffer.clear();

        auto k = reader->fill(buffer, ChunkSize);

        if (ChunkSize != k) {
            reader = std::make_unique<binpack::binpack::CompressedTrainingDataEntryParallelReader>(
                concurrency,
                paths,
                openmode,
                false,
                skipPredicate);
        }
    }

    DataSet& next() {
        active_batch.clear();

        for (int i = 0; i < batch_size; i++) {
            if (current_batch_index >= active_buffer.size()) {

                current_batch_index = 0;

                if (readingThread.joinable()) {
                    readingThread.join();
                }

                loadToActiveBuffer();
                shuffle();

                readingThread = std::thread(&DataLoader::loadNext, this);
            }

            active_batch.push_back(active_buffer[permute_shuffle[current_batch_index++]]);
        }

        // std::cout << current_batch_index << std::endl;

        return active_batch;
    }

    void shuffle() {
        std::iota(permute_shuffle.begin(), permute_shuffle.end(), 0);
        std::shuffle(permute_shuffle.begin(),
                     permute_shuffle.end(),
                     std::mt19937(std::random_device()()));
    }

    void bench(int iterations) {
        // Start timing
        auto start_time = tick();

        // Load batches and count positions
        size_t total_positions = 0;
        for (int i = 0; i < iterations; ++i) {
            next();
            total_positions += batch_size;
        }

        // End timing
        auto end_time = tick();

        // Calculate elapsed time in seconds
        float elapsed_time_seconds = (end_time - start_time) / 1000.0;

        // Calculate positions per second
        float positions_per_second = total_positions / elapsed_time_seconds;

        // Report the result
        std::cout << "Loaded " << total_positions << " positions in " << elapsed_time_seconds
                  << " seconds. Positions per second: " << positions_per_second << std::endl;
    }
};

}    // namespace carbon