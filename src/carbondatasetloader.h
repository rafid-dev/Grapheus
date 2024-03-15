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

using DataSet = std::vector<binpack::binpack::TrainingDataEntry>;

struct DataLoader {

    static constexpr std::size_t                        ChunkSize = (1 << 21);

    std::string                                         path;
    binpack::binpack::CompressedTrainingDataEntryReader reader;

    std::vector<std::size_t>                            permute_shuffle;
    DataSet                                             buffer;
    DataSet                                             active_buffer;
    DataSet                                             active_batch;

    std::thread                                         readingThread;
    int                                                 batch_size;
    int                                                 current_batch_index = 0;

    DataLoader(const std::string& filename, int batch_size)
        : reader(filename)
        , batch_size(batch_size)
        , path(filename) {
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

        for (std::size_t counter = 0; counter < ChunkSize;) {
            // If we finished, go back to the beginning
            if (!reader.hasNext()) {
                reader = binpack::binpack::CompressedTrainingDataEntryReader(path);
            }

            // Get info
            auto entry = reader.next();

            // Skip if the entry score is none
            if (entry.score == 32002) {
                continue;
            }

            // Skip if the entry is too early
            if (entry.ply <= 16) {
                continue;
            }

            // Skip if the entry is a capturing move
            if (entry.isCapturingMove()) {
                continue;
            }

            // Skip if the entry is in check
            if (entry.isInCheck()) {
                continue;
            }

            buffer.push_back(entry);
            ++counter;
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

    void bench() {
        auto start = tick();

        loadNext();

        auto duration = tick() - start;
    }
};

}    // namespace carbon