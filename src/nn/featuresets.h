#pragma once

#include <array>

namespace nn {
namespace FeatureSets {
    class QuadBuckets_hm {
        public:
        // clang-format off
        static constexpr std::array<int, 64> indices {
            0, 0, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 0,
            2, 2, 3, 3, 3, 3, 2, 2,
            2, 2, 3, 3, 3, 3, 2, 2,
            2, 2, 3, 3, 3, 3, 2, 2,
            2, 2, 3, 3, 3, 3, 2, 2,
        };

        static constexpr int COUNT = 4;
        // clang-format on
    };

    class OctaBuckets_hm {
        public:
        // clang-format off
        static constexpr std::array<int, 64> indices[64] {
            0, 0, 1, 1, 1, 1, 0, 0,
            2, 2, 3, 3, 3, 3, 2, 2,
            4, 4, 5, 5, 5, 5, 4, 4,
            4, 4, 5, 5, 5, 5, 4, 4,
            6, 6, 7, 7, 7, 7, 6, 6,
            6, 6, 7, 7, 7, 7, 6, 6,
            6, 6, 7, 7, 7, 7, 6, 6,
            6, 6, 7, 7, 7, 7, 6, 6,
        };
        
        static constexpr int COUNT = 8;
        // clang-format on
    };

    class HalfKA_hm {
        public:
        // clang-format off
        static constexpr std::array<int, 64> indices = {
            0,  1,  2,  3,  3,  2,  1,  0,
            4,  5,  6,  7,  7,  6,  5,  4,
            8,  9,  10, 11, 11, 10, 9,  8,
            12, 13, 14, 15, 15, 14, 13, 12,
            16, 17, 18, 19, 19, 18, 17, 16,
            20, 21, 22, 23, 23, 22, 21, 20,
            24, 25, 26, 27, 27, 26, 25, 24,
            28, 29, 30, 31, 31, 30, 29, 28,
        };
        // clang-format on

        static constexpr int COUNT = 32;
    };

    class HalfKA {
        public:
        // clang-format off
        
        static constexpr std::array<int, 64> indices = {
            0,  1,  2,  3,  4,  5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63,
        };

        static constexpr int COUNT = 64;
    };
};    // namespace FeatureSets
}    // namespace nn