#include "argparse.hpp"
#include "carbondatasetloader.h"
#include "chess/chess.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "dataset/io.h"
#include "dataset/process.h"
#include "misc/csv.h"
#include "misc/timer.h"
#include "nn/nn.h"
#include "operations/operations.h"

#include <fstream>
#include <limits>

namespace fs = std::filesystem;

using namespace nn;
using namespace data;

struct ChessModel : nn::Model {

    int   max_epochs    = 0;
    int   current_epoch = 0;

    float start_lambda  = 0.7;
    float end_lambda    = 0.7;

    // seting inputs
    virtual void setup_inputs_and_outputs(carbon::DataSet& positions) = 0;

    // train function
    void train(carbon::DataLoader& loader,
               int                 epochs             = 1000,
               int                 epoch_size         = 1e7,
               float               start_lambda       = 0.7,
               float               end_lambda         = 0.7,
               int                 epoch_continuation = 0) {
        this->max_epochs   = epochs;
        this->start_lambda = start_lambda;
        this->end_lambda   = end_lambda;
        this->compile(loader.batch_size);

        Timer t {};
        for (int i = epoch_continuation; i < epochs; i++) {
            this->current_epoch = i;
            t.start();
            size_t prev_duration = 0;
            float  batch_loss    = 0;
            float  epoch_loss    = 0;

            for (int b = 0; b < epoch_size / loader.batch_size; b++) {
                // get the next dataset and set it up while the other things
                // are running on the gpu
                auto& ds = loader.next();
                setup_inputs_and_outputs(ds);

                // print score of last iteration
                if (b > 0) {
                    batch_loss = loss_of_last_batch();
                    epoch_loss += batch_loss;
                }

                t.stop();
                if (b > 0
                    && (b == (epoch_size / loader.batch_size) - 1
                        || t.elapsed() - prev_duration > 1000)) {
                    prev_duration = t.elapsed();

                    printf("\rep/ba = [%3d/%5d], ", i, b);
                    printf("batch_loss = [%1.8f], ", batch_loss);
                    printf("epoch_loss = [%1.8f], ", epoch_loss / b);
                    printf("speed = [%9d pos/s], ",
                           (int) round(1000.0f * loader.batch_size * b / t.elapsed()));
                    printf("time = [%3ds]", (int) t.elapsed() / 1000);
                    std::cout << std::flush;
                }

                // start training of new batch
                batch();
            }

            std::cout << std::endl;
            next_epoch(epoch_loss / (epoch_size / loader.batch_size));
        }
    }

    void test_fen(const std::string& fen) {
        this->compile(1);

        binpack::chess::Position pos;

        pos.set(fen);

        binpack::binpack::TrainingDataEntry entry;
        entry.pos    = pos;
        entry.score  = 0;
        entry.result = 0;

        carbon::DataSet entries;
        entries.push_back(entry);

        // setup inputs of network
        setup_inputs_and_outputs(entries);

        // forward pass
        this->upload_inputs();
        this->forward();

        // go through the layers and download values

        std::cout
            << "==================================================================================\n";
        std::cout << "testing fen: " << fen << std::endl;

        int idx = 0;
        for (auto layer : m_layers) {
            layer->dense_output.values >> CPU;

            std::cout << "LAYER " << ++idx << std::endl;
            for (int i = 0; i < std::min((size_t) 16, layer->size); i++) {
                std::cout << std::setw(10) << layer->dense_output.values(i, 0);
            }
            if (layer->size > 16) {
                std::cout << " ......... " << layer->dense_output.values(layer->size - 1, 0);
            }
            std::cout << "\n";
        }
    }

    void distribution(carbon::DataLoader& loader, int batches = 32) {
        this->compile(loader.batch_size);

        std::vector<DenseMatrix<float>> max_values {};
        std::vector<DenseMatrix<float>> min_values {};

        for (auto l : m_layers) {
            max_values.emplace_back(l->dense_output.values.m, 1);
            min_values.emplace_back(l->dense_output.values.m, 1);
            max_values.back().malloc<data::CPU>();
            min_values.back().malloc<data::CPU>();
            math::uniform(max_values.back(), -1000000.0f, -1000000.0f);
            math::uniform(min_values.back(), 1000000.0f, 1000000.0f);
        }

        for (int b = 0; b < batches; b++) {
            auto& ds = loader.next();
            setup_inputs_and_outputs(ds);
            this->upload_inputs();
            this->forward();
            std::cout << "\r" << b << " / " << batches << std::flush;

            // get minimum and maximum values
            for (int i = 0; i < m_layers.size(); i++) {
                auto layer = m_layers[i].get();
                layer->dense_output.values >> data::CPU;
                for (int m = 0; m < layer->dense_output.values.m; m++) {
                    for (int n = 0; n < layer->dense_output.values.n; n++) {
                        max_values[i](m, 0) =
                            std::max(max_values[i](m, 0), layer->dense_output.values(m, n));
                        min_values[i](m, 0) =
                            std::min(min_values[i](m, 0), layer->dense_output.values(m, n));
                    }
                }
            }
        }
        std::cout << std::endl;

        for (int i = 0; i < m_layers.size(); i++) {
            std::cout << "------------ LAYER " << i + 1 << " --------------------" << std::endl;
            std::cout << "min: ";
            for (int j = 0; j < std::min((size_t) 16, min_values[i].size()); j++) {
                std::cout << std::setw(10) << min_values[i](j);
            }
            if (min_values[i].size() > 16) {
                std::cout << " ......... " << min_values[i](min_values.size() - 1);
            }
            std::cout << "\n";

            std::cout << "max: ";
            for (int j = 0; j < std::min((size_t) 16, max_values[i].size()); j++) {
                std::cout << std::setw(10) << max_values[i](j);
            }
            if (max_values[i].size() > 16) {
                std::cout << " ......... " << max_values[i](max_values.size() - 1);
            }

            std::cout << "\n";
            float min = 10000000;
            float max = -10000000;
            for (int m = 0; m < min_values.size(); m++) {
                min = std::min(min, min_values[i](m));
                max = std::max(max, max_values[i](m));
            }
            std::cout << "output bounds: [" << min << " ; " << max << "]\n";

            int died = 0;
            for (int j = 0; j < max_values[i].size(); j++) {
                if (std::abs(max_values[i](j) - min_values[i](j)) < 1e-8) {
                    died++;
                }
            }

            std::cout << "died: " << died << " / " << max_values[i].size();
            std::cout << "\n";

            for (auto p : m_layers[i]->params()) {
                float min = 10000000;
                float max = -10000000;
                for (int m = 0; m < p->values.m; m++) {
                    for (int n = 0; n < p->values.n; n++) {
                        min = std::min(min, p->values(m, n));
                        max = std::max(max, p->values(m, n));
                    }
                }

                std::cout << "param bounds: [" << min << " ; " << max << "]\n";
            }
        }
    }
};

struct RiceModel : ChessModel {
    SparseInput* in1;
    SparseInput* in2;

    // clang-format off
    // King bucket indicies
    static constexpr int indices[64] = {
        0,  1,  2,  3,  3,  2,  1,  0,
        4,  5,  6,  7,  7,  6,  5,  4,
        8,  9,  10, 11, 11, 10, 9,  8,
        8,  9,  10, 11, 11, 10, 9,  8,
        12, 12, 13, 13, 13, 13, 12, 12,
        12, 12, 13, 13, 13, 13, 12, 12,
        14, 14, 15, 15, 15, 15, 14, 14,
        14, 14, 15, 15, 15, 15, 14, 14,
    };
    // clang-format on

    RiceModel(size_t n_ft, float lambda, size_t save_rate)
        : ChessModel() {
        in1     = add<SparseInput>(12 * 64 * 16, 32);
        in2     = add<SparseInput>(12 * 64 * 16, 32);

        auto ft = add<FeatureTransformer>(in1, in2, n_ft);
        auto re = add<ReLU>(ft);
        auto af = add<Affine>(re, 1);
        auto sm = add<Sigmoid>(af, 2.5 / 400);

        add_optimizer(AdamWarmup({{OptimizerEntry {&ft->weights}},
                                  {OptimizerEntry {&ft->bias}},
                                  {OptimizerEntry {&af->weights}},
                                  {OptimizerEntry {&af->bias}}},
                                 0.95,
                                 0.999,
                                 1e-8,
                                 5 * 16384));

        add_quantization(Quantizer {
            "quant",
            save_rate,
            QuantizerEntry<int16_t>(&ft->weights.values, 32, true),
            QuantizerEntry<int16_t>(&ft->bias.values, 32),
            QuantizerEntry<int16_t>(&af->weights.values, 128),
            QuantizerEntry<int32_t>(&af->bias.values, 32 * 128),
        });
        set_save_frequency(save_rate);
    }

    inline int king_square_index(int kingSquare, uint8_t kingColor) {
        kingSquare = (56 * kingColor) ^ kingSquare;
        return indices[kingSquare];
    }

    inline int
        index(uint8_t pieceType, uint8_t pieceColor, int square, uint8_t view, int kingSquare) {
        const int ksIndex = king_square_index(kingSquare, view);
        square            = square ^ (56 * view);
        square            = square ^ (7 * !!(kingSquare & 0x4));

        // clang-format off
        return square
            + pieceType * 64
            + !(pieceColor ^ view) * 64 * 6 + ksIndex * 64 * 6 * 2;
        // clang-format on
    }

    void setup_inputs_and_outputs(carbon::DataSet& positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static, 64) num_threads(6)
        for (int b = 0; b < positions.size(); b++) {
            const auto pos = positions[b].pos;
            // fill in the inputs and target values

            const auto wKingSq = pos.kingSquare(binpack::chess::Color::White);
            const auto bKingSq = pos.kingSquare(binpack::chess::Color::Black);

            const auto pieces  = pos.piecesBB();

            for (auto sq : pieces) {
                const auto         piece                 = pos.pieceAt(sq);
                const std::uint8_t pieceType             = static_cast<uint8_t>(piece.type());
                const std::uint8_t pieceColor            = static_cast<uint8_t>(piece.color());

                auto               piece_index_white_pov = index(pieceType,
                                                   pieceColor,
                                                   static_cast<int>(sq),
                                                   static_cast<uint8_t>(binpack::chess::Color::White),
                                                   static_cast<int>(wKingSq));
                auto               piece_index_black_pov = index(pieceType,
                                                   pieceColor,
                                                   static_cast<int>(sq),
                                                   static_cast<uint8_t>(binpack::chess::Color::Black),
                                                   static_cast<int>(bKingSq));

                if (pos.sideToMove() == binpack::chess::Color::White) {
                    in1->sparse_output.set(b, piece_index_white_pov);
                    in2->sparse_output.set(b, piece_index_black_pov);
                } else {
                    in2->sparse_output.set(b, piece_index_white_pov);
                    in1->sparse_output.set(b, piece_index_black_pov);
                }
            }

            float p_value = positions[b].score;
            float w_value = positions[b].result;

            // flip if black is to move -> relative network style
            // if (pos.sideToMove() == binpack::chess::Color::Black) {
            //     p_value = -p_value;
            //     w_value = -w_value;
            // }

            float p_target = 1 / (1 + expf(-p_value * 2.5 / 400));
            float w_target = (w_value + 1) / 2.0f;

            float actual_lambda =
                start_lambda + (end_lambda - start_lambda) * (current_epoch / max_epochs);

            target(b) = (actual_lambda * p_target + (1.0f - actual_lambda) * w_target) / 1.0f;
        }
    }
};

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("Grapheus");

    program.add_argument("data").required().help("Directory containing training files");
    program.add_argument("--output").required().help("Output directory for network files");
    program.add_argument("--resume").help("Weights file to resume from");
    program.add_argument("--epochs")
        .default_value(1000)
        .help("Total number of epochs to train for")
        .scan<'i', int>();
    program.add_argument("--save-rate")
        .default_value(10)
        .help("How frequently to save quantized networks + weights")
        .scan<'i', int>();
    program.add_argument("--ft-size")
        .default_value(512)
        .help("Number of neurons in the Feature Transformer")
        .scan<'i', int>();
    program.add_argument("--lambda")
        .default_value(0.7f)
        .help("Ratio of evaluation scored to use while training")
        .scan<'f', float>();
    program.add_argument("--lr")
        .default_value(0.01f)
        .help("The starting learning rate for the optimizer")
        .scan<'f', float>();
    program.add_argument("--batch-size")
        .default_value(16384)
        .help("Number of positions in a mini-batch during training")
        .scan<'i', int>();
    program.add_argument("--lr-drop-epoch")
        .default_value(1)
        .help("Epoch to execute an LR drop at")
        .scan<'i', int>();
    program.add_argument("--lr-drop-ratio")
        .default_value(0.995f)
        .help("How much to scale down LR when dropping")
        .scan<'f', float>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    math::seed(0);

    init();

    const int   total_epochs  = program.get<int>("--epochs");
    const int   save_rate     = program.get<int>("--save-rate");
    const int   ft_size       = program.get<int>("--ft-size");
    const float lambda        = program.get<float>("--lambda");
    const float lr            = program.get<float>("--lr");
    const int   batch_size    = program.get<int>("--batch-size");
    const int   lr_drop_epoch = program.get<int>("--lr-drop-epoch");
    const float lr_drop_ratio = program.get<float>("--lr-drop-ratio");

    std::cout << "Epochs: " << total_epochs << "\n"
              << "Save Rate: " << save_rate << "\n"
              << "FT Size: " << ft_size << "\n"
              << "Lambda: " << lambda << "\n"
              << "LR: " << lr << "\n"
              << "Batch: " << batch_size << "\n"
              << "LR Drop @ " << lr_drop_epoch << "\n"
              << "LR Drop R " << lr_drop_ratio << std::endl;

    carbon::DataLoader loader(program.get("data"), batch_size);
    loader.start();

    RiceModel model {static_cast<size_t>(ft_size), lambda, static_cast<size_t>(save_rate)};
    model.set_loss(MPE {2.5, true});
    model.set_lr_schedule(StepDecayLRSchedule {lr, lr_drop_ratio, lr_drop_epoch});

    auto output_dir = program.get("--output");
    model.set_file_output(output_dir);
    for (auto& quantizer : model.m_quantizers)
        quantizer.set_path(output_dir);

    std::cout << "Files will be saved to " << output_dir << std::endl;

    if (auto previous = program.present("--resume")) {
        model.load_weights(*previous);
        std::cout << "Loaded weights from previous " << *previous << std::endl;
    }

    model.train(loader, total_epochs, 1e8, lambda, lambda, 0);

    close();
    return 0;
}