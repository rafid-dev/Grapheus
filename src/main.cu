
#include "dataset/dataset.h"
#include "dataset/process.h"
#include "dataset/io.h"
#include "misc/timer.h"
#include "nn/nn.h"
#include "operations/operations.h"
#include "chess/chess.h"
#include "dataset/batchloader.h"

#include <fstream>

using namespace nn;
using namespace data;

struct KoiModel : nn::Model{

    static constexpr int B   = 16384;   // positions per batch
    static constexpr int BPE = 1e8 / B; // batches per epoch
    static constexpr int E   = 1000;    // maximum epochs

    static constexpr int T   = 16;      // threads to use on the cpu

    SparseInput* in1;
    SparseInput* in2;

    KoiModel() {
        in1 = add<SparseInput>(16 * 12 * 64, 32);
        in2 = add<SparseInput>(16 * 12 * 64, 32);

        auto ft = add<FeatureTransformer>(in1, in2, 512);
        auto re = add<ReLU>(ft);
        auto af = add<Affine>(re,1);
        auto sm = add<Sigmoid>(af, 2.5 / 400);

        set_loss(MPE {2.5, false});
        set_lr_schedule(StepDecayLRSchedule{0.01,0.3,100});
        add_optimizer(Adam({
                 {OptimizerEntry{&ft->weights}},
                 {OptimizerEntry{&ft->bias   }},
                 {OptimizerEntry{&af->weights}},
                 {OptimizerEntry{&af->bias   }}
        }, 0.95, 0.999, 1e-7));

        this->compile(B);
    }


    inline int king_square_index(chess::Square relative_king_square) {

        // clang-format off
        constexpr int indices[chess::N_SQUARES] {
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

        return indices[relative_king_square];
    }

    inline int index(chess::Square piece_square,
                     chess::Piece  piece,
                     chess::Square king_square,
                     chess::Color  view) {
        constexpr int          PIECE_TYPE_FACTOR  = 64;
        constexpr int          PIECE_COLOR_FACTOR = PIECE_TYPE_FACTOR * 6;
        constexpr int          KING_SQUARE_FACTOR = PIECE_COLOR_FACTOR * 2;

        const chess::PieceType piece_type         = chess::type_of(piece);
        const chess::Color     piece_color        = chess::color_of(piece);


        chess::Square relative_king_square;
        chess::Square relative_piece_square;

        if (view == chess::WHITE) {
            relative_king_square  = king_square;
            relative_piece_square = piece_square;
        } else {
            relative_king_square  = chess::mirror_ver(king_square);
            relative_piece_square = chess::mirror_ver(piece_square);
        }

        const int king_square_idx = king_square_index(relative_king_square);
        if (chess::file_index(king_square) > 3) {
            relative_piece_square = chess::mirror_hor(relative_piece_square);
        }

        const int index =   relative_piece_square
                          + piece_type            * PIECE_TYPE_FACTOR
                          + (piece_color == view) * PIECE_COLOR_FACTOR
                          + king_square_idx       * KING_SQUARE_FACTOR;
        return index;
    }

    void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions){
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(T)
        for(int b = 0; b < B; b++){
            chess::Position* pos = &positions->positions[b];
            // fill in the inputs and target values

            chess::Square wKingSq = pos->get_king_square<chess::WHITE>();
            chess::Square bKingSq = pos->get_king_square<chess::BLACK>();

            chess::BB     bb {pos->m_occupancy};
            int    idx = 0;

            while (bb) {
                chess::Square sq                    = chess::lsb(bb);
                chess::Piece  pc                    = pos->m_pieces.get_piece(idx);

                auto   piece_index_white_pov = index(sq, pc, wKingSq, chess::WHITE);
                auto   piece_index_black_pov = index(sq, pc, bKingSq, chess::BLACK);

                if (pos->m_meta.stm() == chess::WHITE) {
                    in1->sparse_output.set(b, piece_index_white_pov);
                    in2->sparse_output.set(b, piece_index_black_pov);
                } else {
                    in2->sparse_output.set(b, piece_index_white_pov);
                    in1->sparse_output.set(b, piece_index_black_pov);
                }

                bb = chess::lsb_reset(bb);
                idx++;
            }

            float p_value = pos->m_result.score;
            float w_value = pos->m_result.wdl;

            // flip if black is to move -> relative network style
            if (pos->m_meta.stm() == chess::BLACK) {
                p_value = -p_value;
                w_value = -w_value;
            }

            float p_target = 1 / (1 + expf(-p_value * 2.5 / 400));
            float w_target = (w_value + 1) / 2.0f;

            target(b) = (p_target + w_target) / 2;

        }

    }

    void train(dataset::BatchLoader<chess::Position>& loader){

        Timer t {};
        for(int i = 0; i < E; i++){
            t.start();
            size_t prev_duration = 0;
            float batch_loss = 0;
            float epoch_loss = 0;

            for(int b = 0; b < BPE; b++){
                // get the next dataset and set it up while the other things
                // are running on the gpu
                auto* ds = loader.next();
                setup_inputs_and_outputs(ds);

                // print score of last iteration
                if(b > 0) {
                    batch_loss  = loss_of_last_batch();
                    epoch_loss += batch_loss;
                }

                t.stop();
                if (b > 0 && (b == BPE-1 || t.elapsed() - prev_duration > 1000)) {
                    prev_duration = t.elapsed();

                    printf("\rep/ba = [%3d/%5d], ", i, b);
                    printf("batch_loss = [%1.8f], ", batch_loss);
                    printf("epoch_loss = [%1.8f], ", epoch_loss / b);
                    printf("speed = [%9d pos/s], ", (int) round(1000.0f * B * b / t.elapsed()));
                    printf("time = [%3ds]", (int) t.elapsed() / 1000);
                    std::cout << std::flush;
                }

                // start training of new batch
                batch();
            }

            std::cout << std::endl;
            next_epoch();
        }
    }
};

int main() {
    init();

    std::vector<std::string> files{};
    files.push_back(R"(D:\Koivisto Resourcen\Training Data Shuffled + Berserk\koi_ber_1.bin)");
    files.push_back(R"(D:\Koivisto Resourcen\Training Data Shuffled + Berserk\koi_ber_2.bin)");
    files.push_back(R"(D:\Koivisto Resourcen\Training Data Shuffled + Berserk\koi_ber_3.bin)");

    dataset::BatchLoader<chess::Position> loader{files, KoiModel::B};
    loader.start();

    KoiModel model{};
    model.train(loader);

    loader.kill();

    close();
    return 0;
}