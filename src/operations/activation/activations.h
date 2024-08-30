
#pragma once

#include "activation.h"

namespace operations {

DEFINE_ACTIVATION(linear, A[ida] * scalar, B_grd[idb] * scalar);

DEFINE_ACTIVATION(sigmoid,
                  1.0f / (1 + exp(-A[ida] * scalar)),
                  B_grd[idb] * B[idb] * (1 - B[idb]) * scalar);

DEFINE_ACTIVATION(relu, A[ida] > 0 ? A[ida] : 0, A[ida] > 0 ? B_grd[idb] : 0);

DEFINE_ACTIVATION(crelu,
                  A[ida] > 0 ? (A[ida] < scalar ? A[ida] : scalar) : 0,
                  A[ida] > 0 && A[ida] < scalar ? B_grd[idb] : 0);

DEFINE_ACTIVATION(lrelu,
                  A[ida] > 0 ? A[ida] : A[ida] * scalar,
                  A[ida] > 0 ? B_grd[idb] : B_grd[idb] * scalar);

#define crelu2_forward_expr                                                                          \
    float val = A[ida] > 0 ? (A[ida] < scalar ? A[ida] : scalar) : 0;                                \
    B[idb]    = val * val;

DEFINE_ACTIVATION_2(crelu2, crelu2_forward_expr, A[ida] > 0 && A[ida] < scalar ? 2 * B_grd[idb] : 0);

}    // namespace operations
