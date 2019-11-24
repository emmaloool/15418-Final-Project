// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// CUDA variant of methods for lossless encoder
//
// Authors: Emma Liu (emmaliu@andrew.cmu.edu) and Kevin Geng (khg@andrew.cmu.edu)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_CUDA)
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/enc/vp8li_enc.h"

//------------------------------------------------------------------------------
// GetBestGreenToRed helper routine: GetPredictionCostCrossColorRed
//  (GetBestGreenToRed => GetPredictionCostCrossColorRed => VP8LCollectColorRedTransforms)

__device__ __inline__ static float GetPredictionCostCrossColorRed_CUDA(const uint32_t* argb, 
                                                                        int stride, int tile_width, int tile_height,
                                                                        VP8LMultipliers prev_x, VP8LMultipliers prev_y, int green_to_red,
                                                                        const int accumulated_red_histo[256]) {
    int histo[256] = { 0 }; 
    float cur_diff;

    // This function doesn't alter the contents of argb
    VP8LCollectColorRedTransforms(argb, stride, tile_width, tile_height, green_to_red, histo);

    cur_diff = PredictionCostCrossColor(accumulated_red_histo, histo);
    if ((uint8_t)green_to_red == prev_x.green_to_red_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
    }
    if ((uint8_t)green_to_red == prev_y.green_to_red_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
    }
    if (green_to_red == 0) {
        cur_diff -= 3;
    }
    return cur_diff;
}
