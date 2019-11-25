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

extern "C" {

#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
            cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//------------------------------------------------------------------------------

static WEBP_INLINE void MultipliersClear(VP8LMultipliers* const m) {
  m->green_to_red_ = 0;
  m->green_to_blue_ = 0;
  m->red_to_blue_ = 0;
}

static WEBP_INLINE int GetMin(int a, int b) { return (a > b) ? b : a; }

static WEBP_INLINE void ColorCodeToMultipliers(uint32_t color_code,
                                               VP8LMultipliers* const m) {
  m->green_to_red_  = (color_code >>  0) & 0xff;
  m->green_to_blue_ = (color_code >>  8) & 0xff;
  m->red_to_blue_   = (color_code >> 16) & 0xff;
}

static WEBP_INLINE uint32_t MultipliersToColorCode(
    const VP8LMultipliers* const m) {
  return 0xff000000u |
         ((uint32_t)(m->red_to_blue_) << 16) |
         ((uint32_t)(m->green_to_blue_) << 8) |
         m->green_to_red_;
}

//------------------------------------------------------------------------------

__device__ __inline__ float SlowSLog2_device(float x) {
    return x * log2(x);
}

// Compute the combined Shanon's entropy for distribution {X} and {X+Y}
__device__ __inline__ float CombinedShannonEntropy_device(const int X[256], const int Y[256]) {
  int i;
  double retval = 0.;
  int sumX = 0, sumXY = 0;
  for (i = 0; i < 256; ++i) {
    const int x = X[i];
    if (x != 0) {
      const int xy = x + Y[i];
      sumX += x;
      retval -= SlowSLog2_device(x);
      sumXY += xy;
      retval -= SlowSLog2_device(xy);
    } else if (Y[i] != 0) {
      sumXY += Y[i];
      retval -= SlowSLog2_device(Y[i]);
    }
  }
  retval += SlowSLog2_device(sumX) + SlowSLog2_device(sumXY);
  return (float)retval;
}

__device__ __inline__ float PredictionCostSpatial_device(const int counts[256], int weight_0,
                                   double exp_val) {
  const int significant_symbols = 256 >> 4;
  const double exp_decay_factor = 0.6;
  double bits = weight_0 * counts[0];
  int i;
  for (i = 1; i < significant_symbols; ++i) {
    bits += exp_val * (counts[i] + counts[256 - i]);
    exp_val *= exp_decay_factor;
  }
  return (float)(-0.1 * bits);
}

__device__ __inline__ float PredictionCostCrossColor_device(const int accumulated[256],
                                      const int counts[256]) {
  // Favor low entropy, locally and globally.
  // Favor small absolute values for PredictionCostSpatial
  static const double kExpValue = 2.4;
  return CombinedShannonEntropy_device(counts, accumulated) +
         PredictionCostSpatial_device(counts, 3, kExpValue);
}

//------------------------------------------------------------------------------
// Red functions

__device__ __inline__ int ColorTransformDelta(int8_t color_pred, int8_t color) {
    return ((int)color_pred * color) >> 5;
}

__device__ __inline__  int8_t U32ToS8(uint32_t v) {
    return (int8_t)(v & 0xff);
}

__device__ __inline__ uint8_t TransformColorRed(uint8_t green_to_red,
                                                uint32_t argb) {
    const int8_t green = U32ToS8(argb >> 8);
    int new_red = argb >> 16;
    new_red -= ColorTransformDelta(green_to_red, green);
    return (new_red & 0xff);
}

__device__ __inline__ void CollectColorRedTransforms_device(
                                 const uint32_t* argb, int stride,
                                 int tile_width, int tile_height,
                                 int green_to_red, int histo[]) {

    // Overall index from position of thread in current block, and given the block we are in.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < tile_width && y < tile_height) {
        int transform_index = TransformColorRed((uint8_t)green_to_red, argb[stride * y + x]);
        atomicAdd(&histo[transform_index], 1);
    }
}

__global__ void GetPredictionCostCrossColorRed_kernel(
        const uint32_t* device_argb, int stride, int tile_width, int tile_height,
        VP8LMultipliers prev_x, VP8LMultipliers prev_y, int kMaxIters,
        const int device_accumulated_red_histo[256],
        float *device_diff_results) {

    const int green_to_red_offset = -(1 << kMaxIters) + 1;
    const int green_to_red_multiplier = 64 >> kMaxIters;

    const int results_index = blockIdx.z;
    const int green_to_red = green_to_red_multiplier * (results_index + green_to_red_offset);

    const int ind = threadIdx.y * tile_width + threadIdx.x;
    __shared__ int histo[256];
    if (ind < 256) {
        histo[ind] = 0;
    }
    __syncthreads();

    CollectColorRedTransforms_device(device_argb, stride, tile_width, tile_height,
                                green_to_red, histo);

    float cur_diff = PredictionCostCrossColor_device(device_accumulated_red_histo, histo);
    if ((uint8_t)green_to_red == prev_x.green_to_red_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
    }
    if ((uint8_t)green_to_red == prev_y.green_to_red_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
    }
    if (green_to_red == 0) {
        cur_diff -= 3;
    }

    device_diff_results[results_index] = cur_diff;
}

static void GetBestGreenToRed(
        const uint32_t* device_argb, int stride, int tile_width, int tile_height,
        VP8LMultipliers prev_x, VP8LMultipliers prev_y, int quality,
        const int device_accumulated_red_histo[256], VP8LMultipliers* const best_tx) {
    const int kMaxIters = 4 + ((7 * quality) >> 8);  // in range [4..6]

    // If kMaxIters is 6, then the largest possible deviation is 2^7 - 1 = 127.
    // Then the offset value for green_to_red will be 1 - 2^6 = -63.
    const int num_results = (2 << kMaxIters) - 1;
    const int green_to_red_offset = -(1 << kMaxIters) + 1;
    const int green_to_red_multiplier = 64 >> kMaxIters;

    // Allocate array for cur_diff results
    float *device_diff_results;
    cudaCheckError(cudaMalloc(&device_diff_results, num_results * sizeof(*device_diff_results))); 

    assert(tile_width <= 32);
    assert(tile_height <= 32);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(1, 1, num_results);

    GetPredictionCostCrossColorRed_kernel<<<gridDim, blockDim>>>(
                          device_argb, stride, tile_width, tile_height, prev_x, prev_y,
                          kMaxIters,
                          device_accumulated_red_histo, device_diff_results);

    float diff_results[num_results];
    cudaCheckError(cudaMemcpy(diff_results, device_diff_results, num_results * sizeof(*diff_results), cudaMemcpyDeviceToHost));

    int green_to_red_best = 0;
    float best_diff = INFINITY;
    for (int i = 0; i < num_results; ++i) {
        float cur_diff = diff_results[i];
        int green_to_red_cur = green_to_red_multiplier * (i + green_to_red_offset);
        if (cur_diff < best_diff) {
            best_diff = cur_diff;
            green_to_red_best = green_to_red_cur;
        }
    }

    best_tx->green_to_red_ = (green_to_red_best & 0xff);
    cudaCheckError(cudaFree(device_diff_results));
}


//------------------------------------------------------------------------------
// Blue functions

static float PredictionCostSpatial(const int counts[256], int weight_0,
                                   double exp_val) {
  const int significant_symbols = 256 >> 4;
  const double exp_decay_factor = 0.6;
  double bits = weight_0 * counts[0];
  int i;
  for (i = 1; i < significant_symbols; ++i) {
    bits += exp_val * (counts[i] + counts[256 - i]);
    exp_val *= exp_decay_factor;
  }
  return (float)(-0.1 * bits);
}

static float PredictionCostCrossColor(const int accumulated[256],
                                      const int counts[256]) {
  // Favor low entropy, locally and globally.
  // Favor small absolute values for PredictionCostSpatial
  static const double kExpValue = 2.4;
  return VP8LCombinedShannonEntropy(counts, accumulated) +
         PredictionCostSpatial(counts, 3, kExpValue);
}

static float GetPredictionCostCrossColorBlue(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y,
    int green_to_blue, int red_to_blue, const int accumulated_blue_histo[256]) {
  int histo[256] = { 0 };
  float cur_diff;

  VP8LCollectColorBlueTransforms(argb, stride, tile_width, tile_height,
                                 green_to_blue, red_to_blue, histo);

  cur_diff = PredictionCostCrossColor(accumulated_blue_histo, histo);
  if ((uint8_t)green_to_blue == prev_x.green_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)green_to_blue == prev_y.green_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)red_to_blue == prev_x.red_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)red_to_blue == prev_y.red_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if (green_to_blue == 0) {
    cur_diff -= 3;
  }
  if (red_to_blue == 0) {
    cur_diff -= 3;
  }
  return cur_diff;
}

#define kGreenRedToBlueNumAxis 8
#define kGreenRedToBlueMaxIters 7
static void GetBestGreenRedToBlue(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int quality,
    const int accumulated_blue_histo[256],
    VP8LMultipliers* const best_tx) {
  const int8_t offset[kGreenRedToBlueNumAxis][2] =
      {{0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const int8_t delta_lut[kGreenRedToBlueMaxIters] = { 16, 16, 8, 4, 2, 2, 2 };
  const int iters =
      (quality < 25) ? 1 : (quality > 50) ? kGreenRedToBlueMaxIters : 4;
  int green_to_blue_best = 0;
  int red_to_blue_best = 0;
  int iter;
  // Initial value at origin:
  float best_diff = GetPredictionCostCrossColorBlue(
      argb, stride, tile_width, tile_height, prev_x, prev_y,
      green_to_blue_best, red_to_blue_best, accumulated_blue_histo);
  for (iter = 0; iter < iters; ++iter) {
    const int delta = delta_lut[iter];
    int axis;
    for (axis = 0; axis < kGreenRedToBlueNumAxis; ++axis) {
      const int green_to_blue_cur =
          offset[axis][0] * delta + green_to_blue_best;
      const int red_to_blue_cur = offset[axis][1] * delta + red_to_blue_best;
      const float cur_diff = GetPredictionCostCrossColorBlue(
          argb, stride, tile_width, tile_height, prev_x, prev_y,
          green_to_blue_cur, red_to_blue_cur, accumulated_blue_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_blue_best = green_to_blue_cur;
        red_to_blue_best = red_to_blue_cur;
      }
      if (quality < 25 && iter == 4) {
        // Only axis aligned diffs for lower quality.
        break;  // next iter.
      }
    }
    if (delta == 2 && green_to_blue_best == 0 && red_to_blue_best == 0) {
      // Further iterations would not help.
      break;  // out of iter-loop.
    }
  }
  best_tx->green_to_blue_ = green_to_blue_best & 0xff;
  best_tx->red_to_blue_ = red_to_blue_best & 0xff;
}
#undef kGreenRedToBlueMaxIters
#undef kGreenRedToBlueNumAxis

//------------------------------------------------------------------------------
// GetBestColorTransformForTile + Subroutines

static VP8LMultipliers GetBestColorTransformForTile(int tile_x, int tile_y, int bits,
                                                    VP8LMultipliers prev_x,
                                                    VP8LMultipliers prev_y,
                                                    int quality, int xsize, int ysize,
                                                    const int device_accumulated_red_histo[256],
                                                    const int accumulated_blue_histo[256],
                                                    const uint32_t* const argb,
                                                    const uint32_t* const device_argb) {

    const int max_tile_size = 1 << bits;
    const int tile_y_offset = tile_y * max_tile_size;
    const int tile_x_offset = tile_x * max_tile_size;
    const int all_x_max = GetMin(tile_x_offset + max_tile_size, xsize);
    const int all_y_max = GetMin(tile_y_offset + max_tile_size, ysize);
    const int tile_width = all_x_max - tile_x_offset;
    const int tile_height = all_y_max - tile_y_offset;
    const uint32_t* const tile_argb = argb + tile_y_offset * xsize + tile_x_offset;
    const uint32_t* const device_tile_argb = device_argb + tile_y_offset * xsize + tile_x_offset;
    VP8LMultipliers best_tx;
    MultipliersClear(&best_tx);

    GetBestGreenToRed(device_tile_argb, xsize, tile_width, tile_height,
                    prev_x, prev_y, quality, device_accumulated_red_histo, &best_tx);
    GetBestGreenRedToBlue(tile_argb, xsize, tile_width, tile_height,
                        prev_x, prev_y, quality, accumulated_blue_histo,
                        &best_tx);
    return best_tx;
}

//------------------------------------------------------------------------------

static void CopyTileWithColorTransform(int xsize, int ysize,
                                       int tile_x, int tile_y,
                                       int max_tile_size,
                                       VP8LMultipliers color_transform,
                                       uint32_t* argb) {
  const int xscan = GetMin(max_tile_size, xsize - tile_x);
  int yscan = GetMin(max_tile_size, ysize - tile_y);
  argb += tile_y * xsize + tile_x;
  while (yscan-- > 0) {
    VP8LTransformColor(&color_transform, argb, xscan);
    argb += xsize;
  }
}

//------------------------------------------------------------------------------
// ColorSpaceTransform

void VP8LColorSpaceTransform_CUDA(int width, int height, int bits, int quality,
                               uint32_t* const argb, uint32_t* image) {

    uint32_t *device_argb;
    cudaCheckError(cudaMalloc(&device_argb, width * height * sizeof(uint32_t)));
    cudaCheckError(cudaMemcpy(device_argb, argb, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice)); //only histo modified

    int *device_accumulated_red_histo;
    cudaCheckError(cudaMalloc(&device_accumulated_red_histo, 256 * sizeof(*device_accumulated_red_histo)));
    cudaCheckError(cudaMemset(device_accumulated_red_histo, 0, 256 * sizeof(*device_accumulated_red_histo)));

    const int max_tile_size = 1 << bits;
    const int tile_xsize = VP8LSubSampleSize(width, bits);
    const int tile_ysize = VP8LSubSampleSize(height, bits);
    int accumulated_red_histo[256] = { 0 };
    int accumulated_blue_histo[256] = { 0 };
    int tile_x, tile_y;
    VP8LMultipliers prev_x, prev_y;
    MultipliersClear(&prev_y);
    MultipliersClear(&prev_x);
    for (tile_y = 0; tile_y < tile_ysize; ++tile_y) {
        for (tile_x = 0; tile_x < tile_xsize; ++tile_x) {
            int y;
            const int tile_x_offset = tile_x * max_tile_size;
            const int tile_y_offset = tile_y * max_tile_size;
            const int all_x_max = GetMin(tile_x_offset + max_tile_size, width);
            const int all_y_max = GetMin(tile_y_offset + max_tile_size, height);
            const int offset = tile_y * tile_xsize + tile_x;
            if (tile_y != 0) {
                ColorCodeToMultipliers(image[offset - tile_xsize], &prev_y);
            }

            // Note that device_accumulated_red_histo is passed as const.
            // So it won't be changed by this function call

            prev_x = GetBestColorTransformForTile(tile_x, tile_y, bits,
                                                prev_x, prev_y,
                                                quality, width, height,
                                                device_accumulated_red_histo,
                                                accumulated_blue_histo,
                                                argb, device_argb);
            image[offset] = MultipliersToColorCode(&prev_x);
            CopyTileWithColorTransform(width, height, tile_x_offset, tile_y_offset,
                                     max_tile_size, prev_x, argb);

            // Gather accumulated histogram data.
            for (y = tile_y_offset; y < all_y_max; ++y) {
                int ix = y * width + tile_x_offset;
                const int ix_end = ix + all_x_max - tile_x_offset;
                for (; ix < ix_end; ++ix) {
                    const uint32_t pix = argb[ix];
                    if (ix >= 2 && pix == argb[ix - 2] && pix == argb[ix - 1]) {
                        continue;  // repeated pixels are handled by backward references
                    }
                    if (ix >= width + 2 && argb[ix - 2] == argb[ix - width - 2] &&
                        argb[ix - 1] == argb[ix - width - 1] && pix == argb[ix - width]) {
                        continue;  // repeated pixels are handled by backward references
                    }
                    ++accumulated_red_histo[(pix >> 16) & 0xff];
                    ++accumulated_blue_histo[(pix >> 0) & 0xff];
                }
            }

            cudaCheckError(cudaMemcpy(device_accumulated_red_histo, accumulated_red_histo, 
                                256 * sizeof(int), cudaMemcpyHostToDevice));
            // todo: make faster
            cudaCheckError(cudaMemcpy(device_argb, argb, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    cudaCheckError(cudaFree(device_argb));
    cudaCheckError(cudaFree(device_accumulated_red_histo));
}

} // extern "C"

#endif // WEBP_USE_CUDA

