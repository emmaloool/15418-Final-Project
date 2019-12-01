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

// Computes sampled size of 'size' when sampling using 'sampling bits'.
__device__ __inline__
static uint32_t VP8LSubSampleSize_device(uint32_t size, uint32_t sampling_bits) {
    return (size + (1 << sampling_bits) - 1) >> sampling_bits;
}

__device__ __inline__
static void MultipliersClear(VP8LMultipliers* const m) {
  m->green_to_red_ = 0;
  m->green_to_blue_ = 0;
  m->red_to_blue_ = 0;
}

__device__ __inline__
static void ColorCodeToMultipliers(uint32_t color_code,
                                               VP8LMultipliers* const m) {
  m->green_to_red_  = (color_code >>  0) & 0xff;
  m->green_to_blue_ = (color_code >>  8) & 0xff;
  m->red_to_blue_   = (color_code >> 16) & 0xff;
}

__device__ __inline__
static uint32_t MultipliersToColorCode(
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

__device__ __inline__ float PredictionCostSpatial_device(
    const int counts[256], int weight_0, double exp_val) {
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

__device__ __inline__ float PredictionCostCrossColor_device(
        const int accumulated[256], const int counts[256]) {

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

__device__ __inline__ uint8_t TransformColorRed_device(
        uint8_t green_to_red, uint32_t argb) {
    const int8_t green = U32ToS8(argb >> 8);
    int new_red = argb >> 16;
    new_red -= ColorTransformDelta(green_to_red, green);
    return (new_red & 0xff);
}

__device__ __inline__ void CollectColorRedTransforms_device(
                                 const uint32_t* argb, int stride,
                                 int tile_width, int tile_height,
                                 int green_to_red, int histo[]) {

    // Position inside block (assume tile == block)
    int x = threadIdx.x;
    int y = threadIdx.y;

    if (x < tile_width && y < tile_height) {
        int transform_index = TransformColorRed_device(
            (uint8_t)green_to_red, argb[stride * y + x]);
        atomicAdd(&histo[transform_index], 1);
    }
    __syncthreads();
}


__device__ __inline__
static float GetPredictionCostCrossColorRed_device(
        const uint32_t* argb, int stride, int tile_width, int tile_height,
        VP8LMultipliers prev_x, VP8LMultipliers prev_y, int green_to_red,
        const int accumulated_red_histo[256]) {

    const int ind = threadIdx.y * tile_width + threadIdx.x;
    __shared__ int histo[256];
    if (ind < 256) {
        histo[ind] = 0;
    }
    __syncthreads();

    // NOTE: this work has been split over threads
    CollectColorRedTransforms_device(
        argb, stride, tile_width, tile_height,
        green_to_red, histo);

    __shared__ float cur_diff;

    // TODO: this work is currently duplicated by all threads
    if (ind == 0) {
        cur_diff = PredictionCostCrossColor_device(accumulated_red_histo, histo);
        if ((uint8_t)green_to_red == prev_x.green_to_red_) {
            cur_diff -= 3;  // favor keeping the areas locally similar
        }
        if ((uint8_t)green_to_red == prev_y.green_to_red_) {
            cur_diff -= 3;  // favor keeping the areas locally similar
        }
        if (green_to_red == 0) {
            cur_diff -= 3;
        }
    }
    __syncthreads();

    return cur_diff;
}


// Note: this function is unchanged (pass-through)
__device__ __inline__
static void GetBestGreenToRed_device(
        const uint32_t* argb, int stride, int tile_width, int tile_height,
        VP8LMultipliers prev_x, VP8LMultipliers prev_y, int quality,
        const int accumulated_red_histo[256], VP8LMultipliers* const best_tx) {

    const int kMaxIters = 4 + ((7 * quality) >> 8);  // in range [4..6]
    int green_to_red_best = 0;
    int iter, offset;
    float best_diff = GetPredictionCostCrossColorRed_device(
        argb, stride, tile_width, tile_height, prev_x, prev_y,
        green_to_red_best, accumulated_red_histo);
    for (iter = 0; iter < kMaxIters; ++iter) {
        // ColorTransformDelta is a 3.5 bit fixed point, so 32 is equal to
        // one in color computation. Having initial delta here as 1 is sufficient
        // to explore the range of (-2, 2).
        const int delta = 32 >> iter;
        // Try a negative and a positive delta from the best known value.
        for (offset = -delta; offset <= delta; offset += 2 * delta) {
            const int green_to_red_cur = offset + green_to_red_best;
            const float cur_diff = GetPredictionCostCrossColorRed_device(
                argb, stride, tile_width, tile_height, prev_x, prev_y,
                green_to_red_cur, accumulated_red_histo);
            if (cur_diff < best_diff) {
                best_diff = cur_diff;
                green_to_red_best = green_to_red_cur;
            }
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        best_tx->green_to_red_ = (green_to_red_best & 0xff);
    }
    __syncthreads();
}


//------------------------------------------------------------------------------
// Blue functions

__device__ __inline__ uint8_t TransformColorBlue_device(
        uint8_t green_to_blue, uint8_t red_to_blue, uint32_t argb) {
    const int8_t green = U32ToS8(argb >>  8);
    const int8_t red   = U32ToS8(argb >> 16);
    uint8_t new_blue = argb & 0xff;
    new_blue -= ColorTransformDelta(green_to_blue, green);
    new_blue -= ColorTransformDelta(red_to_blue, red);
    return (new_blue & 0xff);
}

__device__ __inline__ void CollectColorBlueTransforms_device(
        const uint32_t* argb, int stride,
        int tile_width, int tile_height,
        int green_to_blue, int red_to_blue, int histo[]) {

    // Position inside block (assume tile == block)
    int x = threadIdx.x;
    int y = threadIdx.y;

    if (x < tile_width && y < tile_height) {
        int transform_index = TransformColorBlue_device(
            (uint8_t)green_to_blue, (uint8_t)red_to_blue,
            argb[stride * y + x]);
        atomicAdd(&histo[transform_index], 1);
    }
    __syncthreads();
}

__device__
static float GetPredictionCostCrossColorBlue_device(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y,
    int green_to_blue, int red_to_blue, const int accumulated_blue_histo[256]) {

    const int ind = threadIdx.y * tile_width + threadIdx.x;
    __shared__ int histo[256];
    if (ind < 256) {
        histo[ind] = 0;
    }
    __syncthreads();

    // NOTE: this work is parallelized over threads
    CollectColorBlueTransforms_device(
        argb, stride, tile_width, tile_height,
        green_to_blue, red_to_blue, histo);

    __shared__ float cur_diff;

    // TODO: this work is currently duplicated by all threads
    if (ind == 0) {
        cur_diff = PredictionCostCrossColor_device(accumulated_blue_histo, histo);
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
    }
    __syncthreads();

    return cur_diff;
}

#define kGreenRedToBlueNumAxis 8
#define kGreenRedToBlueMaxIters 7

__device__
static void GetBestGreenRedToBlue_device(
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
  float best_diff = GetPredictionCostCrossColorBlue_device(
      argb, stride, tile_width, tile_height, prev_x, prev_y,
      green_to_blue_best, red_to_blue_best, accumulated_blue_histo);
  for (iter = 0; iter < iters; ++iter) {
    const int delta = delta_lut[iter];
    int axis;
    for (axis = 0; axis < kGreenRedToBlueNumAxis; ++axis) {
      const int green_to_blue_cur =
          offset[axis][0] * delta + green_to_blue_best;
      const int red_to_blue_cur = offset[axis][1] * delta + red_to_blue_best;
      const float cur_diff = GetPredictionCostCrossColorBlue_device(
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

  if (threadIdx.x == 0 && threadIdx.y == 0) {
      best_tx->green_to_blue_ = green_to_blue_best & 0xff;
      best_tx->red_to_blue_ = red_to_blue_best & 0xff;
  }
  __syncthreads();
}
#undef kGreenRedToBlueMaxIters
#undef kGreenRedToBlueNumAxis

//------------------------------------------------------------------------------
// GetBestColorTransformForTile + Subroutines

__device__ __inline__
static VP8LMultipliers GetBestColorTransformForTile_device(
        int tile_x, int tile_y, int bits,
        VP8LMultipliers prev_x,
        VP8LMultipliers prev_y,
        int quality, int xsize, int ysize,
        const int accumulated_red_histo[256],
        const int accumulated_blue_histo[256],
        const uint32_t* const argb) {

    const int max_tile_size = 1 << bits;
    const int tile_y_offset = tile_y * max_tile_size;
    const int tile_x_offset = tile_x * max_tile_size;
    const int all_x_max = min(tile_x_offset + max_tile_size, xsize);
    const int all_y_max = min(tile_y_offset + max_tile_size, ysize);
    const int tile_width = all_x_max - tile_x_offset;
    const int tile_height = all_y_max - tile_y_offset;
    const uint32_t* const tile_argb = argb + tile_y_offset * xsize + tile_x_offset;

    __shared__ VP8LMultipliers best_tx;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        MultipliersClear(&best_tx);
    }
    __syncthreads();

    GetBestGreenToRed_device(
            tile_argb, xsize, tile_width, tile_height,
            prev_x, prev_y, quality, accumulated_red_histo, &best_tx);

    GetBestGreenRedToBlue_device(
            tile_argb, xsize, tile_width, tile_height,
            prev_x, prev_y, quality, accumulated_blue_histo,
            &best_tx);

    return best_tx;
}

//------------------------------------------------------------------------------

__device__ __inline__
void TransformColor_device(const VP8LMultipliers* const m, uint32_t* data,
                           int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint32_t argb = data[i];
    const int8_t green = U32ToS8(argb >>  8);
    const int8_t red   = U32ToS8(argb >> 16);
    int new_red = red & 0xff;
    int new_blue = argb & 0xff;
    new_red -= ColorTransformDelta(m->green_to_red_, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta(m->green_to_blue_, green);
    new_blue -= ColorTransformDelta(m->red_to_blue_, red);
    new_blue &= 0xff;
    data[i] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
  }
}

__device__ __inline__
static void CopyTileWithColorTransform_device(
        int xsize, int ysize,
        int tile_x, int tile_y,
        int max_tile_size,
        VP8LMultipliers color_transform,
        uint32_t* argb) {

    const int xscan = min(max_tile_size, xsize - tile_x);
    int yscan = min(max_tile_size, ysize - tile_y);
    argb += tile_y * xsize + tile_x;
    while (yscan-- > 0) {
        TransformColor_device(&color_transform, argb, xscan);
        argb += xsize;
    }
}

//------------------------------------------------------------------------------
// ColorSpaceTransform

__global__ void
__launch_bounds__(1024)
ColorSpaceTransform_kernel(
        int width, int height, int bits, int quality,
        uint32_t* const argb, uint32_t* image,
        int accumulated_red_histo[256], int accumulated_blue_histo[256]) {

    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y;

    const int max_tile_size = 1 << bits;
    const int tile_xsize = VP8LSubSampleSize_device(width, bits);
    const int tile_ysize = VP8LSubSampleSize_device(height, bits);

    // TODO: prev_x and prev_y are always zeroed here
    __shared__ VP8LMultipliers prev_x, prev_y;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        MultipliersClear(&prev_y);
        MultipliersClear(&prev_x);
    }
    __syncthreads();

    const int tile_x_offset = tile_x * max_tile_size;
    const int tile_y_offset = tile_y * max_tile_size;
    const int all_x_max = min(tile_x_offset + max_tile_size, width);
    const int all_y_max = min(tile_y_offset + max_tile_size, height);
    const int offset = tile_y * tile_xsize + tile_x;
    // TODO: disabled to avoid nondeterminism
    //if (threadIdx.x == 0 && threadIdx.y == 0) {
    //    if (tile_y != 0) {
    //        ColorCodeToMultipliers(image[offset - tile_xsize], &prev_y);
    //    }
    //}
    //__syncthreads();

    // Note that device_accumulated_red_histo is passed as const.
    // So it won't be changed by this function call

    prev_x = GetBestColorTransformForTile_device(
            tile_x, tile_y, bits,
            prev_x, prev_y,
            quality, width, height,
            accumulated_red_histo,
            accumulated_blue_histo,
            argb);

    // These need to only be performed by one thread
    // TODO: parallelize this
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        image[offset] = MultipliersToColorCode(&prev_x);

        CopyTileWithColorTransform_device(
                width, height, tile_x_offset, tile_y_offset,
                max_tile_size, prev_x, argb);
    }
    __syncthreads();

    // Gather accumulated histogram data.
    // TODO: this is disabled due to nondeterminism
    //int y = tile_y_offset + threadIdx.y;
    //int ix = y * width + tile_x_offset + threadIdx.x;
    //int ix_end = y * width + all_x_max;

    //if (y < all_y_max && ix < ix_end) {
    //    const uint32_t pix = argb[ix];
    //    bool skip =
    //        (ix >= 2 && pix == argb[ix - 2] && pix == argb[ix - 1])
    //            // repeated pixels are handled by backward references
    //        ||
    //        (ix >= width + 2 && argb[ix - 2] == argb[ix - width - 2] &&
    //            argb[ix - 1] == argb[ix - width - 1] && pix == argb[ix - width])
    //            // repeated pixels are handled by backward references
    //        ;
    //    if (!skip) {
    //        ++accumulated_red_histo[(pix >> 16) & 0xff];
    //        ++accumulated_blue_histo[(pix >> 0) & 0xff];
    //    }
    //}
    //__syncthreads();
}


void VP8LColorSpaceTransform_CUDA(int width, int height, int bits, int quality,
                               uint32_t* const argb, uint32_t* image) {

    const int max_tile_size = 1 << bits;
    const int tile_xsize = VP8LSubSampleSize(width, bits);
    const int tile_ysize = VP8LSubSampleSize(height, bits);

    assert(max_tile_size == 32);
    assert(max_tile_size == 32);

    // Allocate device_argb, copying from argb
    uint32_t *device_argb;
    cudaCheckError(cudaMalloc(&device_argb, width * height * sizeof(uint32_t)));
    cudaCheckError(cudaMemcpy(device_argb, argb, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate device_image, uninitialized
    uint32_t *device_image;
    cudaCheckError(cudaMalloc(&device_image, tile_xsize * tile_ysize * sizeof(uint32_t)));

    // Allocate device_accumulated_red_histo, zeroed
    int *device_accumulated_red_histo;
    cudaCheckError(cudaMalloc(&device_accumulated_red_histo, 256 * sizeof(*device_accumulated_red_histo)));
    cudaCheckError(cudaMemset(device_accumulated_red_histo, 0, 256 * sizeof(*device_accumulated_red_histo)));

    // Allocate device_accumulated_blue_histo, zeroed
    int *device_accumulated_blue_histo;
    cudaCheckError(cudaMalloc(&device_accumulated_blue_histo, 256 * sizeof(*device_accumulated_blue_histo)));
    cudaCheckError(cudaMemset(device_accumulated_blue_histo, 0, 256 * sizeof(*device_accumulated_blue_histo)));

    // Perform kernel launch
    dim3 blockDim(max_tile_size, max_tile_size);
    dim3 gridDim(tile_xsize, tile_ysize);

    ColorSpaceTransform_kernel<<<gridDim, blockDim>>>(
        width, height, bits, quality,
        device_argb, device_image,
        device_accumulated_red_histo, device_accumulated_blue_histo);

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaMemcpy(argb, device_argb, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(image, device_image, tile_xsize * tile_ysize * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(device_argb));
    cudaCheckError(cudaFree(device_image));
    cudaCheckError(cudaFree(device_accumulated_red_histo));
    cudaCheckError(cudaFree(device_accumulated_blue_histo));
}



} // extern "C"

#endif // WEBP_USE_CUDA

