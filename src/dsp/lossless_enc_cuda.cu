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
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"

//------------------------------------------------------------------------------
// Subtract-Green Transform


__global__ void SubtractGreenFromBlueAndRed_kernel(uint32_t* argb_data, int num_pixels) {
    // Overall index from position of thread in current block, and given the block we are in.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pixels) return;    // loop guard: for 0 <= i < num_pixels

    const int argb = argb_data[index];
    const int green = (argb >> 8) & 0xff;
    const uint32_t new_r = (((argb >> 16) & 0xff) - green) & 0xff;
    const uint32_t new_b = (((argb >>  0) & 0xff) - green) & 0xff;
    argb_data[index] = (argb & 0xff00ff00u) | (new_r << 16) | new_b;
}

static void SubtractGreenFromBlueAndRed_CUDA(uint32_t* argb_data,
                                             int num_pixels) {

    // Compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory buffer for result on the GPU
    uint32_t* result;
    cudaMalloc(&result, num_pixels * sizeof(uint32_t));

    // Copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(result, argb_data, num_pixels * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    SubtractGreenFromBlueAndRed_kernel<<<blocks, threadsPerBlock>>>(result, num_pixels);

    // Copy result from GPU using cudaMemcpy
    cudaMemcpy(argb_data, result, num_pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

}

//------------------------------------------------------------------------------
// Color Transform

__device__ __inline__ int ColorTransformDelta(int8_t color_pred, int8_t color) {
    return ((int)color_pred * color) >> 5;
}

__device__ __inline__  int8_t U32ToS8(uint32_t v) {
    return (int8_t)(v & 0xff);
}

__global__ void TransformColor_kernel(const VP8LMultipliers* const m, 
                                            uint32_t* data, 
                                            int num_pixels) {

    // Overall index from position of thread in current block, and given the block we are in.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pixels) return;    // loop guard: for 0 <= i < num_pixels

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
    data[index] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);

}

static void TransformColor_CUDA(const VP8LMultipliers* const m, 
                                 uint32_t* data,
                                 int num_pixels) {
    // Compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory buffer for result on the GPU
    uint32_t* result;
    cudaMalloc(&result, num_pixels * sizeof(uint32_t));

    // Copy input params to the GPU using cudaMemcpy
    cudaMemcpy(result, data, num_pixels * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    TransformColor_kernel<<<blocks, threadsPerBlock>>>(m, result, num_pixels);

    // Copy result from GPU using cudaMemcpy
    cudaMemcpy(data, result, num_pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

}

//------------------------------------------------------------------------------
// Collect Color Red/Blue Transforms

__device__ __inline__ uint8_t TransformColorRed(uint8_t green_to_red,
                                                uint32_t argb) {
  const int8_t green = U32ToS8(argb >> 8);
  int new_red = argb >> 16;
  new_red -= ColorTransformDelta(green_to_red, green);
  return (new_red & 0xff);
}

__device__ __inline__ uint8_t TransformColorBlue(uint8_t green_to_blue,
                                                    uint8_t red_to_blue,
                                                    uint32_t argb) {
  const int8_t green = U32ToS8(argb >>  8);
  const int8_t red   = U32ToS8(argb >> 16);
  uint8_t new_blue = argb & 0xff;
  new_blue -= ColorTransformDelta(green_to_blue, green);
  new_blue -= ColorTransformDelta(red_to_blue, red);
  return (new_blue & 0xff);
}


/*void VP8LCollectColorRedTransforms_C(const uint32_t* argb, int stride,
                                     int tile_width, int tile_height,
                                     int green_to_red, int histo[]) {
  while (tile_height-- > 0) {
    int x;
    for (x = 0; x < tile_width; ++x) {
      ++histo[TransformColorRed((uint8_t)green_to_red, argb[x])];
    }
    argb += stride;
  }


// int y_limit = tile_height * stride;
// for (y = 1; y < tile_height; y++) {
//     for (x = 0; x < tile_width; x++) {
//         // I'm assuming that this line is equivalent to: 
//         // ++histo[TransformColorRed((uint8_t)green_to_red, argb[x])];
        
//     }
//     argb += (tile_height - 1 - y) * stride;
// }


// for (t_h = tile_height - 1; t_h > 0; t_h -= 1) {
//     for (x = 0; x < tile_width; ++x) {
//         // I'm assuming that this line is equivalent to: 
//         //++histo[TransformColorRed((uint8_t)green_to_red, argb[x])];
        
//     }
//     argb += stride;
// }


// while (tile_height-- > 0) {          // INCREMENT FIRST, so range is from (tile_height - 1) to 1
//     int x;
//     for (x = 0; x < tile_width; ++x) {
//       ++histo[TransformColorRed((uint8_t)green_to_red, argb[x])];
//     }
//     argb += stride;
// }
}*/


__device__ void CollectColorRedTransforms_kernel(const uint32_t* argb, int stride,
                                 int tile_width, int tile_height,
                                 int green_to_red, int histo[]) {

    // Overall index from position of thread in current block, and given the block we are in.
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    if (index >= tile_width * tile_height) return;    // loop guard: for 0 <= i < num_pixels

    // The hell is the length of histo???
    __shared__ int histo_temp;
    int transform_index = TransformColorRed((uint8_t)green_to_red, argb[(tile_height - 1 - threadIdy.y) * stride + index])];
    histo_temp[transform_index] = histo[transform_index];
    atomicAdd(histo_temp[transformindex]);

    __syncthreads();

    histo[transform_index] = histo_temp[transform_index];

}

static void CollectColorRedTransforms_CUDA(const uint32_t* argb, int stride,
                                     int tile_width, int tile_height,
                                     int green_to_red, int histo[]) {

    // Compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory buffer for result on the GPU
    uint32_t *argb_result;
    int histo_result[];
    cudaMalloc(&argb_result, tile_height * tile_width * sizeof(uint32_t));
    cudaMalloc(&histo_result, tile_height * tile_width * sizeof(int));

    // Copy input params to the GPU using cudaMemcpy
    cudaMemcpy(argb_result, argb, tile_height * tile_width * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(histo_result, histo, tile_height * tile_width * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    CollectColorRedTransforms_kernel<<<blocks, threadsPerBlock>>>(argb_result, stride, tile_width, tile_height, green_to_red, histo_result);

    // Copy result from GPU using cudaMemcpy
    cudaMemcpy(histo, histo_result, tile_height * tile_width * sizeof(int), cudaMemcpyDeviceToHost); //only histo modified

}


// void VP8LCollectColorBlueTransforms_C(const uint32_t* argb, int stride,
//                                       int tile_width, int tile_height,
//                                       int green_to_blue, int red_to_blue,
//                                       int histo[]) {
//   while (tile_height-- > 0) {
//     int x;
//     for (x = 0; x < tile_width; ++x) {
//       ++histo[TransformColorBlue((uint8_t)green_to_blue, (uint8_t)red_to_blue,
//                                  argb[x])];
//     }
//     argb += stride;
//   }
// }

__device__ void CollectColorBlueTransforms_kernel(const uint32_t* argb, int stride,
                                                    int tile_width, int tile_height,
                                                    int green_to_blue, int red_to_blue,
                                                    int histo[]) {

    // Overall index from position of thread in current block, and given the block we are in.
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    if (index >= tile_width * tile_height) return;    // loop guard: for 0 <= i < num_pixels

    // The hell is the length of histo???
    __shared__ int histo_temp;
    int transform_index = TransformColorBlue((uint8_t)green_to_blue, (uint8_t)red_to_blue, argb[(tile_height - 1 - threadIdy.y) * stride + index])];
    histo_temp[transform_index] = histo[transform_index];
    atomicAdd(histo_temp[transformindex]);

    __syncthreads();

    histo[transform_index] = histo_temp[transform_index];

}

static void CollectColorBlueTransforms_CUDA(const uint32_t* argb, int stride,
                                            int tile_width, int tile_height,
                                            int green_to_blue, int red_to_blue,
                                            int histo[]) {

    // Compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory buffer for result on the GPU
    uint32_t *argb_result;
    int histo_result[];
    cudaMalloc(&argb_result, tile_height * tile_width * sizeof(uint32_t));
    cudaMalloc(&histo_result, tile_height * tile_width * sizeof(int));

    // Copy input params to the GPU using cudaMemcpy
    cudaMemcpy(argb_result, argb, tile_height * tile_width * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(histo_result, histo, tile_height * tile_width * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    CollectColorBlueTransforms_kernel<<<blocks, threadsPerBlock>>>(argb_result, stride, tile_width, tile_height, green_to_blue, red_to_blue, histo_result);

    // Copy result from GPU using cudaMemcpy
    cudaMemcpy(histo, histo_result, tile_height * tile_width * sizeof(int), cudaMemcpyDeviceToHost); //only histo modified

}


//------------------------------------------------------------------------------
// Entry point

extern "C" void VP8LEncDspInitCUDA(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitCUDA(void) {
    VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_CUDA;
    VP8LTransformColor = TransformColor_CUDA;
    VP8LCollectColorRedTransforms = CollectColorRedTransforms_CUDA;
    VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_CUDA;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(VP8LEncDspInitCUDA)

#endif  // WEBP_USE_SSE2
