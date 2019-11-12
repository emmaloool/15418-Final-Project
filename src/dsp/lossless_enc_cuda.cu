// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE2 variant of methods for lossless encoder
//
// Author: Skal (pascal.massimino@gmail.com)

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


__global__ void VP8LSubtractGreenFromBlueAndRed_kernel(uint32_t* argb_data, int num_pixels) {
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

  printf("HELLO WORLD WE ARE CALLING CUDA FUNCTION\n");

  // Compute number of blocks and threads per block
  const int threadsPerBlock = 512;
  const int blocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

  // Allocate device memory buffer for result on the GPU
  uint32_t* result;
  cudaMalloc(&result, num_pixels * sizeof(uint32_t));

  // Copy input arrays to the GPU using cudaMemcpy
  cudaMemcpy(result, argb_data, num_pixels * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
  VP8LSubtractGreenFromBlueAndRed_kernel<<<blocks, threadsPerBlock>>>(result, num_pixels);

  // Copy result from GPU using cudaMemcpy
  cudaMemcpy(argb_data, result, num_pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);

}

//------------------------------------------------------------------------------
// Entry point

extern "C" void VP8LEncDspInitCUDA(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitCUDA(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_CUDA;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(VP8LEncDspInitCUDA)

#endif  // WEBP_USE_SSE2
