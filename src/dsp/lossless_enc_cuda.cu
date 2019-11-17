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


#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
            cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
    cudaCheckError(cudaMalloc(&result, num_pixels * sizeof(uint32_t)));

    // Copy input arrays to the GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(result, argb_data, num_pixels * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    SubtractGreenFromBlueAndRed_kernel<<<blocks, threadsPerBlock>>>(result, num_pixels);

    // Copy result from GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(argb_data, result, num_pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost))

    cudaCheckError(cudaFree(result));
}

//------------------------------------------------------------------------------
// Color Transform

__device__ __inline__ int ColorTransformDelta(int8_t color_pred, int8_t color) {
    return ((int)color_pred * color) >> 5;
}

__device__ __inline__  int8_t U32ToS8(uint32_t v) {
    return (int8_t)(v & 0xff);
}

__global__ void TransformColor_kernel(int green_to_red, int green_to_blue, int red_to_blue, 
                                            uint32_t* data, 
                                            int num_pixels) {

    // Overall index from position of thread in current block, and given the block we are in.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pixels) return;    // loop guard: for 0 <= i < num_pixels

    const uint32_t argb = data[index];
    const int8_t green = U32ToS8(argb >>  8);
    const int8_t red   = U32ToS8(argb >> 16);
    int new_red = red & 0xff;
    int new_blue = argb & 0xff;
    new_red -= ColorTransformDelta(green_to_red, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta(green_to_blue, green);
    new_blue -= ColorTransformDelta(red_to_blue, red);
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

    cudaCheckError(cudaMalloc(&result, num_pixels * sizeof(uint32_t)));

    // Copy input params to the GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(result, data, num_pixels * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    TransformColor_kernel<<<blocks, threadsPerBlock>>>(m->green_to_red_, m->green_to_blue_, m->red_to_blue_, result, num_pixels);

    // Copy result from GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(data, result, num_pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(result));
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

__global__ void CollectColorRedTransforms_kernel(const uint32_t* argb, int stride,
                                 int tile_width, int tile_height,
                                 int green_to_red, int histo[]) {

    // Overall index from position of thread in current block, and given the block we are in.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = y * tile_width + x;

    __shared__ int histo_temp[256];
    if (ind < 256) histo_temp[ind] = 0;
    __syncthreads();

    if (x < tile_width && y < tile_height) {
        int transform_index = TransformColorRed((uint8_t)green_to_red, argb[stride * y + x]);
        atomicAdd(&histo_temp[transform_index], 1);
    }
    __syncthreads();

    if (ind < 256) atomicAdd(&histo[ind], histo_temp[ind]);
    __syncthreads();
}

static void CollectColorRedTransforms_CUDA(const uint32_t* argb, int stride,
                                     int tile_width, int tile_height,
                                     int green_to_red, int histo[]) {
      // Dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((tile_width  + blockDim.x - 1) / blockDim.x,
                 (tile_height + blockDim.y - 1) / blockDim.y);

    size_t argb_size = tile_height * (stride - 1) + tile_width;

    // Allocate device memory buffer for result on the GPU
    uint32_t *argb_result;
    int *histo_result;
    //printf("tile_height * stride = %d * %d = %d\n", tile_height, stride, tile_height * stride);
    cudaCheckError(cudaMalloc(&histo_result, 256 * sizeof(int)));
    cudaCheckError(cudaMalloc(&argb_result, argb_size * sizeof(uint32_t)));

    // Copy input params to the GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(histo_result, histo, 256 * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(argb_result, argb, argb_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    CollectColorRedTransforms_kernel<<<gridDim, blockDim>>>(argb_result, stride, tile_width, tile_height, green_to_red, histo_result);

    // Copy result from GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(histo, histo_result, 256 * sizeof(int), cudaMemcpyDeviceToHost)); //only histo modified

    cudaCheckError(cudaFree(argb_result));
    cudaCheckError(cudaFree(histo_result));

}


__global__ void CollectColorBlueTransforms_kernel(const uint32_t* argb, int stride,
                                                    int tile_width, int tile_height,
                                                    int green_to_blue, int red_to_blue,
                                                    int histo[]) {
    // Overall index from position of thread in current block, and given the block we are in.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ int histo_temp[256];
    if (ind < 256) histo_temp[ind] = 0;
    __syncthreads();

    if (x < tile_width && y < tile_height) {
        //printf("stride * y + x = %d * %d + %d = %d\n", );
        int transform_index = TransformColorBlue((uint8_t)green_to_blue, (uint8_t)red_to_blue, argb[stride * y + x]);
        // printf("BEFORE: histo_temp[%d] = %d\n", transform_index, histo_temp[transform_index]);
        atomicAdd(&histo_temp[transform_index], 1);
        // printf("tile_height * stride = %d * %d = %d. ||  stride * y + x = %d * %d + %d = %d. ||  histo_temp[%d] = %d\n",
        //     tile_height, stride, tile_height * stride,
        //     stride, y, x, stride * y + x,
        //     transform_index, histo_temp[transform_index]);
    }
    __syncthreads();

    if (ind < 256) {
        atomicAdd(&histo[ind], histo_temp[ind]);

        //printf("value @ ind = %d: \t histo_temp: %d, histo: %d\n", ind, histo_temp[ind], histo[ind]);
    }
    __syncthreads();

    //printf("histo[0] = %d\n", histo[0]);

}


static void CollectColorBlueTransforms_CUDA(const uint32_t* argb, int stride,
                                            int tile_width, int tile_height,
                                            int green_to_blue, int red_to_blue,
                                            int histo[]) {

    // Dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((tile_width  + blockDim.x - 1) / blockDim.x,
                 (tile_height + blockDim.y - 1) / blockDim.y);

    size_t argb_size = tile_height * (stride - 1) + tile_width;

    // Allocate device memory buffer for result on the GPU
    uint32_t *argb_result;
    int *histo_result;
    //printf("tile_height * stride = %d * %d = %d\n", tile_height, stride, tile_height * stride);
    cudaCheckError(cudaMalloc(&histo_result, 256 * sizeof(int)));
    cudaCheckError(cudaMalloc(&argb_result, argb_size * sizeof(uint32_t)));

    // Copy input params to the GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(histo_result, histo, 256 * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(argb_result, argb, argb_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    CollectColorBlueTransforms_kernel<<<gridDim, blockDim>>>(argb_result, stride, tile_width, tile_height, green_to_blue, red_to_blue, histo_result);

    // Copy result from GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(histo, histo_result, 256 * sizeof(int), cudaMemcpyDeviceToHost)); //only histo modified

    cudaCheckError(cudaFree(argb_result));
    cudaCheckError(cudaFree(histo_result));
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
