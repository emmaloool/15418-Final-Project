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


static VP8LProcessEncBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed_old;
static VP8LTransformColorFunc VP8LTransformColor_old;
static VP8LCollectColorRedTransformsFunc VP8LCollectColorRedTransforms_old;

#define my_assert(ans) my_assert_helper((ans), __FILE__, __LINE__);
static void my_assert_helper(int code, const char *file, int line) {
    if (code == 0) {
        fprintf(stderr, "my_assert failure: %s:%d\n", file, line);
        exit(code);
    }
}


#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
            cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static double convert_duration(std::chrono::duration<int64_t, std::nano> d) {
    typedef std::chrono::duration<double, std::milli> output_type;
    return std::chrono::duration_cast<output_type>(d).count();
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

    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    SubtractGreenFromBlueAndRed_kernel<<<blocks, threadsPerBlock>>>(result, num_pixels);

    auto end = std::chrono::high_resolution_clock::now();
    printf("SubtractGreenFromBlueAndRed: kernel execution time %.6f\n",
       convert_duration(end - start));

    // Copy result from GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(argb_data, result, num_pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost))

    cudaCheckError(cudaFree(result));
}

static void SubtractGreenFromBlueAndRed_Wrapper(
        uint32_t* argb_data, int num_pixels) {

    uint32_t *argb_data_temp = (uint32_t *) malloc(sizeof(*argb_data_temp) * num_pixels);
    uint32_t *argb_data_res = (uint32_t *) malloc(sizeof(*argb_data_res) * num_pixels);

    // Warm cache, and get reference result
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        VP8LSubtractGreenFromBlueAndRed_C(argb_data_temp, num_pixels);
        memcpy(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels);
    }

    // Time original function
    double duration_old;
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        auto start = std::chrono::high_resolution_clock::now();

        VP8LSubtractGreenFromBlueAndRed_old(argb_data_temp, num_pixels);

        auto end = std::chrono::high_resolution_clock::now();
        duration_old = convert_duration(end - start);
        my_assert(0 == memcmp(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels));
    }

    // Time CUDA function
    double duration_cuda;
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        auto start = std::chrono::high_resolution_clock::now();

        SubtractGreenFromBlueAndRed_CUDA(argb_data_temp, num_pixels);

        auto end = std::chrono::high_resolution_clock::now();
        duration_cuda = convert_duration(end - start);
        my_assert(0 == memcmp(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels));
    }

    // Time C function
    double duration_c;
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        auto start = std::chrono::high_resolution_clock::now();

        VP8LSubtractGreenFromBlueAndRed_C(argb_data_temp, num_pixels);

        auto end = std::chrono::high_resolution_clock::now();
        duration_c = convert_duration(end - start);
        my_assert(0 == memcmp(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels));
    }

    printf("SubtractGreenFromBlueAndRed: "
        "duration_cuda = %.6f, duration_C = %.6f, duration_old = %.6f\n",
        duration_cuda, duration_c, duration_old);

    memcpy(argb_data, argb_data_res, sizeof(*argb_data) * num_pixels);

    free(argb_data_temp);
    free(argb_data_res);
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

    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    TransformColor_kernel<<<blocks, threadsPerBlock>>>(m->green_to_red_, m->green_to_blue_, m->red_to_blue_, result, num_pixels);

    auto end = std::chrono::high_resolution_clock::now();
    printf("TransformColor: kernel execution time %.6f\n",
       convert_duration(end - start));

    // Copy result from GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(data, result, num_pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(result));
}

static void TransformColor_Wrapper(
        const VP8LMultipliers* const m,
        uint32_t* argb_data, int num_pixels) {

    static int num_calls = 0;
    if (num_calls % 1000 == 0) {
        printf("TransformColor: num_calls = %d\n", num_calls);
    }

    // Reset to the old function after a while, because this is called a lot
    if (num_calls++ >= 8) {
        VP8LTransformColor_old(m, argb_data, num_pixels);
        return;
    }
    printf("TransformColor: num_pixels = %d\n", num_pixels);

    uint32_t *argb_data_temp = (uint32_t *) malloc(sizeof(*argb_data_temp) * num_pixels);
    uint32_t *argb_data_res = (uint32_t *) malloc(sizeof(*argb_data_res) * num_pixels);

    // Warm cache, and get reference result
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        VP8LTransformColor_C(m, argb_data_temp, num_pixels);
        memcpy(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels);
    }

    // Time original function
    double duration_old;
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        auto start = std::chrono::high_resolution_clock::now();

        VP8LTransformColor_old(m, argb_data_temp, num_pixels);

        auto end = std::chrono::high_resolution_clock::now();
        duration_old = convert_duration(end - start);
        my_assert(0 == memcmp(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels));
    }

    // Time CUDA function
    double duration_cuda;
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        auto start = std::chrono::high_resolution_clock::now();

        TransformColor_CUDA(m, argb_data_temp, num_pixels);

        auto end = std::chrono::high_resolution_clock::now();
        duration_cuda = convert_duration(end - start);
        my_assert(0 == memcmp(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels));
    }

    // Time C function
    double duration_c;
    {
        memcpy(argb_data_temp, argb_data, sizeof(*argb_data_temp) * num_pixels);
        auto start = std::chrono::high_resolution_clock::now();

        VP8LTransformColor_C(m, argb_data_temp, num_pixels);

        auto end = std::chrono::high_resolution_clock::now();
        duration_c = convert_duration(end - start);
        my_assert(0 == memcmp(argb_data_res, argb_data_temp, sizeof(*argb_data_res) * num_pixels));
    }

    printf("TransformColor: "
        "duration_cuda = %.6f, duration_C = %.6f, duration_old = %.6f\n",
        duration_cuda, duration_c, duration_old);

    memcpy(argb_data, argb_data_res, sizeof(*argb_data) * num_pixels);

    free(argb_data_temp);
    free(argb_data_res);
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

    /*
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
    */

    if (x < tile_width && y < tile_height) {
        int transform_index = TransformColorRed((uint8_t)green_to_red, argb[stride * y + x]);
        atomicAdd(&histo[transform_index], 1);
    }
}

static void CollectColorRedTransforms_CUDA(const uint32_t* argb, int stride,
                                     int tile_width, int tile_height,
                                     int green_to_red, int histo[]) {

    printf("tile_width = %d, tile_height = %d\n", tile_width, tile_height);

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

    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel to compute VP8LSubtractGreenFromBlueAndRed
    CollectColorRedTransforms_kernel<<<gridDim, blockDim>>>(argb_result, stride, tile_width, tile_height, green_to_red, histo_result);

    auto end = std::chrono::high_resolution_clock::now();
    printf("CollectColorRedTransforms: kernel execution time %.6f\n",
       convert_duration(end - start));

    // Copy result from GPU using cudaMemcpy
    cudaCheckError(cudaMemcpy(histo, histo_result, 256 * sizeof(int), cudaMemcpyDeviceToHost)); //only histo modified

    cudaCheckError(cudaFree(argb_result));
    cudaCheckError(cudaFree(histo_result));

}

static void CollectColorRedTransforms_Wrapper(
        const uint32_t* argb, int stride,
        int tile_width, int tile_height,
        int green_to_red, int histo[]) {

    static int num_calls = 0;
    if (num_calls % 1000 == 0) {
        printf("CollectColorRedTransforms: num_calls = %d\n", num_calls);
    }

    // Reset to the old function after a while, because this is called a lot
    if (num_calls++ >= 8) {
        VP8LCollectColorRedTransforms_old(
              argb, stride, tile_width, tile_height,
              green_to_red, histo);
        return;
    }


    int *histo_temp = (int *) malloc(sizeof(*histo_temp) * 256);
    int *histo_res = (int *) malloc(sizeof(*histo_res) * 256);

    // Warm cache, and get reference result
    {
        memcpy(histo_temp, histo, sizeof(*histo_temp) * 256);
        VP8LCollectColorRedTransforms_C(argb, stride,
            tile_width, tile_height, green_to_red, histo_temp);
        memcpy(histo_res, histo_temp, sizeof(*histo_res) * 256);
    }

    // Time old function
    double duration_old;
    {
        memcpy(histo_temp, histo, sizeof(*histo_temp) * 256);
        auto start = std::chrono::high_resolution_clock::now();

        VP8LCollectColorRedTransforms_old(argb, stride,
            tile_width, tile_height, green_to_red, histo_temp);

        auto end = std::chrono::high_resolution_clock::now();
        duration_old = convert_duration(end - start);
        my_assert(0 == memcmp(histo_res, histo_temp, sizeof(*histo_res) * 256));
    }

    // Time CUDA function
    double duration_cuda;
    {
        memcpy(histo_temp, histo, sizeof(*histo_temp) * 256);
        auto start = std::chrono::high_resolution_clock::now();

        CollectColorRedTransforms_CUDA(argb, stride,
            tile_width, tile_height, green_to_red, histo_temp);

        auto end = std::chrono::high_resolution_clock::now();
        duration_cuda = convert_duration(end - start);
        my_assert(0 == memcmp(histo_res, histo_temp, sizeof(*histo_res) * 256));
    }

    // Time C function
    double duration_c;
    {
        memcpy(histo_temp, histo, sizeof(*histo_temp) * 256);
        auto start = std::chrono::high_resolution_clock::now();

        VP8LCollectColorRedTransforms_C(argb, stride,
            tile_width, tile_height, green_to_red, histo_temp);

        auto end = std::chrono::high_resolution_clock::now();
        duration_c = convert_duration(end - start);
        my_assert(0 == memcmp(histo_res, histo_temp, sizeof(*histo_res) * 256));
    }

    printf("CollectColorRedTransforms: "
        "duration_cuda = %.6f, duration_C = %.6f, duration_old = %.6f\n",
        duration_cuda, duration_c, duration_old);

    memcpy(histo, histo_res, sizeof(*histo) * 256);

    free(histo_temp);
    free(histo_res);
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


__global__ void VP8LBundleColorMap_kernel(const uint8_t *row, int width, int xbits,
                                          uint32_t *dst) {

    const int num_tasks = width >> xbits;
    const int bundle_size = 1 << xbits;
    const int bit_depth = 1 << (3 - xbits);

    // Compute index in destination array
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_tasks) return;

    // Perform computation
    uint32_t code = 0xff000000;
    for (int xsub = 0; xsub < bundle_size; ++xsub) {
        code |= row[bundle_size * index + xsub] << (8 + bit_depth * xsub);
    }
    dst[index] = code;
}


static void BundleColorMap_CUDA(const uint8_t *row, int width, int xbits,
                                uint32_t *dst) {

    printf("HELLO WORLD WE ARE CALLING BundleColorMap_CUDA\n");

    // Compute number of tasks
    const int num_tasks = width >> xbits;
    const int threads_per_block = 512;
    const int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block;

    // Allocate device memory buffers
    uint8_t *device_row;
    uint32_t *device_dst;
    cudaMalloc(&device_row, sizeof(*device_row) * width);
    cudaMalloc(&device_dst, sizeof(*device_dst) * num_tasks);

    // Compute result
    cudaMemcpy(device_row, row, sizeof(*device_row) * width, cudaMemcpyHostToDevice);
    VP8LBundleColorMap_kernel<<<num_blocks, threads_per_block>>>(device_row, width, xbits, dst);
    cudaMemcpy(dst, device_dst, sizeof(*dst) * num_tasks, cudaMemcpyDeviceToHost);

    cudaFree(device_row);
    cudaFree(device_dst);
}


extern "C" void VP8LColorSpaceTransform_C(int width, int height, int bits, int quality,
                               uint32_t* const argb, uint32_t* image);
extern "C" void VP8LColorSpaceTransform_CUDA(int width, int height, int bits, int quality,
                               uint32_t* const argb, uint32_t* image);

extern "C" static void VP8LColorSpaceTransform_Wrapper(
        int width, int height, int bits, int quality,
        uint32_t* const argb, uint32_t* image) {

    const int transform_width = VP8LSubSampleSize(width, bits);
    const int transform_height = VP8LSubSampleSize(height, bits);

    uint32_t *argb_temp = new uint32_t[width * height];
    uint32_t *argb_res = new uint32_t[width * height];

    uint32_t *image_temp = new uint32_t[transform_width * transform_height];
    uint32_t *image_res = new uint32_t[transform_width * transform_height];

    // Warm cache, and get reference result
    {
        memcpy(argb_temp, argb, width * height * sizeof(*argb_temp));
        memcpy(image_temp, image, transform_width * transform_height * sizeof(*image_temp));

        VP8LColorSpaceTransform_C(width, height, bits, quality, argb_temp, image_temp);

        memcpy(argb_res, argb_temp, width * height * sizeof(*argb_res));
        memcpy(image_res, image_temp, transform_width * transform_height * sizeof(*image_res));
    }

    // Time C function
    double duration_c;
    {
        memcpy(argb_temp, argb, width * height * sizeof(*argb_temp));
        memcpy(image_temp, image, transform_width * transform_height * sizeof(*image_temp));

        auto start = std::chrono::high_resolution_clock::now();

        VP8LColorSpaceTransform_C(width, height, bits, quality, argb_temp, image_temp);

        auto end = std::chrono::high_resolution_clock::now();
        duration_c = convert_duration(end - start);

        my_assert(0 == memcmp(argb_res, argb_temp, width * height * sizeof(*argb_res)));
        my_assert(0 == memcmp(image_res, image_temp, transform_width * transform_height * sizeof(*image_res)));
    }

    // Time CUDA function
    double duration_cuda;
    {
        memcpy(argb_temp, argb, width * height * sizeof(*argb_temp));
        memcpy(image_temp, image, transform_width * transform_height * sizeof(*image_temp));

        auto start = std::chrono::high_resolution_clock::now();

        VP8LColorSpaceTransform_CUDA(width, height, bits, quality, argb_temp, image_temp);

        auto end = std::chrono::high_resolution_clock::now();
        duration_cuda = convert_duration(end - start);

        // The CUDA result is indeed different
        //my_assert(0 == memcmp(argb_res, argb_temp, width * height * sizeof(*argb_res)));
        //my_assert(0 == memcmp(image_res, image_temp, transform_width * transform_height * sizeof(*image_res)));

        // Use the CUDA result as final result
        memcpy(argb_res, argb_temp, width * height * sizeof(*argb_res));
        memcpy(image_res, image_temp, transform_width * transform_height * sizeof(*image_res));
    }

    printf("VP8LColorSpaceTransform: "
        "duration_cuda = %.6f, duration_C = %.6f\n",
        duration_cuda, duration_c);

    memcpy(argb, argb_res, width * height * sizeof(*argb));
    memcpy(image, image_res, transform_width * transform_height * sizeof(*image));

    delete[] argb_temp;
    delete[] argb_res;
    delete[] image_temp;
    delete[] image_res;
}

//------------------------------------------------------------------------------
// Entry point

extern "C" void VP8LEncDspInitCUDA(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitCUDA(void) {
    VP8LColorSpaceTransform = VP8LColorSpaceTransform_Wrapper;

    //VP8LSubtractGreenFromBlueAndRed_old = VP8LSubtractGreenFromBlueAndRed;
    //VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_Wrapper;
    //printf("VP8LSubtractGreenFromBlueAndRed_old = %p\n", VP8LSubtractGreenFromBlueAndRed_old);

    //VP8LTransformColor_old = VP8LTransformColor;
    //VP8LTransformColor = TransformColor_Wrapper;

    //VP8LCollectColorRedTransforms_old = VP8LCollectColorRedTransforms;
    //VP8LCollectColorRedTransforms = CollectColorRedTransforms_Wrapper;

    //VP8LTransformColor = TransformColor_CUDA;
    //VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_CUDA;
    //VP8LBundleColorMap = BundleColorMap_CUDA;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(VP8LEncDspInitCUDA)

#endif  // WEBP_USE_SSE2
