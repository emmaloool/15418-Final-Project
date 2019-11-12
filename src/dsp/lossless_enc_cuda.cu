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

static void SubtractGreenFromBlueAndRed_CUDA(uint32_t* argb_data,
                                             int num_pixels) {

  printf("HELLO WORLD WE ARE CALLING CUDA FUNCTION\n");
  int i = 0;
  // fallthrough and finish off with plain-C
  if (i != num_pixels) {
    VP8LSubtractGreenFromBlueAndRed_C(argb_data + i, num_pixels - i);
  }
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
