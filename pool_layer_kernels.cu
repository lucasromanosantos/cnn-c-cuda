#include "stdio.h"
#include "float.h"
#include "cuda.h"

extern "C" {
  #include "pool_layer_kernels.h"
}

__global__ void pooling_kernel(float *in, float *out) {
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;

  int y = tx / 12;
  int x = tx % 12;

  int mapped_x = x * 2;
  int mapped_y = y * 2;

  float max = -FLT_MAX;
  for (int i = 0; i < 2; i += 1)
    for (int j = 0; j < 2; j += 1) {
      int idx = (bx * 24 * 24) + (24 * (mapped_y + j)) + (mapped_x + i);
      float v = in[idx];
      if (v > max) max = v;
    }

  int out_idx = (12 * 12 * bx) + (12 * y) + x;
  out[out_idx] = max;
}


void activate_pooling_gpu(float *in, float *out) {
  // 24 x 24 * 8 => 12 * 12 * 8
  float *device_in;
  float *device_out;

  cudaMalloc((void **) &device_in, 8 * 24 * 24 * sizeof(float));
  cudaMalloc((void **) &device_out, 8 * 12 * 12 * sizeof(float));

  cudaMemcpy(device_in, in, 24 * 24 * 8 * sizeof(float), cudaMemcpyHostToDevice);

  pooling_kernel<<<8, 12 * 12>>>(device_in, device_out);
  cudaDeviceSynchronize();

  cudaMemcpy(out, device_out, sizeof(float) * 8 * 12 * 12, cudaMemcpyDeviceToHost);
}
