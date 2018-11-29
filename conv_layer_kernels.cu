#include "stdio.h"
#include "cuda.h"

extern "C" {
  #include "conv_layer_kernels.h"
}

__global__ void convolutional_kernel(float *in, float *weights, float *out) {
  const int bx = blockIdx.x;

  int y = threadIdx.x / 24;
  int x = threadIdx.x % 24;

  float sum = 0;
  for (int i = 0; i < 5; i += 1)
    for (int j = 0; j < 5; j += 1) {
      int weight_idx = (bx * 5 * 5) + (5 * j) + i;
      int input_idx = (28 * (y + j)) + (x + i);
      float v = in[input_idx];
      float f = weights[weight_idx];
      sum += f * v;
    }

  int out_idx = (24 * 24 * bx) + (24 * y) + x;
  out[out_idx] = sum;
}

void activate_convolutional_gpu(float *in, float *weights, float *out) {
  // 28 x 28 => 8 x 24 x 24 (size of filter = 5)
  float *device_in;
  float *device_filters;
  float *device_out;

  cudaMalloc((void **) &device_in, 1 * 28 * 28 * sizeof(float));
  cudaMalloc((void **) &device_filters, 8 * 5 * 5 * sizeof(float));
  cudaMalloc((void **) &device_out, 8 * 24 * 24 * sizeof(float));

  cudaMemcpy(device_in, in, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_filters, weights, 8 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);

  convolutional_kernel<<<8, 24*24>>>(device_in, device_filters, device_out);
  cudaDeviceSynchronize();

  cudaMemcpy(out, device_out, sizeof(float) * 8 * 24 * 24, cudaMemcpyDeviceToHost);
}
