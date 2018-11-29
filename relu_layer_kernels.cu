#include "stdio.h"
#include "cuda.h"

extern "C" {
  #include "relu_layer_kernels.h"
}

__global__ void relu_kernel(float *in, float *out) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  float v = in[col];
  if (v < 0) v = 0;
  out[col] = v;
}

void activate_relu_gpu(float *in, float *out) {
  // 24 x 24 * 8 => 24 * 24 * 8
  float *device_in;
  float *device_out;

  cudaMalloc((void **) &device_in, 8 * 24 * 24 * sizeof(float));
  cudaMalloc((void **) &device_out, 8 * 24 * 24 * sizeof(float));

  cudaMemcpy(device_in, in, 24 * 24 * 8 * sizeof(float), cudaMemcpyHostToDevice);

  relu_kernel<<<24, 24 * 8>>>(device_in, device_out);
  cudaDeviceSynchronize();

  cudaMemcpy(out, device_out, sizeof(float) * 8 * 24 * 24, cudaMemcpyDeviceToHost);
}
