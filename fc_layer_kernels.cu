#include "stdio.h"
#include "cuda.h"

extern "C" {
  #include "fc_layer_kernels.h"
}

__global__ void fc_kernel(float *in, float *weights, float *out) {
  const int tx = threadIdx.x;

  float inputv = 0;
  for (int i = 0; i < 12; i += 1)
    for (int j = 0; j < 12; j += 1)
      for (int z = 0; z < 8; z += 1) {
        int in_idx = (z * 12 * 12) + (j * 12) + i;
        int weight_idx = (12 * 12 * 8 * tx) + in_idx;
        inputv += in[in_idx] * weights[weight_idx];
      }

  float result = 1.0f / (1.0f + exp(-inputv)); // activator function
  out[tx] = result;
  __syncthreads();
}

void activate_fc_gpu(float *in, float *weights, float *out) {
  float *device_in;
  float *device_weights;
  float *device_out;

  cudaMalloc((void **) &device_in, 8 * 12 * 12 * sizeof(float));
  cudaMalloc((void **) &device_weights, 8 * 12 * 12 * 10 * sizeof(float));
  cudaMalloc((void **) &device_out, 10 * sizeof(float));

  cudaMemcpy(device_in, in, 8 * 12 * 12 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_weights, weights, 8 * 12 * 12 * 10 * sizeof(float), cudaMemcpyHostToDevice);

  fc_kernel<<<1, 10>>>(device_in, device_weights, device_out);
  cudaDeviceSynchronize();

  cudaMemcpy(out, device_out, sizeof(float) * 10, cudaMemcpyDeviceToHost);
}
