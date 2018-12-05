# CNN implementation in C + CUDA :dog:
----
Implementation in progress of a Convolutional Neural Network in C and CUDA

Handwritten digit recognition using MNIST dataset
http://yann.lecun.com/exdb/mnist/

```
conv_layer(1, 5, 8, ...);   // 28 * 28 * 1 -> 24 * 24 * 8
relu_layer
pool_layer(2, 2, ...);      // 24 * 24 * 8 -> 12 * 12 * 8
fc_layer_t(..., 10);        // 4 * 4 * 16 -> 10
```

Currently, only inference is GPU supported

heavly inspired by:
https://github.com/can1357/simple_cnn
