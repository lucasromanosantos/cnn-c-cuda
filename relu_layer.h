#include "tensor.h"

struct Relu_Layer {
  Tensor grads_in;
  Tensor in;
  Tensor out;
};
typedef struct Relu_Layer *Relu_Layer;

// int idx(Tensor t, int x, int y, int z) {
//   return (z * t->height * t->width) + (y * t->width) + x;
// }

Relu_Layer init_relu_layer(int width, int height, int depth) {
  Relu_Layer relu_layer = malloc(sizeof(struct Relu_Layer));

  relu_layer->grads_in = initialize_tensor(width, height, depth);
  relu_layer->in = initialize_tensor(width, height, depth);
  relu_layer->out = initialize_tensor(width, height, depth);

  return relu_layer;
}

void activate_relu(Relu_Layer layer, Tensor data) {
  layer->in = data;

  for (int i = 0; i < layer->in->width; i += 1)
    for (int j = 0; j < layer->in->height; j += 1)
      for (int z = 0; z < layer->in->depth; z += 1) {
        float v = layer->in->data[idx(layer->in, i, j, z)];
        if (v < 0) v = 0;
        layer->out->data[idx(layer->out, i, j, z)] = v;
      }
}

void calc_relu_grads(Relu_Layer layer, Tensor grad_next_layer) {
  for (int i = 0; i < layer->in->width; i += 1)
    for (int j = 0; j < layer->in->height; j += 1)
      for (int z = 0; z < layer->in->depth; z += 1) {
        float result = layer->in->data[idx(layer->in, i, j, z)] < 0
          ? 0 : (1 * grad_next_layer->data[idx(grad_next_layer, i, j, z)]);
        layer->grads_in->data[idx(layer->grads_in, i, j, z)] = result;
      }
}

void fix_relu_weights() {}
