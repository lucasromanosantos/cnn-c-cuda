#ifndef __FC_LAYER__
#define __FC_LAYER__

#define OUT_SIZE 10

struct FC_Layer {
  Tensor grads_in;
  Tensor in;
  Tensor out;
  Tensor weights;

  float *input;
  float *gradients;
  float *old_gradients;
};
typedef struct FC_Layer *FC_Layer;

FC_Layer init_fc_layer(int width, int height, int depth) {
  FC_Layer fc_layer = malloc(sizeof(struct FC_Layer));

  fc_layer->grads_in = initialize_tensor(width, height, depth);
  fc_layer->in = initialize_tensor(width, height, depth);
  fc_layer->out = initialize_tensor(OUT_SIZE, 1, 1);
  fc_layer->input = malloc(sizeof(float) * OUT_SIZE);
  fc_layer->weights = initialize_tensor(width * height * depth, OUT_SIZE, 1);

  int maxval = width * height * depth;
  for (int i = 0; i < OUT_SIZE; i += 1)
    for (int h = 0; h < width * height * depth; h += 1) {
      fc_layer->weights->data[idx(fc_layer->weights, h, i, 0)] = 2.19722f / (float) maxval * rand() / (float) RAND_MAX;
    }

  fc_layer->gradients = malloc(sizeof(float) * OUT_SIZE);
  fc_layer->old_gradients = malloc(sizeof(float) * OUT_SIZE);
  for (int i = 0; i < 10; i += 1) {
    fc_layer->gradients[i] = 0;
    fc_layer->old_gradients[i] = 0;
  }
  return fc_layer;
}

float activator_function(float x) {
  float sig = 1.0f / (1.0f + exp(-x));
  return sig;
}

void activate_fc(FC_Layer layer, Tensor data) {
  layer->in = data;
  for (int n = 0; n < layer->out->width; n += 1) { // 10
    float inputv = 0;
    for (int i = 0; i < layer->in->width; i += 1)
      for (int j = 0; j < layer->in->height; j += 1)
        for (int z = 0; z < layer->in->depth; z += 1) {
          int m = idx(layer->in, i, j, z);
          inputv += layer->in->data[m] * layer->weights->data[idx(layer->weights, m, n, 0)];
        }
    layer->input[n] = inputv; // ?
    layer->out->data[idx(layer->out, n, 0, 0)] = activator_function(inputv);
  }
}

float activator_derivative(float x) {
  float sig = 1.0f / (1.0f + exp( -x ));
  return sig * (1 - sig);
}

void calc_fc_grads(FC_Layer layer, Tensor grad_next_layer) {
  for (int i = 0; i < layer->in->width * layer->in->height * layer->in->depth; i += 1) {
    layer->grads_in->data[i] = 0;
  }

  for (int n = 0; n < layer->out->width; n += 1) {
    float *grad = &layer->gradients[n];
    layer->gradients[n] = grad_next_layer->data[idx(grad_next_layer, n, 0, 0)] * activator_derivative(layer->input[n]);

    for (int i = 0; i < layer->in->width; i += 1)
      for (int j = 0; j < layer->in->height; j += 1)
        for (int z = 0; z < layer->in->depth; z += 1) {
          int m = idx(layer->in, i, j, z);
          layer->grads_in->data[idx(layer->grads_in, i, j, z)] +=
            *grad * layer->weights->data[idx(layer->weights, m, n, 0)];
        }
  }
}

void fix_fc_weights(FC_Layer layer) {
  for (int n = 0; n < layer->out->width; n += 1) {
    float grad = layer->gradients[n];
    float *old_grad = &layer->old_gradients[n];
    for (int i = 0; i < layer->in->width; i += 1)
      for (int j = 0; j < layer->in->height; j += 1)
        for (int z = 0; z < layer->in->depth; z += 1) {
          int m = idx(layer->in, i, j, z);
          float w = layer->weights->data[idx(layer->weights, m, n, 0)];
          layer->weights->data[idx(layer->weights, m, n, 0)] = update_weight(w, grad, *old_grad, layer->in->data[m]);
        }
    update_gradient(grad, old_grad);
  }
}

#endif
