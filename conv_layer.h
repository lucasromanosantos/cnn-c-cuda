#ifndef __CONV_LAYER__
#define __CONV_LAYER__

#define DEPTH 1

#include "conv_layer_kernels.h"

struct Conv_Layer {
  Tensor grads_in;
  Tensor in;
  Tensor out;

  Tensor *filters;
  Tensor *filter_grads;
  Tensor *filter_old_grads;

  int stride;
  int number_of_filters;
  int size_of_filter;
};
typedef struct Conv_Layer *Conv_Layer;

struct Range {
  int min_x, min_y, min_z;
  int max_x, max_y, max_z;
};
typedef struct Range *Range;

Conv_Layer init_convolutional_layer(int in_size) {
  Conv_Layer conv_layer = malloc(sizeof(struct Conv_Layer));

  int size_of_filter = 5;
  int number_of_filters = 8;
  conv_layer->number_of_filters = number_of_filters;
  conv_layer->size_of_filter = size_of_filter;
  conv_layer->stride = 1;

  conv_layer->grads_in = initialize_tensor(in_size, in_size, DEPTH);
  conv_layer->in = initialize_tensor(in_size, in_size, DEPTH);

  int out_size = (in_size - size_of_filter) / conv_layer->stride + 1;
  conv_layer->out = initialize_tensor(out_size, out_size, number_of_filters);

  conv_layer->filters = malloc(sizeof(struct Tensor) * number_of_filters);
  for (int c = 0; c < number_of_filters; c += 1) {
    Tensor t = initialize_tensor(size_of_filter, size_of_filter, DEPTH);

    int maxval = size_of_filter * size_of_filter * DEPTH;
    for (int i = 0; i < size_of_filter; i += 1)
      for (int j = 0; j < size_of_filter; j += 1)
        for (int k = 0; k < DEPTH; k += 1) {
          t->data[idx(t, i, j, k)] = 1.0f / maxval * rand() / (float) RAND_MAX;
        }
      conv_layer->filters[c] = t;
  }

  conv_layer->filter_grads = malloc(sizeof(Tensor) * number_of_filters);
  conv_layer->filter_old_grads = malloc(sizeof(Tensor) * number_of_filters);
  for (int c = 0; c < number_of_filters; c += 1) {
    Tensor t = initialize_tensor(size_of_filter, size_of_filter, 1);
    Tensor t2 = initialize_tensor(size_of_filter, size_of_filter, 1);

    for (int a = 0; a < size_of_filter * size_of_filter * 1; a += 1) {
      t->data[a] = 0;
      t2->data[a] = 0;
    }

    conv_layer->filter_grads[c] = t;
    conv_layer->filter_old_grads[c] = t2;
  }

  return conv_layer;
}

void activate_convolutional(Conv_Layer layer, float *data) {
  layer->in->data = data;

  clock_t t;
  #ifdef GPU
    float* mapped_weights = malloc(sizeof(float) * 8 * 5 * 5);
    for (int i = 0; i < layer->number_of_filters; i += 1) {
      int offset = 5 * 5 * i;
      memcpy(mapped_weights + offset, layer->filters[i]->data, 5 * 5 * sizeof(float));
    }
    t = clock();
    activate_convolutional_gpu(layer->in->data, mapped_weights, layer->out->data);
    t = clock() - t;
    if (VERBOSE) printf("(gpu) conv: %f seconds\n", (double) t / CLOCKS_PER_SEC);
    return;
  #endif

  t = clock();
  for (int filter = 0; filter < layer->number_of_filters; filter += 1) {
    Tensor filter_data = layer->filters[filter];
    for (int x = 0; x < layer->out->width; x += 1)
      for (int y = 0; y < layer->out->height; y += 1) {
        int mapped_x = x * layer->stride;
        int mapped_y = y * layer->stride;
        float sum = 0;
        for (int i = 0; i < layer->size_of_filter; i += 1)
          for (int j = 0; j < layer->size_of_filter; j += 1)
            for (int z = 0; z < DEPTH; z += 1) {
              float f = filter_data->data[idx(filter_data, i, j, z)];
              int in_index = idx(layer->in, mapped_x + i, mapped_y + j, z);
              float v = layer->in->data[in_index];
              sum += f * v;
            }

        layer->out->data[idx(layer->out, x, y, filter)] = sum;
      }
  }
  t = clock() - t;
  if (VERBOSE) printf("(cpu) conv: %f seconds\n", (double) t / CLOCKS_PER_SEC);
}


int normalize_range(float f, int max, int is_limit_min) {
  if (f <= 0)
    return 0;
  if (f >= (max - 1))
    return max - 1;
  if (is_limit_min) // left side of inequality
    return ceil(f);
  else
    return floor(f);
}

Range map_to_output(Conv_Layer l, int x, int y) {
  Range r = malloc(sizeof(struct Range));
  float a = x;
  float b = y;

  r->min_x = normalize_range((a - l->size_of_filter + 1) / l->stride, l->out->width, 1);
  r->min_y = normalize_range((b - l->size_of_filter + 1) / l->stride, l->out->height, 1);
  r->min_z = 0;
  r->max_x = normalize_range((a / l->stride), l->out->width, 0);
  r->max_y = normalize_range((b / l->stride), l->out->height, 0);
  r->max_z = l->number_of_filters - 1;
  return r;
}

void calc_conv_grads(Conv_Layer layer, Tensor grad_next_layer) {
  for (int k = 0; k < layer->number_of_filters; k += 1)
    for (int i = 0; i < layer->size_of_filter; i += 1)
      for (int j = 0; j < layer->size_of_filter; j += 1)
        for (int z = 0; z < layer->in->depth; z += 1)
          layer->filter_grads[k]->data[idx(layer->filter_grads[k], i, j, z)] = 0;

  for (int x = 0; x < layer->in->width; x += 1)
    for (int y = 0; y < layer->in->height; y += 1) {
      Range r = map_to_output(layer, x, y);
      for (int z = 0; z < layer->in->depth; z += 1) {
        float sum_error = 0;

        for (int i = r->min_x; i <= r->max_x; i += 1) {
          int minx = i * layer->stride;
          for (int j = r->min_y; j <= r->max_y; j += 1) {
            int miny = j * layer->stride;
            for (int k = r->min_z; k <= r->max_z; k += 1) {
              int w_applied = layer->filters[k]->data[idx(layer->filters[k], x - minx, y - miny, z)]; // !!!!!!! int?
              sum_error += w_applied * grad_next_layer->data[idx(grad_next_layer, i, j, k)];
              float a = layer->in->data[idx(layer->in, x, y, z)];
              float b = grad_next_layer->data[idx(grad_next_layer, i, j, k)];
              layer->filter_grads[k]->data[idx(layer->filter_grads[k], x - minx, y - miny, z)] += a * b;
            }
          }
        }
        layer->grads_in->data[idx(layer->grads_in, x, y, z)] = sum_error;
      }
    }

}

void fix_conv_weights(Conv_Layer layer) {
  for (int a = 0; a < layer->number_of_filters; a += 1)
    for (int i = 0; i < layer->size_of_filter; i += 1)
      for (int j = 0; j < layer->size_of_filter; j += 1)
        for (int z = 0; z < 1; z += 1) {
          float *w = &layer->filters[a]->data[idx(layer->filters[a], i, j, z)];
          float *grad = &layer->filter_grads[a]->data[idx(layer->filter_grads[a], i, j, z)];
          float *old_grad = &layer->filter_old_grads[a]->data[idx(layer->filter_old_grads[a], i, j, z)];
          *w = update_weight(*w, *grad, *old_grad, 1);
          *old_grad = update_gradient(*grad, old_grad);
        }
}

#endif
