#ifndef __POOL_LAYER__
#define __POOL_LAYER__

#include "pool_layer_kernels.h"

#define EPSILON 0.000001

struct Pool_Layer {
  Tensor grads_in;
  Tensor in;
  Tensor out;

  int stride;
  int size_of_filter;
};
typedef struct Pool_Layer *Pool_Layer;

Pool_Layer init_pool_layer(int width, int height, int depth) {
  Pool_Layer pool_layer = malloc(sizeof(struct Pool_Layer));

  int stride = 2;
  int size_of_filter = 2;
  pool_layer->size_of_filter = size_of_filter;
  pool_layer->stride = stride;

  pool_layer->grads_in = initialize_tensor(width, height, depth);
  pool_layer->in = initialize_tensor(width, height, depth);
  pool_layer->out = initialize_tensor((width - size_of_filter) / stride + 1, (height - size_of_filter) / stride + 1, depth);
}

void activate_pooling(Pool_Layer layer, Tensor data) {
  layer->in = data;

  clock_t t;
  #ifdef GPU
    t = clock();
    activate_pooling_gpu(layer->in->data, layer->out->data);
    t = clock() - t;
    if (VERBOSE) printf("(gpu) pool: %f seconds\n", (double) t / CLOCKS_PER_SEC);
    return;
  #endif

  t = clock();
  for (int x = 0; x < layer->out->width; x += 1)
    for (int y = 0; y < layer->out->height; y += 1)
      for (int z = 0; z < layer->out->depth; z += 1) {
        float max = -FLT_MAX;
        int mapped_x = x * layer->stride;
        int mapped_y = y * layer->stride;

        for (int i = 0; i < layer->size_of_filter; i += 1)
          for (int j = 0; j < layer->size_of_filter; j += 1) {
            float v = layer->in->data[idx(layer->in, mapped_x + i, mapped_y + j, z)];
            if (v > max) max = v;
          }

        layer->out->data[idx(layer->out, x, y, z)] = max;
      }

  t = clock() - t;
  if (VERBOSE) printf("(cpu) pool: %f seconds\n", (double) t / CLOCKS_PER_SEC);
}

 int normalize_pool_range(float f, int max, int is_limit_min) {
   if (f <= 0)
     return 0;
   if (f >= (max - 1))
     return max - 1;
   if (is_limit_min) // left side of inequality
     return ceil(f);
   else
     return floor(f);
}

Range map_to_pool_output(Pool_Layer l, int x, int y) {
  Range r = malloc(sizeof(struct Range));
  float a = x;
  float b = y;
  r->min_x = normalize_pool_range((a - l->size_of_filter + 1) / l->stride, l->out->width, 1);
  r->min_y = normalize_pool_range((b - l->size_of_filter + 1) / l->stride, l->out->height, 1);
  r->min_z = 0;
  r->max_x = normalize_pool_range((a / l->stride), l->out->width, 0);
  r->max_y = normalize_pool_range((b / l->stride), l->out->height, 0);
  r->max_z = l->out->depth;
  return r;
}

void calc_pool_grads(Pool_Layer layer, Tensor grad_next_layer) {
  for (int x = 0; x < layer->in->width; x += 1) {
    for (int y = 0; y < layer->in->height; y += 1) {
      Range range = map_to_pool_output(layer, x, y);
      for (int z = 0; z < layer->in->depth; z += 1) {
        float sum_error = 0;
        for (int i = range->min_x; i <= range->max_x; i += 1)
          for (int j = range->min_y; j <= range->max_y; j += 1) {
            int is_max = 1;
            if (fabs(layer->in->data[idx(layer->in, x, y, z)] - layer->out->data[idx(layer->out, i, j, z)]) > EPSILON) {
              is_max = 0;
            }
            sum_error += is_max * grad_next_layer->data[idx(grad_next_layer, i, j, z)];
          }
        layer->grads_in->data[idx(layer->grads_in, x, y, z)] = sum_error;
      }
    }
  }
}

#endif
