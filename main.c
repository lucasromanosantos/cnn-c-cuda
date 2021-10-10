#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <unistd.h>
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#include "main.h"
#include "tensor.h"
#include "optimization.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "pool_layer.h"
#include "fc_layer.h"

#define MAX_TRAIN 5000

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

void load_data() {
  mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
  mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

Conv_Layer conv_1;
Relu_Layer relu_2;
Pool_Layer pool_3;
FC_Layer fc_4;

void initialize_cnn() {
  conv_1 = init_convolutional_layer(28); // 28 * 28 * 1 -> 24 * 24 * 8
  relu_2 = init_relu_layer(conv_1->out->width, conv_1->out->height, conv_1->out->depth);
  pool_3 = init_pool_layer(relu_2->out->width, relu_2->out->height, relu_2->out->depth);
  fc_4 = init_fc_layer(pool_3->out->width, pool_3->out->height, pool_3->out->depth); // 12 * 12 * 8 -> 10 * 1 * 1
}

void train() {
  float err_total = 0;
  for (int i = 0; i < train_cnt; i += 1) {
    if (i > MAX_TRAIN) break;

    float *data = malloc(sizeof(float) * 28 * 28);
    for (int x = 0; x < 28; ++x) {
      for (int y = 0; y < 28; ++y) {
        data[x * 28 + y] = train_set[i].data[x][y];
      }
    }

    Tensor expected = initialize_tensor(10, 1, 1);
    for (int b = 0; b < 10; b += 1)
      expected->data[idx(expected, b, 0, 0)] = train_set[i].label == b ? 1.0f : 0.0f;

    // 1. inference
    activate_convolutional(conv_1, data);
    activate_relu(relu_2, conv_1->out);
    activate_pooling(pool_3, relu_2->out);
    activate_fc(fc_4, pool_3->out);
    Tensor grads = subtract_tensor(fc_4->out, expected);

    // 2. gradient
    calc_fc_grads(fc_4, grads);
    calc_pool_grads(pool_3, fc_4->grads_in);
    calc_relu_grads(relu_2, pool_3->grads_in);
    calc_conv_grads(conv_1, relu_2->grads_in);

    // 3. fix weights
    fix_conv_weights(conv_1);
    fix_fc_weights(fc_4);

    // if (i % 1000 == 0) {
    float err = 0;
    for (int a = 0; a < grads->width * grads->height * grads->depth; a += 1) {
      float f = expected->data[a];
      if (f > 0.5)
        err += fabs(grads->data[a]);
    }
    err *= 100;
    err_total += err;

    printf("image %d err %f\n", i, err_total / (i + 1));
    // }
  }
}

void inference() {
  int correct = 0;
  for (int i = 0; i < test_cnt; i += 1) {
    float *data = malloc(sizeof(float) * 28 * 28);
    for (int x = 0; x < 28; ++x) {
      for (int y = 0; y < 28; ++y) {
        data[x * 28 + y] = test_set[i].data[x][y];
      }
    }

    Tensor expected = initialize_tensor(10, 1, 1);
    for (int b = 0; b < 10; b += 1)
      expected->data[idx(expected, b, 0, 0)] = test_set[i].label == b ? 1.0f : 0.0f;

    activate_convolutional(conv_1, data);
    activate_relu(relu_2, conv_1->out);
    activate_pooling(pool_3, relu_2->out);
    activate_fc(fc_4, pool_3->out);

    float max_output = -FLT_MAX;
    int index = -1;
    for (int b = 0; b < 10; b += 1) {
      if (fc_4->out->data[b] > max_output) {
        max_output = fc_4->out->data[b];
        index = b;
      }
    }
    if (index == test_set[i].label) correct += 1;
    printf("case %d => inference: %d, correct: %d\n", i, index, test_set[i].label);
  }
  printf("%d of %u correct. err = %f\n", correct, test_cnt, (float) correct / test_cnt);
}

void save_model() {
  save_tensor(conv_1->grads_in, "conv_1", "grads_in");
  save_tensor(conv_1->in, "conv_1", "in");
  save_tensor(conv_1->out, "conv_1", "out");

  for (int i = 0; i < conv_1->number_of_filters; i++) {
    char *name = NULL;
    asprintf(&name, "filters_%d", i);

    save_tensor(conv_1->filters[i], "conv_1", name);

    free(name);
  }
}

int main() {
  printf("1. loading data...\n");
  load_data();
  printf("2. initialize cnn...\n");
  initialize_cnn();
  printf("3. training...\n");
  train();
}
