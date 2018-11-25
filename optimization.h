#ifndef __OPTIMIZATION__
#define __OPTIMIZATION__

#include "gradient.h"

#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

float update_weight(float w, float grad, float old_grad, float multp) {
  float m = (grad + old_grad * MOMENTUM);
  float w2 = w - ((LEARNING_RATE * m * multp) + (LEARNING_RATE * WEIGHT_DECAY * w));
  return w2;
}

void update_gradient(float grad, float *old_grad) {
  *old_grad = (grad + *old_grad * MOMENTUM);
}

#endif
