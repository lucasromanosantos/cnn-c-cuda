#ifndef __OPTIMIZATION__
#define __OPTIMIZATION__

#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

float update_weight(float w, float grad, float old_grad, float multp) {
  // printf("atualizando ... w = %f e grad  %f e old_grad %f\n", w, grad, old_grad);
  float m = (grad + old_grad * MOMENTUM);
  float w2 = w - ((LEARNING_RATE * m * multp) + (LEARNING_RATE * WEIGHT_DECAY * w));
  return w2;
}

float update_gradient(float grad, float *old_grad) {
  return (grad + *old_grad * MOMENTUM);
}

#endif
