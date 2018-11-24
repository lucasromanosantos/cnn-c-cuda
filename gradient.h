#ifndef __GRADIENT__
#define __GRADIENT__

struct Gradient {
  float grad;
  float old_grad;
};
typedef struct Gradient *Gradient;

Gradient initialize_gradient() {
  Gradient g = malloc(sizeof(struct Gradient));
  g->grad  = 0;
  g->old_grad = 0;
  return g;
}

#endif
