#ifndef __TENSOR__
#define __TENSOR__

struct Tensor {
  int width;
  int height;
  int depth;

  float *data;
};
typedef struct Tensor *Tensor;

int idx(Tensor t, int x, int y, int z) {
  return (z * t->height * t->width) + (y * t->width) + x;
}

Tensor initialize_tensor(float width, float height, float depth) {
  Tensor t = malloc(sizeof(struct Tensor));
  t->width = width;
  t->height = height;
  t->depth = depth;
  t->data = malloc(sizeof(float) * width * height * depth);
  return t;
}

static void print_tensor(Tensor t) {
	int mx = t->width;
	int my = t->height;
	int mz = t->depth;

	for (int z = 0; z < mz; z++ ) {
		printf( "[Dim%d]\n", z);
		for (int y = 0; y < my; y++ ) {
			for (int x = 0; x < mx; x++ )
				printf( "%.4f ", t->data[idx(t, x, y, z)]);
			printf( "\n" );
		}
	}
}

Tensor subtract_tensor(Tensor a, Tensor b) {
  Tensor t = initialize_tensor(a->width, a->height, a->depth);
  for (int i = 0; i < a->width * a->height * a->depth; i += 1) {
    t->data[i] = a->data[i] - b->data[i];
  }
  return t;
}

#endif
