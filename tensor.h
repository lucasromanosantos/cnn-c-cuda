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

char save_tensor(Tensor t, const char *layer_name, const char *tensor_name){
  int tensor_size = (3 * sizeof(int)) + (sizeof(float) * t->width * t->height * t->depth);

  char* result_buffer;
  result_buffer = malloc(tensor_size);

  memcpy(result_buffer, &t->width, sizeof(int));
  memcpy(result_buffer + sizeof(int), &t->height, sizeof(int));
  memcpy(result_buffer + 2 * sizeof(int), &t->depth, sizeof(int));
  memcpy(result_buffer + 3 * sizeof(int), t->data, sizeof(float) * t->width * t->height * t->depth );

  printf("sizeof tensor is %d bytes\n", tensor_size);

  char *f_name = NULL;
  asprinf(&f_name, "%s_%s.bin", layer_name, tensor_name);

  FILE * file= fopen(f_name, "w");

  fwrite(result_buffer, tensor_size, 1, file);
  fclose(file);
  free(f_name);
}

#endif
