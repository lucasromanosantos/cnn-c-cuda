default all:
	nvcc main.c conv_layer_kernels.cu pool_layer_kernels.cu relu_layer_kernels.cu fc_layer_kernels.cu -o main -g -lm -DGPU
cpu:
	gcc main.c -o main -g -lm
clean:
	rm *.o
	rm main