#include <iostream>
#include <math.h>

// CUDA Kernel function to init elements of two arrays on the GPU
__global__
void init(int n, float *x, float *y) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
}

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}
}

int main(void)
{
	int N = 1<<20;

	float *x, *y;
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	// Init x and y arrays and run add kernel on the CPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	init<<<numBlocks, blockSize>>>(N, x, y);
	add<<<numBlocks, blockSize>>>(N, x, y);

	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	// Free memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}
