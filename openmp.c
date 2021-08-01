#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define R 3
int n_threads;

float* single_layer_calc(float *x, float (*W)[R], float b, float *y, int N)
{
	int i,j;
	#pragma omp parallel for private(j) num_threads(n_threads)
	for (i=0; i<N-R+1; i++) {
		y[i] = 0.0;
		for (j=0; j<R; j++) {
			y[i] += x[i+j] * W[i][j];
		}
		y[i] = 1.0 / (exp(-(y[i]+b)) + 1);
	}

	return y;
}

int main(int argc, char *argv[])
{
	// The default value of N, K and number of threads to be used
	int N = 10;
	int K = 3;
	n_threads = omp_get_max_threads();

	// get N, K and number of threads from command line
	// the library used here is getopt (GNU) from unistd.h
	int c;
	while ((c = getopt (argc, argv, "t:n:k:")) != -1)
	{
		switch (c)
		{
		case 't': // number of threads
			n_threads = atoi(optarg);
			break;
		case 'n': // N
			N = atoi(optarg);
			break;
		case 'k': // K
			K = atoi(optarg);
			break;
		}
	}
	printf("Using %d threads, input size:%d, number of layers:%d.\n", n_threads, N, K);

	// Judge if the length of the k-th layer is bigger than 0
	if (N - (K-1) * (R-1) <= 0)
	{
		printf("The parameters you input couldn't support K layers. Please give a bigger N or a smaller K.\n");
		return 0;
	}

	// initialize the values of the input to 1
	float x[N];
	for (int i=0; i < N; i++)
	{
		x[i] = -1.0;
	}

	// Create an array for storing the activations of the latest layer got calculated
	// The size of the layers got smaller as the network goes deeper
	// I rewrite the N - t * (R-1) values from beginning
	float latest_layer[N];
	// copy the values from x, making x the initial values of layer 0
	memcpy(latest_layer, x, N * sizeof(float));

	// start recording time
	float t_start = omp_get_wtime();

	// Loop over K layers
	for(int t=1; t<K; t++)
	{
		// calculate length of this layer
		int layer_len = N - t * (R-1);

		// initialize parameters w and b
		//float b = rand() % 3 - 1; // random value from {-1, 0, 1}
		float b = 1.0;

		float W[layer_len][R];
		// random Initialization to values in range [-1,1]
		#pragma omp parallel for collapse(2) num_threads(n_threads)
		for (int i=0; i<layer_len; i++)
		{
			for (int j=0; j<R; j++)
			{
				//W[i][j] = ((rand() % 2000) - 1000) / 1000.0; // 3 effective digits after 0.
				W[i][j] = 1.0 / 3;
			}
		}

		// calculate the avtivations of this layer
		float y[layer_len]; // create a temp y
		int in_layer_len = layer_len + R - 1; // the length of the input layer
		single_layer_calc(latest_layer, W, b, y, in_layer_len);

		// save the temp result y to latest_layer
		memcpy(latest_layer, y, layer_len * sizeof(float));
	}

	// stop recording time
	float t_end = omp_get_wtime();

	// print the final result
	printf("Final result is: ");
	int last_layer_len = N - (K-1) * (R-1);
	for(int i=0; i<last_layer_len; i++)
	{
		printf("%.2f ", latest_layer[i]);
	}
	printf("\n");

	// print the time consumption
	printf("Elapsed time: %f\n", t_end - t_start);

	return 0;
}
