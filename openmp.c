#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define R 3
int n_threads;

/* The calculation of y values for one layer */
void one_layer_calc(float *x, float (*W)[R], float b, float *y, int N)
{
	int i,j;
	#pragma omp parallel for private(j) num_threads(n_threads)
	for (i=0; i<N-R+1; i++) {
		y[i] = 0.0;
		for (j=0; j<R; j++) {
			y[i] += x[i+j] * W[i][j];
		}
		y[i] = 1.0 / (exp(-(y[i]+b)) + 1); // +b, then sigmoid
	}
}

/* Initialize the W and b parameters for one layer */
void init_layer_parameters(float (*W)[R], float w_v, float *b, float b_v, int layer_len)
{
	#pragma omp parallel for collapse(2) num_threads(n_threads)
	for (int i=0; i<layer_len; i++)
	{
		for (int j=0; j<R; j++)
			{
				W[i][j] = w_v;
			}
	}
	
	*b = b_v;
}

/* Read in the network parameters (N, K and # threads) from command-line input.
   the library used here is getopt (GNU) from unistd.h. */
void parse_command_line_parameters(int argc, char *argv[], int *n_threads, int *N, int *K)
{
	int c;
	while ((c = getopt (argc, argv, "t:n:k:")) != -1)
	{
		switch (c)
		{
			case 't': // number of threads
				*n_threads = atoi(optarg);
				break;
			case 'n': // N
				*N = atoi(optarg);
				break;
			case 'k': // K
				*K = atoi(optarg);
				break;
		}
	}
}

int main(int argc, char *argv[])
{
	// The default value of N, K and number of threads to be used
	int N = 10;
	int K = 3;
	n_threads = omp_get_max_threads();

	// get N, K and number of threads from command line
	parse_command_line_parameters(argc, argv, &n_threads, &N, &K);
	printf("Using %d threads, input size:%d, number of layers:%d.\n", n_threads, N, K);

	// Judge if the length of the k-th layer is bigger than 0
	int last_layer_len = N - (K-1) * (R-1);
	if (last_layer_len <= 0)
	{
		printf("The parameters you input couldn't support K layers. Please give a bigger N or a smaller K.\n");
		return EXIT_FAILURE;
	}

	float x[N]; // the first layer
	float latest_layer[N]; // array for storing the latest layer got calculated
	
	for (int i=0; i < N; i++) // initialize x
	{
		x[i] = -1.0;
	}
	memcpy(latest_layer, x, N * sizeof(float)); // the lastest layer is the first layer at the beginning

	// start recording time
	float t_start = omp_get_wtime();

	// Loop over K layers
	for(int t=1; t<K; t++)
	{
		int layer_len = N - t * (R-1);        // calculate length of this layer
		int in_layer_len = layer_len + R - 1; // the length of the input layer

		// create w, b and y
		float W[layer_len][R];
		float b;
		float y[layer_len]; // layer result
		
		//float W_v = ((rand() % 2000) - 1000) / 1000.0; // random Initialization to values in range [-1,1]
		//float b_v = rand() % 3 - 1; // random value from {-1, 0, 1}
		float b_v = 1.0;
		float W_v = 1.0 / 3;
		init_layer_parameters(W, W_v, &b, b_v, layer_len);

		one_layer_calc(latest_layer, W, b, y, in_layer_len); // calculation on this layer, update y

		memcpy(latest_layer, y, layer_len * sizeof(float));  // update the values of latest_layer
	}

	// stop recording time
	float t_end = omp_get_wtime();

	// print the final result
	printf("Final result is: ");
	for(int i=0; i<last_layer_len; i++)
	{
		printf("%.2f ", latest_layer[i]);
	}
	printf("\n");

	// print the time consumption
	printf("Elapsed time: %f\n", t_end - t_start);

	return EXIT_SUCCESS;
}
