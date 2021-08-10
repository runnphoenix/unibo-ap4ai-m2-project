/****************************************************************************
 * serial.c - a simple multi-layer Nerual Network
 *
 * Assignment of Module 2 of Ap4AI course of AI master degree @unibo
 *
 * Last updated in 2021 by Hanying Zhang <hanying.zhang@studio.unibo.it>
 * 
 * To the extent possible under law, the author(s) nave dedicated all
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * --------------------------------------------------------------------------
 *
 * compile with:
 * gcc -fopenmp serial.c -o serial
 * OR
 * gcc -fopenmp serial.c -o serial -lm
 * where -lm is used for linking the math library
 *
 * Run with:
 * ./serial -n # of nodes -k # of layers
 *
 ****************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define R 3

/* The calculation of y values for one layer */
void one_layer_calc(float *x, float (*W)[R], float b, float *y, int N)
{
	int i,j;
	for (i=0; i<N-R+1; i++) {
		y[i] = 0.0;
		for (j=0; j<R; j++) {
			y[i] += x[i+j] * W[i][j];
		}
		y[i] = 1.0 / (exp(-(y[i]+b)) + 1); // +b, then sigmoid
	}
}

/* Random values between -1 and 1 */
float random_init_small()
{
	return ((rand() % 2000) - 1000) / 1000.0;     // random Initialization to values in range [-1,1]
}

/* Initialize the W and b parameters for one layer */
void init_layer_parameters(float (*W)[R], float *b, int layer_len)
{
	for (int i=0; i<layer_len; i++) {
		for (int j=0; j<R; j++) {
			W[i][j] = random_init_small();
		}
	}
	
	*b = random_init_small();
}

/* Read in the network parameters (N, K and # threads) from command-line input. */
/* The library used here is getopt (GNU) from unistd.h. */
void parse_command_line_parameters(int argc, char *argv[], int *N, int *K)
{
	int c;
	while ((c = getopt (argc, argv, "t:n:k:")) != -1) {
		switch (c) {
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
	srand(42);
	
	// The default value of N, K
	int N = 10;
	int K = 5;

	// get N, K and number of threads from command line
	parse_command_line_parameters(argc, argv, &N, &K);
	printf("Input size:%d, number of layers:%d.\n", N, K);

	// Judge if the length of the k-th layer is bigger than 0
	int last_layer_len = N - (K-1) * (R-1);
	if (last_layer_len <= 0) {
		printf("The parameters you input couldn't support K layers. \
		        Please give a bigger N or a smaller K.\n");
		return EXIT_FAILURE;
	}

	float x[N];                   // the first layer
	float latest_layer[N];        // array for storing the latest layer got calculated
	
	for (int i=0; i < N; i++) {   // initialize x
		x[i] = random_init_small();
	}
	
	memcpy(latest_layer, x, N * sizeof(float)); // the lastest layer is the first layer at the beginning

	// Start recording time costage
    clock_t start = clock();

	// Loop over K layers
	for(int t=1; t<K; t++) {
		int layer_len = N - t * (R-1);          // calculate length of this layer
		int in_layer_len = layer_len + R - 1;   // the length of the input layer

		// create w, b and y
		float W[layer_len][R];
		float b;
		float y[layer_len]; // layer result
		
		init_layer_parameters(W, &b, layer_len);

		one_layer_calc(latest_layer, W, b, y, in_layer_len); // calculation on this layer, update y

		memcpy(latest_layer, y, layer_len * sizeof(float));  // update the values of latest_layer
	}

	// calculate elapsed time
    clock_t end = clock();
    double time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;

	// print the final result
	printf("Final result is: ");
	for(int i=0; i<last_layer_len; i++) {
		printf("%.2f ", latest_layer[i]);
	}
	printf("\n");

	// print the time consumption
	printf("Elapsed time: %f\n", time_elapsed);

	return EXIT_SUCCESS;
}
