/****************************************************************************
 * openmp.c - a simple multi-layer Nerual Network
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
 * gcc -fopenmp openmp.c -o openmp
 * OR
 * gcc -fopenmp openmp.c -o openmp -lm
 * where -lm is used for linking the math library
 *
 * Run with:
 * ./openmp -t # of threads -n # of nodes -k # of layers
 *
 ****************************************************************************/

#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define R 3
int n_threads;

/* The calculation of y values for one layer */
void one_layer_calc(float *x, float *W, float *b, float *y, int N)
{
	int i,j;
	#pragma omp parallel for private(j) num_threads(n_threads)
	for (i=0; i<N-R+1; i++) {
		*(y+i) = 0.0;
		for (j=0; j<R; j++) {
			*(y+i) += *(x+i+j) * *(W+i*R+j);
		}
		// Sigmoid
		*(y+i) = 1.0 / (exp(-*(y+i) - *b) + 1); // +b, then sigmoid
	}
}

/* Random values between -1 and 1 */
float random_init_small()
{
	return ((rand() % 2000) - 1000) / 1000.0;     // random Initialization to values in range [-1,1]
}

/* Read in the network parameters (N, K and # threads) from command-line input.
 * the library used here is getopt (GNU) from unistd.h. 
 */
void parse_command_line_parameters(int argc, char *argv[], int *n_threads, int *N, int *K)
{
	int c;
	while ((c = getopt (argc, argv, "t:n:k:")) != -1) {
		switch (c) {
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
	srand(42);
	
	// The default value of N, K and number of threads to be used
	int N = 10;
	int K = 5;
	n_threads = omp_get_max_threads();

	// get N, K and number of threads from command line
	parse_command_line_parameters(argc, argv, &n_threads, &N, &K);
	printf("Using %d threads, input size:%d, number of layers:%d.\n", n_threads, N, K);

	// Judge if the length of the k-th layer is bigger than 0
	int last_layer_len = N - (K-1) * (R-1);
	if (last_layer_len <= 0) {
		printf("The parameters you input couldn't support K layers. \
		        Please give a bigger N or a smaller K.\n");
		return EXIT_FAILURE;
	}

	// create an array which stores all the layer-values of w, b and y
	int first_layer_len = N; // input included
	int total_b_len = K - 1;
	int total_y_len = K * (first_layer_len + last_layer_len) / 2;
	int total_W_len = (total_y_len - N) * R;
	
	float *b = (float*) malloc(total_b_len * sizeof(float));
	float *y = (float*) malloc(total_y_len * sizeof(float));
	float *W = (float*) malloc(total_W_len * sizeof(float));
	
	// initialize the values of y, w and b
	for (int i=0; i < total_y_len; i++) {
        y[i] = random_init_small();
    }
    for (int i=0; i < K-1; i++) {
		b[i] = random_init_small();
    }
    for (int i=0; i < total_W_len; i++) {
		W[i] = random_init_small();
    }
    
    /*/TEST
    for (int i=0; i < total_y_len; i++) {
        printf("%.2f ",y[i]);
    }
    printf("\n");
    for (int i=0; i < K-1; i++) {
		printf("%.2f ",b[i]);
    }
    printf("\n");
    for (int i=0; i < total_W_len; i++) {
		printf("%.2f ",W[i]);
    }
    printf("\n");
    */

	// start recording time
	float t_start = omp_get_wtime();

	for(int k=1; k<K; k++) {
		int layer_len = N - k * (R-1);          // calculate length of this layer
		int in_layer_len = layer_len + R - 1;   // the length of the input layer

        int y_start_idx = k * (N + N - (k-1)*(R-1)) / 2;
        int x_start_idx = (k-1) * (N + N - (k-2)*(R-1)) / 2;
        int W_start_idx = (y_start_idx - N) * R;		

		one_layer_calc(y + x_start_idx, W + W_start_idx, b + (k-1), y + y_start_idx, in_layer_len); // calculation on this layer, update y
	}

	// stop recording time
	float t_end = omp_get_wtime();

	// print the final result
	printf("Final result is: ");
	for(int i=(total_y_len - last_layer_len); i<total_y_len; i++) {
        printf("%.3f ", y[i]);
    }
	printf("\n");

	// print the time consumption
	printf("Elapsed time: %fs\n", t_end - t_start);
	printf("\n");
	
	free(b); free(W); free(y);   // free heap memory

	return EXIT_SUCCESS;
}
