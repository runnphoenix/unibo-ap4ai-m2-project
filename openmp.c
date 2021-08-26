/****************************************************************************
 * openmp.c - a simple multi-layer Nerual Network implemented using openmp
 *
 * Assignment of Module 2 of Ap4AI course of AI master degree @unibo
 *
 * Last updated in 2021 by Hanying Zhang <hanying.zhang@studio.unibo.it>
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
 * ./openmp -t number_of_threads -n number_of_nodes -k number_of_layers
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RANDOM_SEED 42
#define R 3
int n_threads;

/* Sigmoid function */
float Sigmoid(float x)
{
    return 1.0 / (exp(-x) + 1);
}

/* The calculation of y values for one layer */
void one_layer_calc_pre(float *x, float *W, float *b, float *y, int N)
{
    #pragma omp parallel num_threads(n_threads)
    {
        int i,j,k;
        #pragma omp for collapse(2) reduction(+:y[0:N-R+1]) schedule(static)
        for (i=0; i<N-R+1; i++) {
            for (j=0; j<R; j++) {
                y[i] += x[i+j] * W[i*R+j];
            }
        }
        
        //#pragma omp barrier
        #pragma omp for schedule(static)
        for (k=0; k<N-R+1; k++) {
            y[k] = Sigmoid(y[k] + *b);  // +b, then sigmoid
        }
    }
}

/* The calculation of y values for one layer */
void one_layer_calc(float *x, float *W, float *b, float *y, int N)
{
    int i,j;
    #pragma omp parallel for private(j) num_threads(n_threads) schedule(static)
    for (i=0; i<N-R+1; i++) {
        for (j=0; j<R; j++) {
            y[i] += x[i+j] * W[i*R+j];
        }
        y[i] = Sigmoid(y[i] + *b);  // +b, then sigmoid
    }
}

/* Random values between -1 and 1 */
float random_init_small()
{
    return ((rand() % 20000) - 10000) / 10000.0;
}

/* Initialize the values of y, W and b */
void initialize_parameters(float *y, int y_len, int first_layer_len, float *W, int W_len, float *b, int b_len) 
{
    for (int i=0; i < y_len; i++) {
        if(i < first_layer_len) {
            y[i] = random_init_small();
        }
        else {
            y[i] = 0.0;
        }
    }
    for (int i=0; i < b_len; i++) {
        b[i] = random_init_small();
    }
    for (int i=0; i < W_len; i++) {
        W[i] = random_init_small();
    }
}

/* Read in the network parameters (N, K and n_threads) from command-line input
 * the library used here is getopt (GNU) from unistd.h */
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
    // Set random seed
    srand(RANDOM_SEED);
    
    // The default value of N, K and number of threads to be used
    int N = 5;
    int K = 3;
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
    int first_layer_len = N; 
    int total_b_len = K - 1;
    int total_y_len = K * (first_layer_len + last_layer_len) / 2;  // input layer included
    int total_W_len = (total_y_len - N) * R;
    
    float *b = (float*) malloc(total_b_len * sizeof(float));
    float *y = (float*) malloc(total_y_len * sizeof(float));
    float *W = (float*) malloc(total_W_len * sizeof(float));
    
    // initialize parameters
    initialize_parameters(y, total_y_len, first_layer_len, W, total_W_len, b, total_b_len);

    // start recording time
    float t_start = hpc_gettime();
    
    for(int k=1; k<K; k++) {
        int layer_len = N - k * (R-1);          // calculate length of this layer
        int in_layer_len = layer_len + R - 1;   // the length of the input layer

        // calculate the starting indices of w, b and y in this layer
        int x_start_idx = (k-1) * (N + N-(k-2)*(R-1)) / 2;
        int y_start_idx = k * (N + N-(k-1)*(R-1)) / 2;
        int W_start_idx = (y_start_idx - N) * R;

        // calculation on this layer
        one_layer_calc(y + x_start_idx, W + W_start_idx, b + (k-1), \
                       y + y_start_idx, in_layer_len); 
    }

    // stop recording time
    float t_end = hpc_gettime();

    // print the final result
    printf("Final result is: ");
    for(int i=(total_y_len - last_layer_len); i<total_y_len; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");
    
    // print the time consumption
    printf("Elapsed time: %e seconds.\n", t_end - t_start);
    
    // free heap memory
    free(b); free(W); free(y);   

    return EXIT_SUCCESS;
}
