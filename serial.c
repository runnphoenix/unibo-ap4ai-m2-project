/****************************************************************************
 * serial.c - a simple multi-layer Nerual Network implemented serially
 *
 * Assignment of Module 2 of Ap4AI course of AI master degree @unibo
 *
 * Last updated in 2021 by Hanying Zhang <hanying.zhang@studio.unibo.it>
 *
 * --------------------------------------------------------------------------
 *
 * compile with:
 * gcc serial.c -o serial
 * OR
 * gcc serial.c -o serial -lm
 * where -lm is used for linking the math library
 *
 * Run with:
 * ./serial -n number_of_nodes -k number_of_layers
 *
 ****************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define RANDOM_SEED 42
#define R 3

/* Sigmoid function */
float Sigmoid(float x)
{
    return 1.0 / (exp(-x) + 1);
}

/* The calculation of y values for one layer */
void one_layer_calc(float *x, float *W, float *b, float *y, int N)
{
    int i,j;
    for (i=0; i<N-R+1; i++) {
        y[i] = 0.0;
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
    // set random seed
    srand(RANDOM_SEED);
    
    // The default value of N, K
    int N = 5;
    int K = 3;

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

    // create an array which stores all the layer-values of w, b and y
    int first_layer_len = N; 
    int total_b_len = K - 1;
    int total_y_len = K * (first_layer_len + last_layer_len) / 2;  // input layer included
    int total_W_len = (total_y_len - N) * R;
    
    float *b = (float*) malloc(total_b_len * sizeof(float));
    float *y = (float*) malloc(total_y_len * sizeof(float));
    float *W = (float*) malloc(total_W_len * sizeof(float));
    
    // initialize the values of y, w and b
    for (int i=0; i < total_y_len; i++) {
        y[i] = random_init_small();
    }
    for (int i=0; i < total_b_len; i++) {
        b[i] = random_init_small();
    }
    for (int i=0; i < total_W_len; i++) {
        W[i] = random_init_small();
    }

    // Start recording time costage
    clock_t start = clock();

    // Loop over K layers
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

    // calculate elapsed time
    clock_t end = clock();
    double time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %fs\n", time_elapsed);

    // print the final result
    printf("Final result is: ");
    for(int i=(total_y_len - last_layer_len); i<total_y_len; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");
    
    // free heap memory
    free(b); free(W); free(y);   

    return EXIT_SUCCESS;
}
