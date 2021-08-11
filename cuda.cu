/****************************************************************************
 * cuda.cu - a simple multi-layer Nerual Network implemented using cuda
 *
 * Assignment of Module 2 of Ap4AI course of AI master degree @unibo
 *
 * Last updated in 2021 by Hanying Zhang <hanying.zhang@studio.unibo.it>
 *
 * --------------------------------------------------------------------------
 *
 * compile with:
 * nvcc cuda.c -o cuda
 *
 * Run with:
 * ./cuda -n number_of_nodes -k number_of_layers
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define RANDOM_SEED 42
#define R 3
const int BLKDIM = (64/R*R)*R;

/*  BLKDIM optimization
 *  number_of_threads shoule be able to divide (32 * R)
 *  (n_node / R * R) is number_of_nodes being able to divide R  ->
 *  (n_node / R * R) * R is number_of_threads being able to divide R
 */

// Use a __device__ function to calculate Sigmoid
__device__ float Sigmoid(float x)
{
    return 1.0 / ( expf(-x) + 1 );
}

/* The calculation of y values for one layer */
__global__ void one_layer_calc(float *x, float *W, float *b, float *y, int N)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    int lidx = threadIdx.x;                            // local thread index
    int gi = gidx / R;            // global node index
    int li = lidx / R;            // local node index
    int j = gidx - gi * R;        // index of related values in previous layer for each value in y

    int layer_len = N - R + 1; 
    float y_tmp = 0.0;
    
    // shared memory used to store local values in y
    __shared__ float local_y[BLKDIM];

    // calculate (X * W) and store them in shared memory
    if(gi < layer_len && j < R) {
        local_y[lidx] = x[gi + j] * W[gi * R + j];
    }

    __syncthreads();
    
    // Accumulate R values of each node in y
    if(gi < layer_len && j < R){
        for (int p=0; p<R; p++) {
            y_tmp += local_y[li * R + p];
        }
    }
   
    if(gi < layer_len){
        // Sigmoid
        y_tmp = Sigmoid(y_tmp + *b);
        // Copy temp values to y
        y[gi] = y_tmp;
    }
}

/* Random values between -1 and 1 */
float random_init_small()
{
    return ((rand() % 20000) - 10000) / 10000.0; 
}

/* Read in the network parameters (N, K) from command-line input. */
void parse_command_line_parameters(int argc, char *argv[], int *N, int *K)
{
    int c;
    while ((c = getopt (argc, argv, "n:k:")) != -1) {
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

int main( int argc, char *argv[] )
{
    // set random seed
    srand(RANDOM_SEED);
    
    // N, K and their default values
    int N = 5;
    int K = 3;

    // get N, K from command line
    parse_command_line_parameters(argc, argv, &N, &K);
    printf("input size:%d, number of layers:%d.\n",  N, K);

    // Judge if the length of the k-th layer is bigger than 0
    int last_layer_len = N - (K-1) * (R-1);
    if (last_layer_len <= 0) {
        printf("The parameters you input couldn't support k layers. \
                Please give bigger size of layer 0 or use less layers.\n");
        return EXIT_FAILURE;
    } 

    // create an array which stores all the layer-values of w, b and y
    int first_layer_len = N;
    int total_b_len = K - 1;
    int total_y_len = K * (first_layer_len + last_layer_len) / 2;   // input layer included
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
    
    // create GPU version of b, w and y
    float *b_d;
    float *W_d;
    float *y_d;
    
    // Start recording time costage
    clock_t start = clock();
    
    // Cuda memory malloc and copy
    cudaMalloc( (void**)&b_d, total_b_len * sizeof(float) );
    cudaMalloc( (void**)&W_d, total_W_len * sizeof(float) );
    cudaMalloc( (void**)&y_d, total_y_len * sizeof(float) );

    cudaMemcpy(b_d, b, total_b_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_d, W, total_W_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, total_y_len * sizeof(float), cudaMemcpyHostToDevice);

    // Loop over K layers
    for(int k=1; k<K; k++) {
        // calculate lengthes of this layer and the previous layer
        int layer_len = N - k * (R-1);
        int in_layer_len = layer_len + R - 1;

        // calculate the starting indices of w, b and y in this layer
        int x_start_idx = (k-1) * (N + N-(k-2)*(R-1)) / 2;
        int y_start_idx = k * (N + N-(k-1)*(R-1)) / 2;
        int W_start_idx = (y_start_idx - N) * R;
        
        // calculation of each layer
        one_layer_calc<<<(layer_len*R+BLKDIM-1)/BLKDIM, BLKDIM>>>(y_d + x_start_idx, W_d + W_start_idx, b_d + (k-1), \
                                                                  y_d + y_start_idx, in_layer_len);

        cudaDeviceSynchronize();
    }
    
    // Copy result back from device to host
    cudaMemcpy(y, y_d, total_y_len * sizeof(float), cudaMemcpyDeviceToHost);
    
    // calculate elapsed time
    clock_t end = clock();
    double time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", time_elapsed);
    
    // print final result
    printf("Final result is: ");
    for(int i=(total_y_len - last_layer_len); i<total_y_len; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");
    
    // Free memory
    cudaFree(W_d); cudaFree(y_d); cudaFree(b_d);  // free cuda memory
    free(b); free(W); free(y);                    // free heap memory

    return EXIT_SUCCESS;
}
