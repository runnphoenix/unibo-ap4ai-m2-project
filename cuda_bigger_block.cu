/****************************************************************************
 * cuda_bigger_block.cu - a simple multi-layer Nerual Network
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
 * nvcc cuda_bigger_block.c -o cuda_bigger_block
 *
 * Run with:
 * ./cuda_bigger_block -n # of nodes -k # of layers
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define R 3
const int BLKDIM = (64/R*R)*R;

/*  BLKDIM optimization
 *  # of threads shoule be able to divide (32 * R)
 *  (n_node / R * R) is # of nodes being able to divide R  ->
 *  (n_node / R * R) * R is # of threads being able to divide R
 */

// Use a __device__ function to calculate Sigmoid
__device__ float Sigmoid(float x, float b)
{
    return 1.0 / ( expf(-x - b) + 1 );
}

/* The calculation of y values for one layer */
__global__ void one_layer_calc(float *x, float *W, float *b, float *y, int N)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    int lidx = threadIdx.x;                            // local thread index
    int gi = gidx / R;            // global node index
    int li = lidx / R;            // local node index
    int j = gidx - gi * R;        // index of related values in previous layer for each value in y

    int pre_layer_len = N - R + 1; 
 
    float y_tmp = 0.0;
    // shared memory used to store local values in y
    __shared__ float local_y[BLKDIM];

    if(gi < pre_layer_len && j < R) {
        local_y[lidx] = x[gi + j] * W[gi * R + j];
        //printf("i:%d j:%d lidx: %d x:%.2f W:%.2f y:%.2f \n", \
                  gi, j, lidx, x[gi+j], W[gi * R + j], local_y[lidx]);
    }

    __syncthreads();
    //printf("\n");  
    
    // Accumulate R values of each node in y
    for (int p=0; p<R; p++) {
        y_tmp += local_y[li * R + p];
        //printf("i:%d j:%d lidx: %d local_y:%.2f tmp:%.2f \n", gi,j,lidx, local_y[li * R + p], y_tmp);
    }

    __syncthreads();
   
    // Sigmoid
    y_tmp = Sigmoid(y_tmp, *b);

    // Copy temp values to y
    y[gi] = y_tmp;
}

/* Initialize the W and b parameters for one layer */
//TODO: change to initialize all parameters
void init_layer_parameters(float (*W)[R], float w_v, float *b, float b_v, int layer_len)
{
    for (int i=0; i<layer_len; i++) {
        for (int j=0; j<R; j++) {
            W[i][j] = w_v;
        }
    }

    *b = b_v;
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
    int N = 100;
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
	int first_layer_len = N; // input included
	int total_b_len = K - 1;
	int total_y_len = K * (first_layer_len + last_layer_len) / 2;
	int total_W_len = total_y_len * R;
	
	float *b = (float*)malloc(total_b_len * sizeof(float));
	float *y = (float*)malloc(total_y_len * sizeof(float));
	float *W = (float*)malloc(total_W_len * sizeof(float));
	
	// initialize the values of y, w and b
	//float b_v = rand() % 3 - 1;
    //float W_v = ((rand() % 2000) - 1000) / 1000.0;
    //init_layer_parameters(W, W_v, &b, b_v, layer_len);
	for (int i=0; i < total_y_len; i++) {
		if(i < N) {
        	y[i] = -1.0;
        }
        else {
        	y[i] = 0.0;
        }
    }
    for (int i=0; i < K-1; i++) {
		b[i] = 1.0;
    }
    for (int i=0; i < total_y_len * R; i++) {
		W[i] = 1.0 / 3;
    }
	
	// create gpu related b, w and y
	float *b_d;
    float *W_d;
    float *y_d;
    
    // Start recording time costage
    clock_t start = clock();
	
	// cuda memcpy
	cudaMalloc( (void**)&b_d, (K-1)       * sizeof(float) );
    cudaMalloc( (void**)&W_d, total_W_len * sizeof(float) );
    cudaMalloc( (void**)&y_d, total_y_len * sizeof(float) );

    cudaMemcpy(b_d, b, (K-1)       * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_d, W, total_W_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, total_y_len * sizeof(float), cudaMemcpyHostToDevice);

    // Loop over K layers
    for(int k=1; k<K; k++) {
        // calculate lengthes of this layer and the previous layer
        int layer_len = N - k * (R-1);
        int in_layer_len = layer_len + R - 1;

        // calculation of each layer
        // printf("\nGRIDDIM %d BLKDIM: %d\n", (layer_len*R+BLKDIM-1)/BLKDIM, BLKDIM);
        int y_start_idx = k * (N + N - (k-1)*(R-1)) / 2;
        int x_start_idx = (k-1) * (N+N-(k-2)*(R-1)) / 2;
        int W_start_idx = y_start_idx * R;
        
        one_layer_calc<<<(layer_len*R+BLKDIM-1)/BLKDIM, BLKDIM>>>(y_d + x_start_idx, W_d + W_start_idx, b_d + (k-1), y_d + y_start_idx, in_layer_len);

        cudaDeviceSynchronize();
    }
    
    // copy result back to host
    cudaMemcpy(y, y_d, total_y_len * sizeof(float), cudaMemcpyDeviceToHost);
    // Free cuda memory
    cudaFree(W_d); cudaFree(y_d); cudaFree(b_d);

    // calculate elapsed time
    clock_t end = clock();
    double time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed: %.3f\n", time_elapsed);

    cudaFree(b_d); cudaFree(W_d); cudaFree(y_d);

    // print final result
    printf("\nFinal result is: ");
    for(int i=(total_y_len - last_layer_len); i<total_y_len; i++) {
        printf("%.3f ", y[i]);
    }
    printf("\n");

    return EXIT_SUCCESS;
}
