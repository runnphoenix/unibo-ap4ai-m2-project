/****************************************************************************
 * TODO
 * 1. extract layer_initialize function
 * 2. extract parameter_parsing function
 ****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define R 3
const int BLKDIM = R;

__global__ void single_layer(float *x, int N, float *W, float *b, float *y)
{
  	int i = blockIdx.x;
    int j = threadIdx.x;

    __shared__ float local_y[R];

    if(i < N-R+1 && j < R)
    {
  		local_y[j] = x[i+j] * W[i * blockDim.x + j];
        printf("tidx %d %d x:%.2f W:%.2f y:%.2f \t", i, j, x[i+j], W[i * blockDim.x + j], local_y[j]);
    }

    __syncthreads();

    for(int k=0; k<R; k++)
    {
        y[i] += local_y[k];
    }

    if(j == R-1)
    {
        y[i] += *b;
        y[i] = 1.0 / (exp(-y[i]) + 1);
    }
}

int main( int argc, char *argv[] )
{
    int N = 10;
  	int K = 4;

    // get parameters from command line
    int c;
    while ((c = getopt (argc, argv, "n:k:")) != -1)
    {
        switch (c)
        {
            case 'n':
                N = atoi(optarg);
                break;
            case 'k':
                K = atoi(optarg);
                break;
        }
    }

    printf("%d %d\n", N, K);

    // Judge if the length of the k-th layer is bigger than 0
  	if (N - (K-1) * (R-1) <= 0) {
  		printf("The parameters you input couldn't support k layers. Please give bigger size of layer 0 or use less layers.\n");
  		return 0;
  	}

    // initialize the values of the first layer to 1
  	float x[N];
  	for (int i=0; i < N; i++) {
  		x[i] = -1.0;
  	}

    // create an activation
  	float activation[N];

    float *activation_d;
    cudaMalloc((void**)&activation_d, N*sizeof(float));
  	memcpy(activation, x, N*sizeof(float));
    cudaMemcpy(activation_d, activation, N*sizeof(float), cudaMemcpyHostToDevice);

  	// start recording time

    // Loop over k layers
  	for(int t=1; t<K; t++) {
        // calculate length of this layer
        int layer_len = N - t * (R-1);

  		// initialize parameters b
  		//float b = rand() % 3 - 1;
        float b = 1.0;
        float W[layer_len][R];
        float y[layer_len];


        float *b_d;
        float *W_d;
        float *y_d;


        cudaMalloc((void**)&b_d, sizeof(float));
        cudaMalloc((void**)&W_d, layer_len*R*sizeof(float));
        cudaMalloc((void**)&y_d, layer_len*sizeof(float));


  		// random Initialization to range [-1,1]
  		#pragma omp parallel for collapse(2) num_threads(n_threads)
  		for (int i=0; i<layer_len; i++) {
  			for (int j=0; j<R; j++) {
  				//W[i][j] = ((rand() % 2000) - 1000) / 1000.0;
                  W[i][j] = 1.0 / 3;
  			}
  		}
        for(int i=0; i<layer_len; i++){
            y[i] = 0.0;
        }


        cudaMemcpy(b_d, &b, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(W_d, W, layer_len*R*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, y, layer_len*sizeof(float), cudaMemcpyHostToDevice);

  		// do the calculation
  		single_layer<<<layer_len, BLKDIM>>>(activation_d, layer_len+R-1, W_d, b_d, y_d);

        cudaDeviceSynchronize();

        // copy result back
        cudaMemcpy(y, y_d, layer_len*sizeof(float), cudaMemcpyDeviceToHost);

  		// save the activation result
  		memcpy(activation, y, layer_len * sizeof(float));
        cudaMemcpy(activation_d, activation, layer_len*sizeof(float), cudaMemcpyHostToDevice);

        // free cuda memory
        cudaFree(W_d); cudaFree(y_d); cudaFree(b_d);
  	}

	// print final result
	printf("\nFinal result is: ");
	int last_layer_len = N-(K-1)*(R-1);
	for(int i=0; i<last_layer_len; i++){
		printf("%.3f ", activation[i]);
	}
	printf("\n");

    return EXIT_SUCCESS;
}
