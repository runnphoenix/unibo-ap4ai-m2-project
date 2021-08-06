/****************************************************************************
 * TODO

 ****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define R 3
const int BLKDIM = (128/R*R)*R;

/*  BLKDIM optimization
 *  # of threads shoule be able to divide (32 * R)
 *  (n_node / R * R) is # of nodes being able to divide R  ->
 *  (n_node / R * R) * R is # of threads being able to divide R
 */

__global__ void one_layer_calc(float *x, float *W, float *b, float *y, int N)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int lidx = threadIdx.x;
  	int i = gidx / R;
    int j = gidx - i * R;
    int li = lidx / R;
    int lj = lidx - li * R;
 
    float tmp = 0.0;

    __shared__ float local_y[BLKDIM];

    if(i < N-R+1 && j < R)
    {
        local_y[lidx] = x[i+j] * W[i * R + j];
        //printf("i:%d j:%d lidx: %d x:%.2f W:%.2f y:%.2f \n", i, j, lidx, x[i+j], W[i * R + j], local_y[lidx]);
    }

    __syncthreads();
    //printf("\n");  
    
    for (int p=0; p<R; p++)
    {
    	tmp += local_y[li * R + p];
    	//printf("i:%d j:%d lidx: %d local_y:%.2f tmp:%.2f \n", i,j,lidx, local_y[li * R + p], tmp);
    }

    __syncthreads();
    
    tmp = 1.0 / (expf(-tmp - *b) + 1);
    y[i] = tmp;
}

/* Initialize the W and b parameters for one layer */
void init_layer_parameters(float (*W)[R], float w_v, float *b, float b_v, int layer_len)
{
	for (int i=0; i<layer_len; i++)
	{
		for (int j=0; j<R; j++)
			{
				W[i][j] = w_v;
			}
	}

	*b = b_v;
}

/* Read in the network parameters (N, K) from command-line input. */
void parse_command_line_parameters(int argc, char *argv[], int *N, int *K)
{
	int c;
	while ((c = getopt (argc, argv, "n:k:")) != -1)
	{
		switch (c)
		{
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
    memcpy(activation, x, N*sizeof(float));

    float *activation_d;
    cudaMalloc((void**)&activation_d, N*sizeof(float));
    cudaMemcpy(activation_d, activation, N*sizeof(float), cudaMemcpyHostToDevice);

    clock_t start = clock();

    // Loop over k layers
  	for(int t=1; t<K; t++)
    {
        // calculate length of this layer
        int layer_len = N - t * (R-1);

  		// initialize parameters b
        float b;
        float W[layer_len][R];
        float y[layer_len];
     
        //float b_v = rand() % 3 - 1;
        //float W_v = ((rand() % 2000) - 1000) / 1000.0;
        float b_v = 1.0;
        float W_v = 1.0 / 3;
        init_layer_parameters(W, W_v, &b, b_v, layer_len);
     
        float *b_d;
        float *W_d;
        float *y_d;
    
        cudaMalloc((void**)&b_d, sizeof(float));
        cudaMalloc((void**)&W_d, layer_len*R*sizeof(float));
        cudaMalloc((void**)&y_d, layer_len*sizeof(float));

        cudaMemcpy(b_d, &b, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(W_d, W, layer_len*R*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, y, layer_len*sizeof(float), cudaMemcpyHostToDevice);

  		// do the calculation
        //printf("\nGRIDDIM %d BLKDIM: %d\n", (layer_len*R+BLKDIM-1)/BLKDIM, BLKDIM);
  		one_layer_calc<<<(layer_len*R+BLKDIM-1)/BLKDIM, BLKDIM>>>(activation_d, W_d, b_d, y_d, layer_len+R-1);
        cudaDeviceSynchronize();

        // copy result back
        cudaMemcpy(y, y_d, layer_len*sizeof(float), cudaMemcpyDeviceToHost);

		//* TEST
        printf("\nThe layer result got\n");
        for(int i=0; i<layer_len; i++){
            printf("%.2f ", y[i]);
        }
        printf("\n");

  		// save the activation result
  		memcpy(activation, y, layer_len * sizeof(float));
        cudaMemcpy(activation_d, activation, layer_len*sizeof(float), cudaMemcpyHostToDevice);

        // free cuda memory
        cudaFree(W_d); cudaFree(y_d); cudaFree(b_d);
  	}


	// calculate elapsed time
	clock_t end = clock();
	double time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Time elapsed: %.3f\n", time_elapsed);

    cudaFree(activation_d);

	// print final result
	printf("\nFinal result is: ");
	int last_layer_len = N-(K-1)*(R-1);
	for(int i=0; i<last_layer_len; i++){
		printf("%.3f ", activation[i]);
	}
	printf("\n");

    return EXIT_SUCCESS;
}
