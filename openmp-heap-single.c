#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const int R = 3;
int n_threads;

float* single_layer(float *x, int N, float (*W)[R], float b) {
	
	float *y = malloc(N-R+1);
	#pragma omp parallel for num_threads(n_threads)
	for(int i=0; i<N-R+1; i++){
		y[i] = 0.0;
	}
	
	int i,j;
	#pragma omp parallel for collapse(2) num_threads(n_threads)
	for (i=0; i<N-R+1; i++) {
		for (j=0; j<R; j++) {
			#pragma omp atomic
			y[i] += x[i+j] * W[i][j];
			//printf("tid: %d\n", omp_get_thread_num());
			if(j == R-1) {
				y[i] += b;
			}
		}
	}
	
	#pragma omp parallel for num_threads(n_threads)
	for(int i=0; i<N-R+1; i++) {
		y[i] = 1.0 / (exp(-y[i]) + 1);
	}

	return y;
}

int main(int argc, char *argv[]) {
	
	int N = 10;
	int K = 3;
	n_threads = omp_get_max_threads();
	
	// get parameters from command line
	int c;
	while ((c = getopt (argc, argv, "t:n:k:")) != -1)
		switch (c)
	{
		case 't':
			n_threads = atoi(optarg);
			break;
		case 'n':
			N = atoi(optarg);
			break;
		case 'k':
			K = atoi(optarg);
			break;
	}
	printf("%d %d %d\n", n_threads, N, K);
	
	// Judge if the length of the k-th layer is bigger than 0
	if (N - (K-1) * (R-1) <= 0) {
		printf("The parameters you input couldn't support k layers. Please give bigger size of layer 0 or use less layers.\n");
		return 0;
	}
	
	// initialize the values of the first layer to 1
	float x[N];
	for (int i=0; i < N; i++){
		x[i] = 1.0;
	}
	
	// create a activation
	float activation[N];
	memcpy(activation, x, N*sizeof(float));
	
	// start recording time
	float t_start = omp_get_wtime();
	
	// Loop over k layers
	for(int t=1; t<K; t++) {
		// calculate length of this layer
		int layer_len = N - t * (R-1);
		
		// initialize parameters w and b
		float b = rand() % 3 - 1;
		
		float W[layer_len][R];
		// random Initialization to range [-1,1]
		#pragma omp parallel for collapse(2) num_threads(n_threads)
		for (int i=0; i<layer_len; i++) {
			for (int j=0; j<R; j++) {
				W[i][j] = ((rand() % 2000) - 1000) / 1000.0;
			}
		}
				
		// do the calculation
		float *result = single_layer(activation, layer_len+R-1, W, b);
		
		// save the activation result
		memcpy(activation, result, layer_len * sizeof(float));
		
		// release y
		free(result);
		result = NULL;
	}
	
	float t_end = omp_get_wtime();
	
	// print final result
	int last_layer_len = N-(K-1)*(R-1);
	for(int i=0; i<last_layer_len; i++){
		printf("%.3f ", activation[i]);
	}
	printf("\n");
	
	printf("Elapsed time: %f\n", t_end - t_start);
			
	return 0;
}
