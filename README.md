# unibo-ap4ai-m2-project

## Source files
- serial.c

	This is the serial implementation of the Nerual Network

- openmp.c

	This is the Openmp version of the Neural Network

- cuda.cu

	This is the Cuda version of the Neural Network
	
## Usage of the source files
- serial.c
	- Compile with:
		`gcc serial.c -o serial`
		OR
		`gcc serial.c -o serial -lm`
		where -lm is used for linking the math library

	- Run with:
		`./serial -n number_of_nodes -k number_of_layers`

- openmp.c
	- Compile with:
		`gcc -fopenmp openmp.c -o openmp`
		OR
		`gcc -fopenmp openmp.c -o openmp -lm`
		where -lm is used for linking the math library

	- Run with:
		`./openmp -t number_of_threads -n number_of_nodes -k number_of_layers`

- cuda.cu
	- Compile with:
		`nvcc cuda.cu -o cuda`

	- Run with:
		`./cuda -n number_of_nodes -k number_of_layers`
		
	- Profile with
		`sudo nvprof ./cuda -n number_of_nodes -k number_of_layers`
