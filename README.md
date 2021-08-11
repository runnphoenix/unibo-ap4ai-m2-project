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
		`./serial -n # of nodes -k # of layers`

- openmp.c
	- Compile with:
		`gcc -fopenmp openmp.c -o openmp`
		OR
		`gcc openmp.c -o openmp -lm`
		where -lm is used for linking the math library

	- Run with:
		`./openmp -t # of threads -n # of nodes -k # of layers`

- cuda.cu
	- Compile with:
		`nvcc cuda.cu -o cuda`

	- Run with:
		`./cuda -n # of nodes -k # of layers`
		
	- Profile with
		`sudo nvprof ./cuda -n # of nodes -k # of layers`
