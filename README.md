# PQP-for-MPC
This is an implementation of PQP algorithm for MPC in C language

PQP_CPU: 		CPU version of the PQP algorithm
PQP_GPU_unoptimized: 	GPU version without tiling
PQP_GPU_optimized:	GPU version with tiling

to run the code:

PQP_CPU:		gcc -lm PQP_CPU.c -o PQP_CPU
			$ ./PQP_CPU

PQP_GPU_unoptimized:	nvcc PQP_GPU_unoptimized.cu -o PQP_GPU_unoptimized
			$ ./PQP_GPU_unoptimized

PQP_GPU_optimized:	nvcc PQP_GPU_optimized.cu -o PQP_GPU_optimized
			$ ./PQP_GPU_optimized
