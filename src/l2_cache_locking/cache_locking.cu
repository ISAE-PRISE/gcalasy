// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// GCALASY project
// Copyright (C) 2023 ISAE
// 
// Purpose:
// Studying the response of cache memory interference to hardware-based 
// cache locking technique on iGPUs
//
// Contacts:
// alfonso.mascarenas-gonzalez@isae-supaero.fr
// jean-baptiste.chaudron@isae-supaero.fr
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------

/*--------------------------- main.c -------------------------------------
 |  File main.c
 |
 |  Description:  The evaluation of the L2 hardware-based cache locking 
 |		   of NVIDIA's platforms is made. Tested for the Jetson
 |		   AGX Orin 64 GB.
 |
 |  Version: 1.0
 |
 | Contact:
 | alfonso.mascarenas-gonzalez@isae-supaero.fr
 | jean-baptiste.chaudron@isae-supaero.fr
 *-----------------------------------------------------------------------*/

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

// GPU clock management
#include "gpu_clock_meas.cuh"

// GPU cache locking management
#include "gpu_cache_locking.cuh"

// Benchmarks scenarios
#include "benchmarks.h"


/* GLOBAL VARIABLES*/

// #define PRINT_ALL
// #define EVENT_RECORD_TIME 
// #define BENCHMARK_CORRECTNESS


// Cache locking properties
float num_megabytes_persistent_cache = 3;

// Compute kernel variables
unsigned const BLOCK_SIZE = 32;
unsigned const SM_MIN_OPT_THREADS_SIZE = 128;

// Test configuration variables
unsigned benchmark_option = 100;
unsigned multiplier = 18; 



int main(int argc, char* argv[]){

    // Retrieve device properties
    cudaDeviceProp device_prop;
    int current_device = 0;
    checkCudaErrors(cudaGetDevice(&current_device));
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, current_device));
    unsigned persisting_l2_cache_max_size_mb = device_prop.persistingL2CacheMaxSize / 1024 / 1024;
    
    #ifdef PRINT_ALL
   	printf("\n***** Basic Info *****\n");
    	printf("GPU: %s \n", device_prop.name);
    	printf("L2 Cache Size: %u MB \n", device_prop.l2CacheSize / 1024 / 1024);
    	printf("Max Persistent L2 Cache Size: %u MB \n\n", persisting_l2_cache_max_size_mb);
    #endif
    
    // Program arguments processing 
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "-h")){
	printf("***** HELP *****\n");
	printf("Arguments: num_megabytes_persistent_cache, benchmark_option, multiplier\n");
	printf("num_megabytes_persistent_cache: Cache space for locking data. Limited to %u MB for this device\n", persisting_l2_cache_max_size_mb);
	printf("benchmark_option: Type of benchmark to execute: (0) Matrix dot product, (1) Matrix-vector multiplication addition, (2) Matrix convolution, (3) Matrix upsampling, (other) vector reset\n");
	printf("multiplier: Factor used to modify the size of the matrices, thus varying the memory size. Check main function code for further details\n");
    	
    	return -100;
    }
    else if (argc == 4) {
        num_megabytes_persistent_cache = std::atof(argv[1]);
        benchmark_option = std::atoi(argv[2]);
        multiplier = std::atoi(argv[3]);

        if (persisting_l2_cache_max_size_mb < num_megabytes_persistent_cache){
        	printf("The device L2 cache does not support more than: %u MB of persistent space\n", persisting_l2_cache_max_size_mb);
        	
        	return -200;
        }
        
        #ifdef PRINT_ALL
        printf("L2 cache locked memory space: %u MB \n", num_megabytes_persistent_cache);
        #endif
    }
    
    // Matrix dot product
    if (benchmark_option == 0){

    	dim3 dimsA(multiplier * BLOCK_SIZE, multiplier * BLOCK_SIZE, 1);
    	dim3 dimsB(multiplier * BLOCK_SIZE, multiplier * BLOCK_SIZE, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	#endif
    
   	MatrixMultiply(BLOCK_SIZE, dimsA, dimsB);


    }
    // Matrix-vector multiplication-addition chain
    else if (benchmark_option == 1){   

	dim3 dimsA(multiplier * SM_MIN_OPT_THREADS_SIZE, multiplier * SM_MIN_OPT_THREADS_SIZE / 2, 1);
    	dim3 dimsB(multiplier * SM_MIN_OPT_THREADS_SIZE, 1, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d), Vector_i(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	#endif
    	
    	MatrixVectorMulAddChain(BLOCK_SIZE, dimsA, dimsB);
    	
    }
    // Matrix convolution
    else if (benchmark_option == 2){

	dim3 dimsA(multiplier * SM_MIN_OPT_THREADS_SIZE, multiplier * SM_MIN_OPT_THREADS_SIZE / 2, 1);
    	dim3 dimsB(3, 3, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	#endif
    
    	MatrixConvolution2(SM_MIN_OPT_THREADS_SIZE, dimsA, dimsB);

    }
    // Upsampling via Nearest-Neighbor unpooling
    else if (benchmark_option == 3){

    	unsigned const UPSAMPLING = 2;
		
	dim3 dimsA(multiplier * BLOCK_SIZE, multiplier * BLOCK_SIZE, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d) and upsampling of %u \n", dimsA.x, dimsA.y, UPSAMPLING);
    	#endif
    
    	MatrixUpsampling(BLOCK_SIZE, UPSAMPLING, dimsA);

    }
    // Vector data resetting
    else{
        size_t size_persistent = multiplier*1024*1024/sizeof(int);
        size_t size_streaming = 1000*1024*1024/sizeof(int);
	
    	#ifdef PRINT_ALL
    		printf("Vectors size (%d,%d)\n", size_persistent, size_streaming);
    	#endif
    	
    	launch_reset_data(BLOCK_SIZE, size_streaming, size_persistent);
	
    }
    
    

}




