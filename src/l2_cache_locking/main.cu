/*************************************************************************
 * GCALASY project
 * Copyright (C) 2023 ISAE-SUPAERO
*************************************************************************/

/*--------------------------- main.c -------------------------------------
 |  File main.c
 |
 |  Description:  The evaluation of the L2 hardware-based cache locking 
 |		   of NVIDIA's platforms is made. Tested on the Jetson
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

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include <cuda_runtime.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

// GPU clock management
#include "gpu_clock_meas.cuh"

// GPU cache locking management
#include "gpu_cache_locking.cuh"

// GPU benchmarks
#include "gpu_benchmarks.cuh"

// Benchmarks scenarios
#include "scenario1.h" // Inter-SM interference with one application 
#include "scenario2.h" // Inter-SM interference with several applications
#include "scenario3.h" // DDR SDRAM and inter-SM interference with one application


// #define PRINT_ALL

/* GLOBAL VARIABLES*/

// Cache locking properties
float num_megabytes_persistent_cache = 3;

// Compute kernel constant variables
unsigned const BLOCK_SIZE = 32;
unsigned const SM_MIN_OPT_THREADS_SIZE = 128;
unsigned const MAX_BLOCK_SIZE = 1024;

// Test configuration variables
unsigned benchmark_option = 100;
unsigned multiplier = 18; 

// Maximum amount of SMs on Jetson AGX Orin 64GB
unsigned const MAX_SMs = 16;

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
	printf("Arguments: num_megabytes_persistent_cache, benchmark_option, multiplier\n\n");
	printf("num_megabytes_persistent_cache: Cache space for locking data. Limited to %u MB for this device\n", persisting_l2_cache_max_size_mb);
	printf("benchmark_option: \n Scenario 1: Inter-SM interference with one application: (0) Vector reset, (1) 2D convolution, (2) Matrix upsampling \n Scenario 2: Inter-SM interference with several applications: (10) Vector reset, (11) 2D convolution, (12) 3D convolution \n Scenario 3: DDR SDRAM and inter-SM interference with one application: (20) Vector reset, (21) 2D convolution, (22) Matrix upsampling\n");
	printf("multiplier: Factor used to modify the size of the matrices, thus varying the memory size. Check main function code for more details\n");
    	
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
    
   
   
   //********************************************************
   // Scenario 1: Inter-SM interference with one application
   //********************************************************
   
    // Vector data resetting
    if (benchmark_option == 0){
        size_t size_persistent = multiplier*1024*1024/sizeof(int);
        size_t size_streaming = 100*1024*1024/sizeof(int);

    	#ifdef PRINT_ALL
    		printf("Vectors size (%d,%d)\n", size_persistent, size_streaming);
    	#endif
    	
    	launch_reset_data(BLOCK_SIZE, size_streaming, size_persistent);

    }
    // Matrix convolution
    else if (benchmark_option == 1){

	dim3 dimsA(multiplier * SM_MIN_OPT_THREADS_SIZE, multiplier * SM_MIN_OPT_THREADS_SIZE / 2, 1);
    	dim3 dimsB(3, 3, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	#endif
    
    	MatrixConvolution(SM_MIN_OPT_THREADS_SIZE, dimsA, dimsB);

    }
    // Upsampling via Nearest-Neighbor unpooling
    else if (benchmark_option == 2){

    	unsigned const UPSAMPLING = 2;
		
	dim3 dimsA(multiplier * BLOCK_SIZE, multiplier * BLOCK_SIZE, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d) and upsampling of %u \n", dimsA.x, dimsA.y, UPSAMPLING);
    	#endif
    
    	MatrixUpsampling(BLOCK_SIZE, UPSAMPLING, dimsA);

    }

    
   //*************************************************************
   // Scenario 2: Inter-SM interference with several applications
   //*************************************************************
    
    
    // Vector data resetting in a mix-criticality context
    else if (benchmark_option == 10){
        size_t size_persistent = multiplier*1024*1024/sizeof(int);
        size_t size_streaming = 100*1024*1024/sizeof(int);
	
    	#ifdef PRINT_ALL
    		printf("Vectors size (%d,%d)\n", size_persistent, size_streaming);
    	#endif

    	for(int cnt = 0; cnt < MAX_SMs; cnt++){
	#ifdef PRINT_ALL
    		printf("Number of critical SMs: %d, Number of interfering SMs: %d \n", MAX_SMs-cnt, cnt);
	#endif
    		for(int i = 0; i < 100; i++){
	    		mc_launch_reset_data(MAX_BLOCK_SIZE, size_streaming, size_persistent, cnt);
    		}
    	}

    }
    // Matrix 2D convolution in a mix-criticality context
    else if (benchmark_option == 11){
	dim3 dimsA(multiplier * MAX_BLOCK_SIZE, multiplier * MAX_BLOCK_SIZE / 2, 1);
    	dim3 dimsB(3, 3, 1);
    	
        size_t size_persistent = 3*1024*1024/sizeof(int);
        size_t size_streaming = 100*1024*1024/sizeof(int);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	#endif
    
	for(int cnt = 0; cnt < MAX_SMs; cnt++){
		#ifdef PRINT_ALL
	    		printf("Number of critical SMs: %d, Number of interfering SMs: %d \n", MAX_SMs-cnt, cnt);
		#endif
    		for(int i = 0; i < 100; i++){
    			mc_launch_convolution(MAX_BLOCK_SIZE, dimsA, dimsB, size_streaming, size_persistent, cnt);
    		}
    	}

    }

    // Matrix 3D convolution in a mix-criticality context
    else if (benchmark_option == 12){
	dim3 dimsA(multiplier * 32, multiplier * 32, MAX_BLOCK_SIZE);
    	dim3 dimsB(3, 3, MAX_BLOCK_SIZE);
    	
        size_t size_persistent = 3*1024*1024/sizeof(int);
        size_t size_streaming = 100*1024*1024/sizeof(int);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d,%d), MatrixB(%d,%d,%d)\n", dimsA.x, dimsA.y, dimsA.z, dimsB.x, dimsB.y, dimsB.z);
    	#endif
    
	for(int cnt = 0; cnt < MAX_SMs; cnt++){
	#ifdef PRINT_ALL
    		printf("Number of critical SMs: %d, Number of interfering SMs: %d \n", MAX_SMs-cnt, cnt);
	#endif
    		for(int i = 0; i < 100; i++){
    			mc_launch_convolution3d(MAX_BLOCK_SIZE, dimsA, dimsB, size_streaming, size_persistent, cnt);
    		}
    	}

    }
    
    //**********************************************************************
    // Scenario 3: DDR SDRAM and inter-SM interference with one application
    //**********************************************************************
  
    // Vector data resetting with DDR SDRAM interference
    else if (benchmark_option == 20){
        size_t size_persistent = multiplier*1024*1024/sizeof(int);
        size_t size_streaming = 100*1024*1024/sizeof(int);

    	#ifdef PRINT_ALL
    		printf("Vectors size (%d,%d)\n", size_persistent, size_streaming);
    	#endif
    	
    	launch_reset_data_ddr_interf(BLOCK_SIZE, size_streaming, size_persistent);

    }
    
    // Matrix 2D convolution with DDR SDRAM interference
    else if (benchmark_option == 21){

	dim3 dimsA(multiplier * SM_MIN_OPT_THREADS_SIZE, multiplier * SM_MIN_OPT_THREADS_SIZE / 2, 1);
    	dim3 dimsB(3, 3, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    	#endif
    
    	MatrixConvolution_ddr_interf(SM_MIN_OPT_THREADS_SIZE, dimsA, dimsB);

    }
    
   // Upsampling via Nearest-Neighbor unpooling with DDR SDRAM interference
    else if (benchmark_option == 22){

    	unsigned const UPSAMPLING = 2;
		
	dim3 dimsA(multiplier * BLOCK_SIZE, multiplier * BLOCK_SIZE, 1);

    	#ifdef PRINT_ALL
    		printf("MatrixA(%d,%d) and upsampling of %u \n", dimsA.x, dimsA.y, UPSAMPLING);
    	#endif
    
    	MatrixUpsampling_ddr_interf(BLOCK_SIZE, UPSAMPLING, dimsA);

    }
    
    else{
	printf("Test does not exist\n");
    }

    
    

}




