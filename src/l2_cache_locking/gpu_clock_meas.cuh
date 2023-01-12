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

/*--------------------------- gpu_clock_meas.cuh ------------------------
|  File gpu_clock_meas.cuh
|
|  Description: Declarations for the GPU time stamp value retrieval  
|
|  Version: 1.0
*-----------------------------------------------------------------------*/
 
// Host and device clock variables
long long int* clock_start_host = NULL;
long long int* clock_start_device = NULL;
    
long long int* clock_end_host = NULL;
long long int* clock_end_device = NULL;
  
// Resultant execution time   
long long unsigned latency = 0;


/* init_clock
 *
 * Description: Initializes the clock host and device variables following a zero-copy approach
 *
 * Parameter:   None
 *
 * Returns:     Nothing
 *
 * */
void init_clock(){
  checkCudaErrors(cudaHostAlloc((void **)&clock_start_host, sizeof(long long unsigned), cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc((void **)&clock_end_host, sizeof(long long unsigned), cudaHostAllocMapped));
    
  checkCudaErrors(cudaHostGetDevicePointer((void **)&clock_start_device, (void *)clock_start_host, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&clock_end_device, (void *)clock_end_host, 0)); 
}


/* calculate_time_diff_clock
 *
 * Description: Calculates the time difference 
 *
 * Parameter:   
 * 		- nb_iter Number of iterations that the task was executed
 *
 * Returns:     Nothing
 *
 * */
void calculate_time_diff_clock(int nb_iter){
   if (*clock_end_host > *clock_start_host)
   	latency = (*clock_end_host - *clock_start_host) / nb_iter;   
    else
    	latency = ((0xFFFFFFFFFFFFFFFF - *clock_start_host) + *clock_end_host) / nb_iter;
}


/* print_time_diff_clock
 *
 * Description: Displays the execution time in GPU clock cycles 
 *
 * Parameter:   None
 *
 * Returns:     Nothing
 *
 * */
void print_time_diff_clock(){
	#ifdef PRINT_ALL
	    	printf("L2 Cache cache locking execution time: %llu iGPU cycles \n", latency);
	#else
		printf("%llu \n", latency);
	#endif
}


/* clean_clock
 *
 * Description: Frees the dynamic memory allocated for the host at initialization 
 *
 * Parameter:   None
 *
 * Returns:     Nothing
 *
 * */
void clean_clock(){
  checkCudaErrors(cudaFreeHost(clock_start_host));
  checkCudaErrors(cudaFreeHost(clock_end_host));
}


/* getDeviceTimeCUDA
 *
 * Description: CUDA kernel for reading the GPU time stamp register 
 *
 * Parameter:   
 *		- long long int* saved_time: Stores the time stamp value 
 *
 * Returns:     Nothing
 *
 * */
__global__ void getDeviceTimeCUDA(long long int* saved_time){
	*saved_time = clock64();
}




