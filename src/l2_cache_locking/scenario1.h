/*----------------------------- scenario1.h -----------------------------
|  File scenario1.h
|
|  Description: Tests for evaluating the L2 cache locking 
|		under inter-SM interference with one application 
|
|  Version: 1.0
*-----------------------------------------------------------------------*/

//#define RECORD_SM_CYCLES 
#define EVENT_RECORD_TIME 

extern float num_megabytes_persistent_cache;



/* init_array
 *
 * Description: Initializes an array  
 *
 * Parameter:   
 *		- float *data: Array to initialize
 *		- int size: Size of the array
 *		- float val: Value to initialize the array with
 *
 * Returns:     Nothing
 *
 * */
void init_array(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) 
  	data[i] = val;
}


/* launch_reset_data
 *
 * Description: Performs a vector data reset with elements of a smaller vector using a CUDA kernel
 *		 and cache locking (locking the smaller vector)
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- size_t size_streaming: Size of data_streaming
 *		- size_t size_persistent: Size of lut_persistent 
 *
 * Returns:     Nothing
 *
 * */
void launch_reset_data(int block_size, size_t size_streaming, size_t size_persistent){

  #ifdef RECORD_SM_CYCLES 
	  // Initialize variables for the GPU clock
 	 init_clock();
  #endif

	
  // Initialize host vectors
  std::vector<int> lut_persistent_vec(size_persistent, 0);
  for (size_t i = 0; i < lut_persistent_vec.size(); ++i)
  	lut_persistent_vec[i] = i;
	
  std::vector<int> data_streaming_vec(size_streaming, 0);

  int* h_lut_persistent = lut_persistent_vec.data();
  int* h_data_streaming = data_streaming_vec.data();
  
  
  // Allocate device memory for vectors
  int* d_lut_persistent;
  int* d_data_streaming;
  
  cudaMalloc(&d_lut_persistent, size_persistent * sizeof(int));
  cudaMalloc(&d_data_streaming, size_streaming * sizeof(int));
  cudaMemcpy(d_lut_persistent, h_lut_persistent, size_persistent * sizeof(int), cudaMemcpyHostToDevice);
    	
  // Stream creation
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreate(&m0_stream));
	
  // Enable cache locking to the stream m0_stream	
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_lut_persistent, size_persistent*sizeof(int)); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute); 

  // Setup execution parameters
  dim3 const threads_per_block = 1024;
  dim3 const blocks_per_grid = block_size;
  
  // Times the vector resetting is performed      
  unsigned nIter = 100;
  
  #ifdef EVENT_RECORD_TIME 
    	  // Allocate CUDA timing events
	  cudaEvent_t start, stop;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));

	  // Record the start event
	  checkCudaErrors(cudaEventRecord(start, m0_stream));
  #endif
  
  #ifdef RECORD_SM_CYCLES 
	  // Get GPU time stamp value for the first time
	  getDeviceTimeCUDA<<<16, 1, 0, m0_stream>>>(clock_start_device);
	  
	  cudaDeviceSynchronize();   
  #endif                  
  
  // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)   
  	reset_data<<<blocks_per_grid, threads_per_block, 0, m0_stream>>>(d_data_streaming, d_lut_persistent, size_streaming, size_persistent);


  #ifdef RECORD_SM_CYCLES 
	  // Wait for the end of execution of all the threads blocks
	  cudaDeviceSynchronize();     
	    
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<16, 1, 0, m0_stream>>>(clock_end_device);
	  // Wait until the GPU time stamp has been accessed
	  cudaDeviceSynchronize();
	    
	  // Calculate execution time in cycles and display it
	  calculate_time_diff_clock(nIter); 
	  print_time_diff_clock();
	  
	  // Free memory
	  clean_clock();	 
  #endif    

  
  #ifdef EVENT_RECORD_TIME    
	  // Record the stop event
	  checkCudaErrors(cudaEventRecord(stop, m0_stream));

	  // Wait for the stop event to complete
	  checkCudaErrors(cudaEventSynchronize(stop));

	  float msecTotal = 0.0f;
	  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	  // Compute and print the performance
	  float msecPerMatrixMul = msecTotal / nIter;                        
	  printf( "%.3f msec \n", msecPerMatrixMul);
	  
	  // Remove events
  	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
  #endif


  // Unlock persistent memory on the L2 cache
  reset_l2_locking_attr(&stream_locking_attribute, m0_stream);
  
  // Free memory
  cudaStreamDestroy(m0_stream);
  cudaFree(d_lut_persistent);
  cudaFree(d_data_streaming);
  		
}


/* MatrixConvolution
 *
 * Description: Performs a set of matrix convolution operations using a CUDA kernel
 *		and cache locking (locking the input matrix). Eeach convolution is 
 *		stored in a different matrix.
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- const dim3 &dimsA: Dimensions of matrix A 
 *		- const dim3 &dimsB: Dimensions of matrix B
 *
 * Returns:     Nothing
 *
 * */
void MatrixConvolution(int block_size, const dim3 &dimsA, const dim3 &dimsB) {
   #ifdef RECORD_SM_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
   #endif  
         
  // Allocate host memory for matrices A, B and C
  unsigned int size_A = dimsA.x * dimsA.y; // Image
  unsigned int size_B = dimsB.x * dimsB.y; // Filter
  dim3 dimsC(dimsA.x, dimsA.y, 1);  
    
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;
    
  float *h_A, *h_B;
  
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  
  if (h_A == NULL || h_B == NULL) {
    fprintf(stderr, "Failed to allocate host matrix!\n");
    exit(EXIT_FAILURE);
  }
  
  // Number of filters (convolutions)
  int nb_features = 64;
  
  float *h_C[nb_features];
  checkCudaErrors(cudaMallocHost(&h_C[0], nb_features*mem_size_A));

   for (int j = 0; j < nb_features; j++)
	h_C[j] = (float*)((unsigned char *)h_C[0] + j*mem_size_A);
	
  // Stream creation	
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));

  // Initialize host memory
  const float valB = 1.0f;
  init_array(h_A, size_A, 1.0f);
  init_array(h_B, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

  float *d_C[nb_features];
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C[0]), nb_features*mem_size_A));
  
   for (int j = 0; j < nb_features; j++)
	d_C[j] = (float*)((unsigned char *)d_C[0] + j*mem_size_A);
	
	
  // Enable cache locking to the stream m0_stream
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_A, mem_size_A); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute);  

  // Setup execution parameters
  unsigned threads_per_block_mat_vec_mul = block_size; 
  unsigned blocks_per_grid_mat_vec_mul = dimsA.y;

   // copy persistent host memory to device	  
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, m0_stream));
  // copy normal host memory to device
  checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, m0_stream));

  // Times the matrix convolution is performed
  int nIter = 100;
  
  #ifdef EVENT_RECORD_TIME 
    	  // Allocate CUDA events that we'll use for timing
	  cudaEvent_t start, stop;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));

	  // Record the start event
	  checkCudaErrors(cudaEventRecord(start, m0_stream));	
  #endif
  
  #ifdef RECORD_SM_CYCLES  	  
	  // Get GPU time stamp value for the first time
	  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);
	  
	  cudaDeviceSynchronize(); 
  #endif
 
  
  // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)
  	for (int cnt = 0; cnt < nb_features; cnt++)   		
  		conv2DCUDA<128, 3> <<<blocks_per_grid_mat_vec_mul, threads_per_block_mat_vec_mul, 0, m0_stream>>>(d_C[cnt], d_A, d_B, dimsA.x, dimsA.y);

  	
  // Wait for the end of execution of all the threads blocks
  cudaDeviceSynchronize();   
  
  #ifdef RECORD_SM_CYCLES 
	  // Wait for the end of execution of all the threads blocks
	  cudaDeviceSynchronize();     
	    
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<16, 1, 0, m0_stream>>>(clock_end_device);
	  // Wait until the GPU time stamp has been accessed
	  cudaDeviceSynchronize();
	    
	  // Calculate execution time in cycles and display it
	  calculate_time_diff_clock(nIter); 
	  print_time_diff_clock();
	  
	  // Free memory
	  clean_clock();	 
  #endif   

  #ifdef EVENT_RECORD_TIME    
	  // Record the stop event
	  checkCudaErrors(cudaEventRecord(stop, m0_stream));

	  // Wait for the stop event to complete
	  checkCudaErrors(cudaEventSynchronize(stop));

	  float msecTotal = 0.0f;
	  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	  // Compute and print the performance
	  float msecPerMatrixMul = msecTotal / nIter;                        
	  printf( "Time: %.3f msec \n", msecPerMatrixMul);
	  
  	  // Remove events
  	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
  #endif
  
     
  // Copy result from device to host
  checkCudaErrors(cudaMemcpyAsync(h_C[0], d_C[0], mem_size_A, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));

  // Unlock persistent memory on the L2 cache
  reset_l2_locking_attr(&stream_locking_attribute, m0_stream);
  
  // Free memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C[0]));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C[0]));
  
   
}



/* MatrixUpsampling
 *
 * Description: Upsamples a matrix using a CUDA kernel and cache locking. 
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- int upsampling_mult: Upsampling multiplier
 *		- const dim3 &dimsA: Dimensions of matrix A 
 *
 * Returns:     Nothing
 *
 * */
void MatrixUpsampling(int block_size, int upsampling_mult, const dim3 &dimsA) {
 
  #ifdef RECORD_SM_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
  #endif      
        
  // Allocate host memory for matrices A and C
  unsigned int size_A = dimsA.x * dimsA.y; // Image
  dim3 dimsC(upsampling_mult*dimsA.x, upsampling_mult*dimsA.y, 1);  
    
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_C = sizeof(float) * dimsC.x * dimsC.y;
    
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

  if (h_A == NULL) {
    fprintf(stderr, "Failed to allocate host matrix!\n");
    exit(EXIT_FAILURE);
  }
  
  int nb_features = 1;
  float *h_C[nb_features];
  checkCudaErrors(cudaMallocHost(&h_C[0], nb_features*mem_size_C));

   for (int j = 0; j < nb_features; j++)
	h_C[j] = (float*)((unsigned char *)h_C[0] + j*mem_size_C);
	
  // Stream creation	
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));

  // Initialize host memory
  const float valA = 1.0f;
  init_array(h_A, size_A, valA);

  // Allocate device memory
  float *d_A;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));

  float *d_C[nb_features];
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C[0]), nb_features*mem_size_C));
  
   for (int j = 0; j < nb_features; j++)
	d_C[j] = (float*)((unsigned char *)d_C[0] + j*mem_size_C);
	

  // Enable cache locking to the stream m0_stream
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_A, mem_size_A); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute);  

  // Setup execution parameters
  unsigned threads_per_block_mat_vec_mul = block_size; 
  unsigned blocks_per_grid_mat_vec_mul = dimsA.y;

   // copy persistent host memory to device	  
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, m0_stream));
  

  // Times the matrix upsampling is performed
  int nIter = 100;
  
  #ifdef EVENT_RECORD_TIME 
    	  // Allocate CUDA events that we'll use for timing
	  cudaEvent_t start, stop;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));

	  // Record the start event
	  checkCudaErrors(cudaEventRecord(start, m0_stream));
  #endif
  
  #ifdef RECORD_SM_CYCLES   	  
	  // Get GPU time stamp value for the first time	  
	  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);
	  
	  cudaDeviceSynchronize(); 
  #endif
  
  
  // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)
  	for (int cnt = 0; cnt < nb_features; cnt++) 
  		nnUnpoolingCUDA<32, 2> <<<blocks_per_grid_mat_vec_mul, threads_per_block_mat_vec_mul, 0, m0_stream>>>(d_C[cnt], d_A, dimsA.x);	

  // Wait for the end of execution of all the threads blocks
  cudaDeviceSynchronize();   
  
  #ifdef RECORD_SM_CYCLES  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
	  // Wait until the GPU time stamp has been accessed
	  cudaDeviceSynchronize();   
			    	  
	  // Calculate execution time in cycles and display it
	  calculate_time_diff_clock(nIter); 
	  print_time_diff_clock();
	  
	  // Free memory
	  clean_clock();
  #endif

  #ifdef EVENT_RECORD_TIME    
	  // Record the stop event
	  checkCudaErrors(cudaEventRecord(stop, m0_stream));

	  // Wait for the stop event to complete
	  checkCudaErrors(cudaEventSynchronize(stop));

	  float msecTotal = 0.0f;
	  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	  // Compute and print the performance
	  float msecPerMatrixMul = msecTotal / nIter;                        
	  printf("Time: %.3f msec \n", msecPerMatrixMul);
	  
    	  // Remove events
  	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
  #endif
  
  // Copy result from device to host
  checkCudaErrors(cudaMemcpyAsync(h_C[0], d_C[0], mem_size_C, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));


  // Unlock persistent memory on the L2 cache
  reset_l2_locking_attr(&stream_locking_attribute, m0_stream);
  
  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_C[0]));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_C[0]));
   
}


