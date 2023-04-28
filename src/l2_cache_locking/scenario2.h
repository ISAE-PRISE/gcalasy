/*------------------------------ scenario2.h ------------------------------
|  File scenario2.h
|
|  Description: Testing scenarios for evaluating the L2 cache locking in a 
|		 mix-criticality scenario where the critical tasks data is 
|   		 locked and the non-critical ones are persistent interfering 
|		 kernels
|
|  Version: 1.0
*-----------------------------------------------------------------------*/




//#define EVENT_RECORD_TIME 

extern float num_megabytes_persistent_cache;

const unsigned PERSISTENT_KERNEL_THREADS_PER_BLOCK = 1024;
const unsigned PERSISTENT_KERNEL_BLOCKS_PER_GRID = 1;
const unsigned NB_SMs = 16;


/* mc_launch_reset_data
 *
 * Description: Performs a vector data reset with elements of a smaller vector
 *		 using a CUDA kernel and cache locking (locking the smaller vector).
 *		 The critical task is interfered by persistent kernels also executing 
 *		 vector data reset operations.
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- size_t size_streaming: Size of data_streaming
 *		- size_t size_persistent: Size of lut_persistent
 *		- unsigned nb_persistent_sm: Number of interfering persistent SMs
 *
 * Returns:     Nothing
 *
 * */
void mc_launch_reset_data(int block_size, size_t size_streaming, size_t size_persistent, unsigned nb_persistent_sm){

  #ifdef RECORD_SM_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
  #endif  
  
  // Host and device persistent kernel sync
  unsigned* h_persistent_kernel_act = NULL;
  unsigned* d_persistent_kernel_act = NULL;

  // Zero-copy technique
  checkCudaErrors(cudaHostAlloc((void **)&h_persistent_kernel_act, sizeof(unsigned), cudaHostAllocMapped));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_persistent_kernel_act, (void *)h_persistent_kernel_act, 0));

  // Force persistent kernels to execute indefinitely
  *h_persistent_kernel_act = 1;
  
  // Initialize host vectors
  std::vector<int> lut_persistent_vec((nb_persistent_sm+1)*size_persistent, 0);
  for (size_t i = 0; i < lut_persistent_vec.size(); ++i)
  	lut_persistent_vec[i] = i;
	
  std::vector<int> data_streaming_vec((nb_persistent_sm+1)*size_streaming, 0);

  int* h_lut_persistent = lut_persistent_vec.data();
  int* h_data_streaming = data_streaming_vec.data();
  
  // Allocate device memory for vectors
  int* d_lut_persistent[nb_persistent_sm+1];
  int* d_data_streaming[nb_persistent_sm+1];
  
  cudaMalloc(&d_lut_persistent[0], (nb_persistent_sm+1)*size_persistent * sizeof(int));
  cudaMalloc(&d_data_streaming[0], (nb_persistent_sm+1)*size_streaming * sizeof(int));
  
  for(int i = 0; i < nb_persistent_sm+1; i++){
  	d_lut_persistent[i] = d_lut_persistent[0] + i * size_persistent;
  	d_data_streaming[i] = d_data_streaming[0] + i * size_streaming;
  }
  
  // Transfer from host to device
  cudaMemcpy(d_lut_persistent[0], h_lut_persistent, (nb_persistent_sm+1)*size_persistent * sizeof(int), cudaMemcpyHostToDevice);
  	
  // Stream creation
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithPriority(&m0_stream, cudaStreamDefault, 0));
  
  cudaStream_t m1_stream[nb_persistent_sm];
  for(int i = 0; i < nb_persistent_sm; i++)
  	checkCudaErrors(cudaStreamCreateWithPriority(&m1_stream[i], cudaStreamDefault, 50));

	
  // Enable cache locking to the stream m0_stream	
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_lut_persistent[0], size_persistent*sizeof(int)); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute); 
  /*
  for(int i = 0; i < nb_persistent_sm; i++)
  	cudaStreamSetAttribute(m1_stream[i], cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute); 
*/
  
  dim3 const threads_per_block = block_size;
  dim3 const blocks_per_grid = 16; 
  
  // Times the vector resetting is performed      
  unsigned nIter = 100;  

 

  // Execute interfering kernels (persistent SMs)
  for(int i = 1; i < nb_persistent_sm+1; i++)
  	persistent_reset_data<<<PERSISTENT_KERNEL_BLOCKS_PER_GRID, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m1_stream[i-1]>>>(d_data_streaming[i], d_lut_persistent[i], size_streaming, size_persistent, d_persistent_kernel_act); 
  	 
  
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
	  getDeviceTimeCUDA<<<1, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m0_stream>>>(clock_start_device);  // We launch with 1024 threads to avoid putting it together with the pesistent kernels
	  
	  // Wait for the clock
	  cudaDeviceSynchronize(); 
  #endif     
 
   
    // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)
  	reset_data<<<blocks_per_grid, threads_per_block, 0, m0_stream>>>(d_data_streaming[0], d_lut_persistent[0], size_streaming, size_persistent);

  // Wait for the end of execution of all the threads blocks in stream m0_stream 
  checkCudaErrors(cudaStreamSynchronize(m0_stream)); 

  #ifdef RECORD_SM_CYCLES  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m0_stream>>>(clock_end_device); // We launch with 1024 threads to avoid putting it together with the pesistent kernels
	  
	  // Wait until the GPU time stamp has been accessed
	  checkCudaErrors(cudaStreamSynchronize(m0_stream)); 
	    
	  // Calculate execution time in cycles and display it
	  calculate_time_diff_clock(nIter); 
	  print_time_diff_clock();
	  
	  // Free clock memory
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

  // Stop persistent kernels to execute indefinitely
  *h_persistent_kernel_act = 0;
  
  // Wait for persistent kernels to finish
  for(int i = 0; i < nb_persistent_sm; i++)
  	checkCudaErrors(cudaStreamSynchronize(m1_stream[i]));

  // Unlock persistent memory on the L2 cache
  reset_l2_locking_attr(&stream_locking_attribute, m0_stream);
  
  // Free memory
  cudaStreamDestroy(m0_stream);
  for(int i = 0; i < nb_persistent_sm; i++)
  	cudaStreamDestroy(m1_stream[i]);
  checkCudaErrors(cudaFree(d_lut_persistent[0]));
  checkCudaErrors(cudaFree(d_data_streaming[0]));	
  
  // Free persistent kernel activation flag 
  checkCudaErrors(cudaFreeHost(h_persistent_kernel_act));
	
}



/* mc_launch_convolution
 *
 * Description: Performs the convolution of two matrices using a CUDA kernel 
 *		 and cache locking.
 *		 The critical task is interfered by persistent kernels executing 
 *		 vector data reset operations.
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- const dim3 &dimsA: Dimensions of matrix A 
 *		- const dim3 &dimsB: Dimensions of matrix B
 *		- size_t size_streaming: Size of data_streaming (interfering kernels)
 *		- size_t size_persistent: Size of lut_persistent (interfering kernels)
 *		- unsigned nb_persistent_sm: Number of interfering persistent SMs
 *
 * Returns:     Nothing
 *
 * */
void mc_launch_convolution(int block_size, const dim3 &dimsA, const dim3 &dimsB, size_t size_streaming, size_t size_persistent, unsigned nb_persistent_sm){

   #ifdef RECORD_SM_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
   #endif  
  
  // Allocate host memory for matrices A, B and C
  unsigned int size_A = dimsA.x * dimsA.y; // Image
  unsigned int size_B = dimsB.x * dimsB.y; // Filter
    
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
	
	
  // Initialize convolution host memory
  const float valB = 1.0f;
  init_array(h_A, size_A, 1.0f);
  init_array(h_B, size_B, valB);

  // Allocate convolution device memory
  float *d_A, *d_B;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

  float *d_C[nb_features];
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C[0]), nb_features*mem_size_A));
  
   for (int j = 0; j < nb_features; j++)
	d_C[j] = (float*)((unsigned char *)d_C[0] + j*mem_size_A);
	
  
  // Host and device persistent kernel sync
  unsigned* h_persistent_kernel_act = NULL;
  unsigned* d_persistent_kernel_act = NULL;

  // Zero-copy technique
  checkCudaErrors(cudaHostAlloc((void **)&h_persistent_kernel_act, sizeof(unsigned), cudaHostAllocMapped));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_persistent_kernel_act, (void *)h_persistent_kernel_act, 0));

  // Force persistent kernels to execute indefinitely
  *h_persistent_kernel_act = 1;
  
  
  // Initialize interfering host vectors
  std::vector<int> lut_persistent_vec(nb_persistent_sm*size_persistent, 0);
  for (size_t i = 0; i < lut_persistent_vec.size(); ++i)
  	lut_persistent_vec[i] = i;
	
  std::vector<int> data_streaming_vec(nb_persistent_sm*size_streaming, 0);

  int* h_lut_persistent = lut_persistent_vec.data();
  int* h_data_streaming = data_streaming_vec.data();
  
  // Allocate device memory for interfering vectors
  int* d_lut_persistent[nb_persistent_sm];
  int* d_data_streaming[nb_persistent_sm];
  
  if(nb_persistent_sm != 0){
	  checkCudaErrors(cudaMalloc(&d_lut_persistent[0], nb_persistent_sm*size_persistent * sizeof(int)));
	  checkCudaErrors(cudaMalloc(&d_data_streaming[0], nb_persistent_sm*size_streaming * sizeof(int)));
  }
  
  for(int i = 0; i < nb_persistent_sm; i++){
  	d_lut_persistent[i] = d_lut_persistent[0] + i * size_persistent;
  	d_data_streaming[i] = d_data_streaming[0] + i * size_streaming;
  }
  
  // Transfer data from host to device
  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  
  if(nb_persistent_sm != 0)
  	checkCudaErrors(cudaMemcpy(d_lut_persistent[0], h_lut_persistent, nb_persistent_sm*size_persistent * sizeof(int), cudaMemcpyHostToDevice));
  
  	
  // Stream creation
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithPriority(&m0_stream, cudaStreamDefault, 0));
  
  cudaStream_t m1_stream[nb_persistent_sm];
  for(int i = 0; i < nb_persistent_sm; i++)
  	checkCudaErrors(cudaStreamCreateWithPriority(&m1_stream[i], cudaStreamDefault, 50));

	
  // Enable cache locking to the stream m0_stream	
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_A, mem_size_A); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute);  
  
  /*
  for(int i = 0; i < nb_persistent_sm; i++)
  	cudaStreamSetAttribute(m1_stream[i], cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute); 
*/
  dim3 const threads_per_block = block_size;
  dim3 const blocks_per_grid = dimsA.y; 
  
  // Times the vector resetting is performed      
  unsigned nIter = 100;               

  // Execute interfering kernels (persistent SMs)
  for(int i = 0; i < nb_persistent_sm; i++)
  	persistent_reset_data<<<PERSISTENT_KERNEL_BLOCKS_PER_GRID, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m1_stream[i]>>>(d_data_streaming[i], d_lut_persistent[i], size_streaming, size_persistent, d_persistent_kernel_act); 
  	
  	 
  // Allocate CUDA timing events
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
	  getDeviceTimeCUDA<<<1, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m0_stream>>>(clock_start_device);  // We launch with 1024 threads to avoid putting it together with the pesistent kernels
	  
	  // Wait for the clock
	  cudaDeviceSynchronize(); 
  #endif  
   
  // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)
  	for (int cnt = 0; cnt < nb_features; cnt++)   		
  		conv2DCUDA<1024, 3> <<<blocks_per_grid, threads_per_block, 0, m0_stream>>>(d_C[cnt], d_A, d_B, dimsA.x, dimsA.y);

  // Wait for the end of execution of all the threads blocks in stream m0_stream 
  checkCudaErrors(cudaStreamSynchronize(m0_stream)); 

  #ifdef RECORD_SM_CYCLES  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m0_stream>>>(clock_end_device); // We launch with 1024 threads to avoid putting it together with the pesistent kernels
	  
	  // Wait until the GPU time stamp has been accessed
	  checkCudaErrors(cudaStreamSynchronize(m0_stream)); 
	    
	  // Calculate execution time in cycles and display it
	  calculate_time_diff_clock(nIter); 
	  print_time_diff_clock();
	  
	  // Free clock memory
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

  // Stop persistent kernels to execute indefinitely
  *h_persistent_kernel_act = 0;
  
  // Wait for persistent kernels to finish
  for(int i = 0; i < nb_persistent_sm; i++)
  	checkCudaErrors(cudaStreamSynchronize(m1_stream[i]));


  // Copy result from device to host
  checkCudaErrors(cudaMemcpyAsync(h_C[0], d_C[0], nb_features*mem_size_A, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));
  
  // Unlock persistent memory on the L2 cache
  reset_l2_locking_attr(&stream_locking_attribute, m0_stream);
  
  // Convolution related memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C[0]));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C[0]));
  
  cudaStreamDestroy(m0_stream);
  
  // Interfering kernels related memory
  for(int i = 0; i < nb_persistent_sm; i++)
  	cudaStreamDestroy(m1_stream[i]);
  	
  if(nb_persistent_sm != 0){
  	checkCudaErrors(cudaFree(d_lut_persistent[0]));
  	checkCudaErrors(cudaFree(d_data_streaming[0]));	
  }
  // Free persistent kernel activation flag 
  checkCudaErrors(cudaFreeHost(h_persistent_kernel_act));
	
}

/* mc_launch_convolution3d
 *
 * Description: Performs the 3D convolution of two matrices using a CUDA kernel 
 *		 and cache locking.
 *		 The critical task is interfered by persistent kernels executing 
 *		 vector data reset operations.
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- const dim3 &dimsA: Dimensions of matrix A 
 *		- const dim3 &dimsB: Dimensions of matrix B
 *		- size_t size_streaming: Size of data_streaming (interfering kernels)
 *		- size_t size_persistent: Size of lut_persistent (interfering kernels)
 *		- unsigned nb_persistent_sm: Number of interfering persistent SMs
 *
 * Returns:     Nothing
 *
 * */
void mc_launch_convolution3d(int block_size, const dim3 &dimsA, const dim3 &dimsB, size_t size_streaming, size_t size_persistent, unsigned nb_persistent_sm){

   #ifdef RECORD_SM_CYCLES 
	  // Initialize variables for the GPU clock
	  init_clock();
   #endif 
  
  // Number of output features
  int nb_filters = 64;
  
  // Allocate host memory for matrices A, B and C
  unsigned int size_A = dimsA.x * dimsA.y * dimsA.z; // Image
  unsigned int size_B = dimsB.x * dimsB.y * dimsB.z; // Filter
  dim3 dimsC(dimsA.x, dimsA.y, nb_filters);  
    
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;
  unsigned int mem_size_C = sizeof(float) * dimsC.x * dimsC.y * dimsC.z;
    
  float *h_A, *h_B[nb_filters];
  
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  checkCudaErrors(cudaMallocHost(&h_B[0], nb_filters*mem_size_B));
  
  if (h_A == NULL || h_B[0] == NULL) {
    fprintf(stderr, "Failed to allocate host matrix!\n");
    exit(EXIT_FAILURE);
  }
  
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
	
  // Initialize convolution host memory
  init_array(h_A, size_A, 1.0f);
  init_array(h_B[0], nb_filters*size_B, 1.5f);

  // Allocate convolution device memory
  float *d_A, *d_B[nb_filters];
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B[0]), nb_filters*mem_size_B));
  
  for (int j = 0; j < nb_filters; j++)
  	d_B[j] = (float*)((unsigned char *)d_B[0] + j*mem_size_B);

  float *d_C;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  
  // Host and device persistent kernel sync
  unsigned* h_persistent_kernel_act = NULL;
  unsigned* d_persistent_kernel_act = NULL;

  // Zero-copy technique
  checkCudaErrors(cudaHostAlloc((void **)&h_persistent_kernel_act, sizeof(unsigned), cudaHostAllocMapped));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_persistent_kernel_act, (void *)h_persistent_kernel_act, 0));

  // Force persistent kernels to execute indefinitely
  *h_persistent_kernel_act = 1;
  
  // Initialize interfering host vectors
  std::vector<int> lut_persistent_vec(nb_persistent_sm*size_persistent, 0);
  for (size_t i = 0; i < lut_persistent_vec.size(); ++i)
  	lut_persistent_vec[i] = i;
	
  std::vector<int> data_streaming_vec(nb_persistent_sm*size_streaming, 0);

  int* h_lut_persistent = lut_persistent_vec.data();
  int* h_data_streaming = data_streaming_vec.data();
  
  // Allocate device memory for interfering vectors
  int* d_lut_persistent[nb_persistent_sm];
  int* d_data_streaming[nb_persistent_sm];
  
  if(nb_persistent_sm != 0){
	checkCudaErrors(cudaMalloc(&d_lut_persistent[0], nb_persistent_sm*size_persistent * sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_data_streaming[0], nb_persistent_sm*size_streaming * sizeof(int)));
  }
  
  for(int i = 0; i < nb_persistent_sm; i++){
  	d_lut_persistent[i] = d_lut_persistent[0] + i * size_persistent;
  	d_data_streaming[i] = d_data_streaming[0] + i * size_streaming;
  }
  
  // Transfer data from host to device
  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B[0], h_B[0], nb_filters*mem_size_B, cudaMemcpyHostToDevice));
  
  if(nb_persistent_sm != 0)
  	checkCudaErrors(cudaMemcpy(d_lut_persistent[0], h_lut_persistent, nb_persistent_sm*size_persistent * sizeof(int), cudaMemcpyHostToDevice));
  
  	
  // Stream creation
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithPriority(&m0_stream, cudaStreamDefault, 0));
  
  cudaStream_t m1_stream[nb_persistent_sm];
  for(int i = 0; i < nb_persistent_sm; i++)
  	checkCudaErrors(cudaStreamCreateWithPriority(&m1_stream[i], cudaStreamDefault, 50));

	
  // Enable cache locking to the stream m0_stream	
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_A, mem_size_A); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute);  
  
  /*
  for(int i = 0; i < nb_persistent_sm; i++)
  	cudaStreamSetAttribute(m1_stream[i], cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute); 
*/
  
  dim3 const threads_per_block = block_size;
  dim3 const blocks_per_grid = dimsA.y; 
  
  // Times the vector resetting is performed      
  unsigned nIter = 100;               

  // Execute interfering kernels (persistent SMs)
  for(int i = 0; i < nb_persistent_sm; i++)
  	persistent_reset_data<<<PERSISTENT_KERNEL_BLOCKS_PER_GRID, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m1_stream[i]>>>(d_data_streaming[i], d_lut_persistent[i], size_streaming, size_persistent, d_persistent_kernel_act); 
  	 
  // Allocate CUDA timing events
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
	  getDeviceTimeCUDA<<<1, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m0_stream>>>(clock_start_device);  // We launch with 1024 threads to avoid putting it together with the pesistent kernels
	  
	  // Wait for the clock
	  cudaDeviceSynchronize(); 
  #endif  
   
  // Execute compute kernel nIter times with random batch values
  for (int j = 0; j < nIter; j++)
 	 for (int cnt = 0; cnt < nb_filters; cnt++)  
 		 conv3D_normReLu<1024, 3> <<<blocks_per_grid, threads_per_block, 0, m0_stream>>>(d_C, d_A, d_B[cnt], cnt, dimsA.x, dimsA.y, dimsA.z, nb_filters, 0.73f, 2.0f, 4.0f, 0.57f, 0.2f, 0.00001f);
 	 

  // Wait for the end of execution of all the threads blocks in stream m0_stream 
  checkCudaErrors(cudaStreamSynchronize(m0_stream)); 

  #ifdef RECORD_SM_CYCLES  
	  // Get GPU time stamp value for the second time
	  getDeviceTimeCUDA<<<1, PERSISTENT_KERNEL_THREADS_PER_BLOCK, 0, m0_stream>>>(clock_end_device); // We launch with 1024 threads to avoid putting it together with the pesistent kernels
	  
	  // Wait until the GPU time stamp has been accessed
	  checkCudaErrors(cudaStreamSynchronize(m0_stream)); 
	    
	  // Calculate execution time in cycles and display it
	  calculate_time_diff_clock(nIter); 
	  print_time_diff_clock();
	  
	  // Free clock memory
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
  
  // Wait for persistent kernels to finish
  for(int i = 0; i < nb_persistent_sm; i++)
  	checkCudaErrors(cudaStreamSynchronize(m1_stream[i]));


  // Copy result from device to host
  checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));
  
  // Unlock persistent memory on the L2 cache
  reset_l2_locking_attr(&stream_locking_attribute, m0_stream);
  
  // Convolution related memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B[0]));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B[0]));
  checkCudaErrors(cudaFree(d_C));
  
  cudaStreamDestroy(m0_stream);
  
  // Interfering kernels related memory
  for(int i = 0; i < nb_persistent_sm; i++)
  	cudaStreamDestroy(m1_stream[i]);
  	
  if(nb_persistent_sm != 0){
  	checkCudaErrors(cudaFree(d_lut_persistent[0]));
  	checkCudaErrors(cudaFree(d_data_streaming[0]));	
  }
  // Free persistent kernel activation flag 
  checkCudaErrors(cudaFreeHost(h_persistent_kernel_act));
	
}



