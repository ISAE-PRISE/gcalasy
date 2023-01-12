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

/*--------------------------- benchmarks.h -----------------------------
|  File benchmarks.h
|
|  Description: Testing scenarios for evaluating the L2 cache locking  
|
|  Version: 1.0
*-----------------------------------------------------------------------*/


// GPU benchmarks
#include "gpu_benchmarks.cuh"


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

  // Initialize variables for the GPU clock
  init_clock();
	
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

  dim3 const threads_per_block = 1024;
  dim3 const blocks_per_grid = block_size; 
  
  // Times the vector resetting is performed      
  unsigned nIter = 100;
                        
  // Get GPU time stamp value for the first time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);
  
  // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)   
  	reset_data<<<blocks_per_grid, threads_per_block, 0, m0_stream>>>(d_data_streaming, d_lut_persistent, size_streaming, size_persistent);


  // Wait for the end of execution of all the threads blocks
  cudaDeviceSynchronize();     
    
  // Get GPU time stamp value for the second time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
  // Wait until the GPU time stamp has been accessed
  cudaDeviceSynchronize();
    
  // Calculate execution time in cycles and display it
  calculate_time_diff_clock(nIter); 
  print_time_diff_clock();


  clean_clock();
  cudaStreamDestroy(m0_stream);
  cudaFree(d_lut_persistent);
  cudaFree(d_data_streaming);	
	
}



/* MatrixMultiply
 *
 * Description: Executes a matrix-matrix multiplication test using a CUDA kernel
 *		 and cache locking (locking the two matrices involved)
 *
 * Reference:	https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- const dim3 &dimsA: Dimensions of matrix A 
 *		- const dim3 &dimsB: Dimensions of matrix B
 *
 * Returns:     Nothing
 *
 * */
void MatrixMultiply(int block_size, const dim3 &dimsA, const dim3 &dimsB) {
 
  // Initialize variables for the GPU clock
  init_clock();
                   
  // Allocate host memory for matrices A, B and C
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int size_B = dimsB.x * dimsB.y;
  dim3 dimsC(dimsB.x, dimsA.y, 1);
    
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    
  float *h_A, *h_B, *h_C;
  
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
  
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix A, B or C!\n");
    exit(EXIT_FAILURE);
  }
  
  // Stream creation
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));

  // Initialize host memory
  const float valB = 0.01f;
  init_array(h_A, size_A, 1.0f);
  init_array(h_B, size_B, valB);

  // Allocate persistent device memory contiguously 
  float *d_A, *d_B, *d_C;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), 2*mem_size_A));
  d_B = d_A + size_A;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  
  // Allocate CUDA timing events
  #ifdef EVENT_RECORD_TIME 
	  cudaEvent_t start, stop;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));
  #endif
  	
  // Setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
  
  // Enable cache locking to the stream m0_stream
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_A, mem_size_A); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute);  

   // Copy persistent host memory to device	  
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, m0_stream));
  checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, m0_stream));

  #ifdef EVENT_RECORD_TIME 
	  // Record the start event
	  checkCudaErrors(cudaEventRecord(start, m0_stream));
  #endif
  
  // Times the matrix multiplication is performed
  int nIter = 300;
  	  
  // Get GPU time stamp value for the first time	  
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);

  // Execute the matrix dot product compute kernel nIter times
  for (int j = 0; j < nIter; j++)
  	MatrixMulCUDA<32> <<<grid, threads, 0, m0_stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  
  // Wait for the end of execution of all the threads blocks 	
  cudaDeviceSynchronize();   
  
  // Get GPU time stamp value for the second time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
  // Wait until the GPU time stamp has been accessed
  cudaDeviceSynchronize();   

  // Calculate execution time in cycles and display it		    	  
  calculate_time_diff_clock(nIter); 
  print_time_diff_clock();

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
  #endif
  
  
   
  // Copy results from device to host
  checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));


  #ifdef BENCHMARK_CORRECTNESS 
	  // Check result correcteness
	  printf("Checking computed result for correctness: ");
	  bool correct = true;

	  // test relative error by the formula
	  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	  double eps = 1.e-6;  // machine zero

	  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
	    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
	    double dot_length = dimsA.x;
	    double abs_val = fabs(h_C[i]);
	    double rel_err = abs_err / abs_val / dot_length;

	    if (rel_err > eps) {
	      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
	      correct = false;
	    }
	  }
	  
	  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  #endif

  // Clean up memory
  clean_clock();
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A)); 
  checkCudaErrors(cudaFree(d_C));
  
  #ifdef EVENT_RECORD_TIME    
	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
  #endif
   
}



/* MatrixVectorMulAddChain
 *
 * Description: Executes a matrix-vector multiplication-addition chain test using a CUDA kernel
 *		 and cache locking (only locking the matrix)
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- const dim3 &dimsA: Dimensions of matrix A 
 *		- const dim3 &dimsB: Dimensions of vector B
 *
 * Returns:     Nothing
 *
 * */
void MatrixVectorMulAddChain(int block_size, const dim3 &dimsA, const dim3 &dimsB) {
 
  // Initialize variables for the GPU clock
  init_clock();
                   
  // Allocate host memory for matrix A, vector array v_0[] and resultant vector v_out
  unsigned int size_matrix = dimsA.x * dimsA.y;
  unsigned int size_vector = dimsB.x * dimsB.y;
  unsigned int size_vector_output = dimsA.x * dimsB.y;
    
  unsigned int mem_size_matrix = sizeof(float) * size_matrix;
  unsigned int mem_size_vector = sizeof(float) * size_vector;
  unsigned int mem_size_vector_output = sizeof(float) * size_vector_output;

  float *m_0, *v_out;
  
  // Number of vectors that will multiply matrix A
  const int NB_VECTORS = 512;
  float *v_0[NB_VECTORS];
 
  checkCudaErrors(cudaMallocHost(&m_0, mem_size_matrix));
  checkCudaErrors(cudaMallocHost(&v_0[0], NB_VECTORS*mem_size_vector));
  checkCudaErrors(cudaMallocHost(&v_out, mem_size_vector_output));
  
  if (m_0 == NULL || v_0 == NULL || v_out == NULL) {
  	fprintf(stderr, "Failed to allocate host matrix/vector! \n");
  	exit(EXIT_FAILURE);
  }
 
   for (int j = 0; j < NB_VECTORS; j++)
	v_0[j] = (float*)((unsigned char *)v_0[0] + j*mem_size_vector);
  

  // Stream creation
  cudaStream_t m0_stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&m0_stream, cudaStreamNonBlocking));

  // Initialize host memory
  init_array(m_0, size_matrix, 1.0f);
  for (int j = 0; j < NB_VECTORS; j++)
  	init_array(v_0[j], size_vector, 1.0f);
  	
   
  // Allocate device memory
  float *d_m0, *d_v3_0, *d_v3_1;
  float *d_v0[NB_VECTORS];
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_m0), mem_size_matrix));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_v0[0]), NB_VECTORS*mem_size_vector));

   for (int j = 0; j < NB_VECTORS; j++)
	d_v0[j] = (float*)((unsigned char *)d_v0[0] + j*mem_size_vector);
  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_v3_0), mem_size_vector_output));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_v3_1), mem_size_vector_output));
    
  // Allocate CUDA timing events
  #ifdef EVENT_RECORD_TIME 
	  cudaEvent_t start, stop;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));
  #endif
  	  
  // Copy host memory to device
  checkCudaErrors(cudaMemcpyAsync(d_m0, m_0, mem_size_matrix, cudaMemcpyHostToDevice, m0_stream));
  checkCudaErrors(cudaMemcpyAsync(d_v0[0], v_0[0], NB_VECTORS*mem_size_vector, cudaMemcpyHostToDevice, m0_stream));

  unsigned threads_per_block_mat_vec_mul = block_size;
  unsigned blocks_per_grid_mat_vec_mul = dimsA.y;
  unsigned threads_per_block = block_size;
  unsigned blocks_per_grid = (size_vector + threads_per_block)/threads_per_block;
  
  // Enable cache locking to the stream m0_stream
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, d_m0, mem_size_matrix);
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute);

  #ifdef EVENT_RECORD_TIME 
	  // Record the start event
	  checkCudaErrors(cudaEventRecord(start, m0_stream));
  #endif
  
  // Execute the kernel once
  int nIter = 1; 
  
  // Get GPU time stamp value for the first time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);

  // Execute the matrix-vector multiplication-addition chain
  for (int cnt = 0; cnt < nIter; cnt++){ 
  	vectorResetCUDA<<<blocks_per_grid, threads_per_block, 0, m0_stream>>>(d_v3_1, size_vector);
  	for (int j = 0; j < NB_VECTORS; j++){
  		// Matrix-vector multiplication
		MatrixVectorMulCUDAv3<32> <<<blocks_per_grid_mat_vec_mul, threads_per_block_mat_vec_mul, 0, m0_stream>>>(d_v3_0, d_m0, d_v0[j], dimsA.x);
		cudaDeviceSynchronize();
		// Vectors addition
	        vectorAddCUDAv2<<<blocks_per_grid, threads_per_block, 0, m0_stream>>>(d_v3_1, d_v3_0, size_vector_output);
		cudaDeviceSynchronize();
	}  
	
  }
   
  
  // Get GPU time stamp value for the second time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
  // Wait until the GPU time stamp has been accessed
  cudaDeviceSynchronize();       	
    
  // Calculate execution time in cycles and display it
  calculate_time_diff_clock(nIter); 
  print_time_diff_clock();
 

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
  #endif
  
  
  // Copy result from device to host
  checkCudaErrors(cudaMemcpyAsync(v_out, d_v3_1, mem_size_vector, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));

  #ifdef BENCHMARK_CORRECTNESS 
  	bool correct = true;
	for (int j = 0; j < size_vector; j++)
  		if(v_out[j] != NB_VECTORS*dimsA.x)
  			correct = false;
  	
  	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  #endif


  // Clean up memory
  clean_clock();
  checkCudaErrors(cudaFreeHost(m_0));
  checkCudaErrors(cudaFreeHost(v_out));
  checkCudaErrors(cudaFree(d_m0));
  checkCudaErrors(cudaFreeHost(v_0[0]));
  checkCudaErrors(cudaFree(d_v0[0]));
  checkCudaErrors(cudaFree(d_v3_0));
  checkCudaErrors(cudaFree(d_v3_1));
  
  #ifdef EVENT_RECORD_TIME    
	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
  #endif

  
}


/* MatrixConvolution2
 *
 * Description: Performs a set of matrix convolution operations using a CUDA kernel
 *		and cache locking (locking the filtered matrix). In contrast to
 *		MatrixConvolution, each convolution is stored in a different matrix 
 *		instead of overwriting the same matrix all the time.
 *
 * Parameter:   
 *		- int block_size: Number of threads per block
 *		- const dim3 &dimsA: Dimensions of matrix A 
 *		- const dim3 &dimsB: Dimensions of vector B
 *
 * Returns:     Nothing
 *
 * */
void MatrixConvolution2(int block_size, const dim3 &dimsA, const dim3 &dimsB) {
 
  // Initialize variables for the GPU clock
  init_clock();
                   
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
	
	
  // Allocate CUDA events that we'll use for timing
  #ifdef EVENT_RECORD_TIME 
	  cudaEvent_t start, stop;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));
  #endif
  	

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

  #ifdef EVENT_RECORD_TIME 
	  // Record the start event
	  checkCudaErrors(cudaEventRecord(start, m0_stream));
  #endif
  
  // Times the matrix convolution is performed
  int nIter = 100;
  	  
  // Get GPU time stamp value for the first time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);

  // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)
  	for (int cnt = 0; cnt < nb_features; cnt++)   		
  		convImgCUDAv2<128, 3> <<<blocks_per_grid_mat_vec_mul, threads_per_block_mat_vec_mul, 0, m0_stream>>>(d_C[cnt], d_A, d_B, dimsA.x, dimsA.y);

  	
  // Wait for the end of execution of all the threads blocks
  cudaDeviceSynchronize();   
  
  // Get GPU time stamp value for the second time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
  // Wait until the GPU time stamp has been accessed
  cudaDeviceSynchronize();   
		    
  // Calculate execution time in cycles and display it	  
  calculate_time_diff_clock(nIter); 
  print_time_diff_clock();

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
  #endif
  
     
  // Copy result from device to host
  checkCudaErrors(cudaMemcpyAsync(h_C[0], d_C[0], mem_size_A, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));

  #ifdef BENCHMARK_CORRECTNESS 
    	// Not a definitive correcteness test
  	bool correct = true;
	for (int j = 0; j < dimsC.x*dimsC.y; j++){
		if(j % dimsC.x == 0)
			printf("\n");
		
		printf("%0.0f ", h_C[0][j]);
		
		// Check a line of the matrix
		if(j > dimsC.x + 1 && j < 2*dimsC.x - 1)
  			if(h_C[0][j] != 3*3*valB)
  				correct = false;
  	}
  	printf("\n%s \n", correct ? "Result = PASS" : "Result = FAIL");
  #endif

  // Clean up memory
  clean_clock();
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C[0]));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C[0]));
  
  #ifdef EVENT_RECORD_TIME    
	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
  #endif
   
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
 
  // Initialize variables for the GPU clock
  init_clock();
                   
  // Allocate host memory for matrices A, B and C
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
	
  // Allocate CUDA events that we'll use for timing
  #ifdef EVENT_RECORD_TIME 
	  cudaEvent_t start, stop;
	  checkCudaErrors(cudaEventCreate(&start));
	  checkCudaErrors(cudaEventCreate(&stop));
  #endif
  	

  // Enable cache locking to the stream m0_stream
  cudaStreamAttrValue stream_locking_attribute; 
  set_l2_locking_attr(&stream_locking_attribute, num_megabytes_persistent_cache, (void*)d_A, mem_size_A); 
  cudaStreamSetAttribute(m0_stream, cudaStreamAttributeAccessPolicyWindow, &stream_locking_attribute);  

  // Setup execution parameters
  unsigned threads_per_block_mat_vec_mul = block_size; 
  unsigned blocks_per_grid_mat_vec_mul = dimsA.y;

   // copy persistent host memory to device	  
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, m0_stream));


  #ifdef EVENT_RECORD_TIME 
	  // Record the start event
	  checkCudaErrors(cudaEventRecord(start, m0_stream));
  #endif
  
  // Times the matrix upsampling is performed
  int nIter = 50;
  	  
  // Get GPU time stamp value for the first time	  
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_start_device);

  // Execute compute kernel nIter times
  for (int j = 0; j < nIter; j++)
  	for (int cnt = 0; cnt < nb_features; cnt++)  nnUnpoolingImgCUDAv2<32, 2> <<<blocks_per_grid_mat_vec_mul, threads_per_block_mat_vec_mul, 0, m0_stream>>>(d_C[cnt], d_A, dimsA.x);	

  // Wait for the end of execution of all the threads blocks
  cudaDeviceSynchronize();   
  
  // Get GPU time stamp value for the second time
  getDeviceTimeCUDA<<<1, 1, 0, m0_stream>>>(clock_end_device);
  // Wait until the GPU time stamp has been accessed
  cudaDeviceSynchronize();   
		    	  
  // Calculate execution time in cycles and display it
  calculate_time_diff_clock(nIter); 
  print_time_diff_clock();

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
  #endif
  
  
   
  // Copy result from device to host
  checkCudaErrors(cudaMemcpyAsync(h_C[0], d_C[0], mem_size_C, cudaMemcpyDeviceToHost, m0_stream));
  checkCudaErrors(cudaStreamSynchronize(m0_stream));

  #ifdef BENCHMARK_CORRECTNESS 
  	bool correct = true;
  	unsigned total = 0;
	for (int j = 0; j < dimsC.x*dimsC.y; j++){
		if(j % dimsC.x == 0)
			printf("\n");
		
		printf("%0.0f ", h_C[0][j]);
		
		if(h_C[0][j] != valA)
			correct = false;

		total += h_C[0][j];
  	}
  	if (total != dimsC.x*dimsC.y)
		correct = false;
  	
  	printf("\n%s \n", correct ? "Result = PASS" : "Result = FAIL");
  #endif


  // Clean up memory
  clean_clock();
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_C[0]));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_C[0]));
  
  #ifdef EVENT_RECORD_TIME    
	  checkCudaErrors(cudaEventDestroy(start));
	  checkCudaErrors(cudaEventDestroy(stop));
  #endif
   
}


