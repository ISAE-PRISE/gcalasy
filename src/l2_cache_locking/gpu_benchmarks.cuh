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

/*--------------------------- gpu_benchmarks.cuh ---------------------
|  File gpu_benchmarks.cuh
|
|  Description: CUDA kernel declarations for the GPU benchmarks  
|
|  Version: 1.0
*-----------------------------------------------------------------------*/
 
 
 /* MatrixMulCUDA
 *
 * Description: CUDA kernel for the dot product of two matrices
 *		Obtained from NVIDIA sample: 
 *		https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu  
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *
 * Parameter:   
 *		- float *C: Resultant matrix
 *		- float *A: First matrix
 *		- float *B: Second matrix
 *		- float wA: Matrix A width
 *		- float wB: Matrix B width
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {

  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
	    // Declaration of the shared memory array As used to
	    // store the sub-matrix of A
	    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

	    // Declaration of the shared memory array Bs used to
	    // store the sub-matrix of B
	    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	    // Load the matrices from device memory
	    // to shared memory; each thread loads
	    // one element of each matrix
	    As[ty][tx] = A[a + wA * ty + tx];
	    Bs[ty][tx] = B[b + wB * ty + tx];

	    // Synchronize to make sure the matrices are loaded
	    __syncthreads();

	    // Multiply the two matrices together;
	    // each thread computes one element
	    // of the block sub-matrix
	#pragma unroll
	    for (int k = 0; k < BLOCK_SIZE; ++k) 
	    	Csub += As[ty][k] * Bs[k][tx];
	    
	    // Synchronize to make sure that the preceding
	    // computation is done before loading two new
	    // sub-matrices of A and B in the next iteration
	    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}


/* MatrixVectorMulCUDAv3
 *
 * Description: CUDA kernel for the dot product of a matrix and a vector 
 *		A thread block performs a row operation.  
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *
 * Parameter:   
 *		- float *C: Resultant vector
 *		- float *A: First matrix (m x n)
 *		- float *B: First vector (n x 1)
 *		- float wA: Matrix A width
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE> __global__ void MatrixVectorMulCUDAv3(float *C, float *A, float *B, int wA){

  // Block index
  int bx = blockIdx.x;

  // Thread index
  int tx = threadIdx.x;

  // Store a final matrix row x vector multiplication 
  float sum = 0;

  // Store a matrix row x vector multiplication per thread 
  __shared__ float sum_vec[BLOCK_SIZE];
  sum_vec[tx] = 0;
  	
  // Wait for initialization
  __syncthreads();
  
  // Move through one matrix row
  #pragma unroll
  for (int cnt = 0; cnt < wA; cnt += BLOCK_SIZE) 
  	sum_vec[tx] += A[tx + cnt] * B[tx + cnt];
  
  __syncthreads();
  
  // Make one thread perform the final addition and write to the device memory
  if (tx == 0){
    	#pragma unroll	    
  	for (int cnt = 0; cnt < BLOCK_SIZE; cnt++) 
	  	sum += sum_vec[cnt];
	  	
  	// Write to device memory a matrix row x vector multiplication
  	C[bx] = sum;
  }
}



/* MatrixVectorMulCUDAv4
 *
 * Description: CUDA kernel for the dot product of a matrix and a vector. 
 *		In contrast to MatrixVectorMulCUDAv3, here we assume that a single thread block performs
 *		the whole matrix-vector dot product multiplication. This makes this entire operation 
 * 		be performed on 1 SM, leaving other SMs free for other matrix-vector operations.
 *		    
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *
 * Parameter:   
 *		- float *C: Resultant vector
 *		- float *A: First matrix (m x n)
 *		- float *B: First vector (n x 1)
 *		- float wA: Matrix A width
 *		- float hA: Matrix A height
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE> __global__ void MatrixVectorMulCUDAv4(float *C, float *A, float *B, int wA, int hA) {

  // Thread index
  int tx = threadIdx.x;

  // Store a final matrix row x vector multiplication 
  float sum = 0;
  	
  // Loop through all the matrix rows     
  for (int y = 0; y < hA; y++){ 
	// Store a matrix row x vector multiplication per thread 
	__shared__ float sum_vec[BLOCK_SIZE];
	sum_vec[tx] = 0;
	  	
	// Wait for reset
	__syncthreads();
  		  
	// Move through one matrix row
	#pragma unroll
	for (int x = 0; x < wA; x += BLOCK_SIZE) 
	 	sum_vec[tx] += A[tx + x] * B[tx + x];
	  
	__syncthreads();
	  
	// Make one thread perform the final addition of elements for a given row 
	// and write to the device memory
	if (tx == 0){
    		#pragma unroll	    
	  	for (int cnt = 0; cnt < BLOCK_SIZE; cnt++) 
		  	sum += sum_vec[cnt];
		  	
  		// Write to device memory a matrix row x vector multiplication
	  	C[y] = sum;
  	  	sum = 0;
	}
	
	// Wait until thread 0 has finished before proceeding to another row
	__syncthreads();
   }
   
   
}


/* convImgCUDAv2
 *
 * Description: CUDA kernel for the convolution of two matrices
 *		In contrast to convImgCUDA, this compute kernel works indepedently of the matrix width
 *		In Jetson AGX Orin 64 GB, it is advisable to make use of the 4 processing blocks 
 *		(i.e., use at least a BLOCK_SIZE >= n*128 being n 1,2,...8) for faster computations
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the filter matrix 
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input matrix of image
 *		- float *kernel: Filter matrix
 *		- float Nx: img width
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE, const int KERNEL_SIZE> __global__ void convImgCUDAv2(float *imgf, float *img, float *kernel, int Nx, int Ny){

  	// Block index
  	int bx = blockIdx.x;
  
  	// Thread index
	int tx = threadIdx.x;
	
	// the center of the filter
	int center = (KERNEL_SIZE - 1)/2;
	
	// each block is assigned to a row of an image, iy integer index of y
	int iy = bx + center;

	// each thread is assigned to a pixel of a row, ix integer index of x
	int ix = tx + center;
	
	
	const int K2 = KERNEL_SIZE*KERNEL_SIZE;
	// Locked data for the current thread block
	__shared__ float sdata[K2];

	if (tx<K2)
	    sdata[tx] = kernel[tx];

	// Wait until the filter matrix data is locked into the L1 cache
	__syncthreads();
	

	int ii = 0;
	int jj = 0;
	int sum = 0;

	// Across the horizontal 
	for (int cnt = 0; cnt<Nx; cnt+= BLOCK_SIZE){
        	int idx = iy*Nx + (ix + cnt);
        	// Avoid borders
	    	if (idx < Nx*Ny && ix + cnt != Nx-1 && ix + cnt != Nx && iy != Ny-1){
	    	    // Apply filter
		    for (int ki = 0; ki<KERNEL_SIZE; ki++){
			for (int kj = 0; kj<KERNEL_SIZE; kj++){
			   ii = kj + ix - center + cnt;
			   jj = ki + iy - center;
		    	   sum += img[jj*Nx + ii] * sdata[ki*KERNEL_SIZE + kj];
		    	}	 
		   }
		   
		   // Write element of the final image matrix in the device
	           imgf[idx] = sum;
	           
	           // Reset sum variable
	           sum = 0;
	       }  
	}

}



/* maxUnpoolingImgCUDAv2
 *
 * Description: CUDA kernel for Nearest-Neighbor upsampling a matrix. 
 *		The threads move across the output image.
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *		- int UP_SAMP_SIZE: Upsampling size  (total size = KERNEL_SIZE*KERNEL_SIZE)
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input matrix of image
 *		- float Nx: img width
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE, int UP_SAMP_SIZE> __global__ void nnUnpoolingImgCUDAv2(float *imgf, float *img, int Nx){

  	// Block index
  	int bx = blockIdx.x;

  	// Thread index
  	int tx = threadIdx.x;
  
  	// Across the output image horizontal 
  	for (int cnt = 0; cnt<Nx; cnt+= BLOCK_SIZE){
  	    	#pragma unroll	
  		for (int i = 0; i<UP_SAMP_SIZE; i++)
  		    	#pragma unroll	
  			for (int j = 0; j<UP_SAMP_SIZE; j++)
  				imgf[UP_SAMP_SIZE*(bx*Nx) + i*Nx + UP_SAMP_SIZE*(tx + cnt) + j] = img[bx*Nx + (tx + cnt)];
  
  	}
  
}


 /* vectorAddCUDAv2
 *
 * Description: CUDA kernel for vector addition. 
 *		In contrast to vectorAddCUDA, this kernel adds the result in one of the input vectors.
 *
 * Parameter:   
 *		- float *B: Second and resultant vector
 *		- float *A: First vector
 *		- float numElements: Vector size
 *
 * Returns:     Nothing
 *
 * */
__global__ void vectorAddCUDAv2(float *B, const float *A, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    B[idx] += A[idx]; 
}


 /* vectorResetCUDA
 *
 * Description: CUDA kernel for reseting the values of a vector. 
 *
 * Parameter:   
 *		- float *A: Vector to reset
 *		- float numElements: Vector size
 *
 * Returns:     Nothing
 *
 * */
__global__ void vectorResetCUDA(float *A, int numElements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
        A[i] = 0; 
}

 /* reset_data
 *
 * Description: CUDA kernel for reseting the values of a vector. 
 *		Obtained from Lei Mao's blog: 
 *		https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/  
 *
 * Parameter:   
 *		- int* data_streaming: Vector to reset
 *		- int const* lut_persistent: Vector used to reset data_streaming with
 *		- size_t data_streaming_size: Size of data_streaming
 *		- size_t lut_persistent_size: Size of lut_persistent
 *
 * Returns:     Nothing
 *
 * */
__global__ void reset_data(int* data_streaming, int const* lut_persistent, size_t data_streaming_size, size_t lut_persistent_size){
    size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < data_streaming_size; i += stride)
    	data_streaming[i] = lut_persistent[i % lut_persistent_size];
    
}


