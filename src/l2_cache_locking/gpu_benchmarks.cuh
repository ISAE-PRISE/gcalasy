/*--------------------------- gpu_benchmarks.cuh ---------------------
|  File gpu_benchmarks.cuh
|
|  Description: CUDA kernel declarations for the GPU benchmarks.
|	
|  Note: Benchmarks are functional but not optimized. Used for testing.
|
|  Version: 1.1
*-----------------------------------------------------------------------*/
 
 
/* conv2DCUDA
 *
 * Description: CUDA kernel for the 2D convolution of two matrices. 
 *		 Borders are not processed.
 *		 The block ID is used for selecting the input matrix row.
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the filter matrix 
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix
 *		- float *img: Input matrix
 *		- float *kernel: Filter matrix
 *		- int Nx: img width
 *		- int Ny: img height
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE, const int KERNEL_SIZE> __global__ void conv2DCUDA(float *imgf, float *img, float *kernel, int Nx, int Ny){

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

	// Kernel max size
	const int FILTER_MAX_SIZE = KERNEL_SIZE*KERNEL_SIZE;
	
	// Locked data for the current thread block
	__shared__ float sdata[FILTER_MAX_SIZE];

	if (tx<FILTER_MAX_SIZE)
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
	    	if (idx < Nx*Ny && ix + cnt < Nx-center && ix + cnt != Nx && iy < Ny-center){
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

// 3d convolution where the depth of the input matrix and the filter are the same
/* conv3D_normReLu
 *
 * Description: CUDA kernel for the 3D convolution of two matrices (feature and filter). 
 *		 The convolution bias, batch normalization and ReLu operations
 *		 are performed.
 *		 The block ID is used for selecting the input matrix row.
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the filter matrix 
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix
 *		- float *img: Input matrix
 *		- float *kernel: Filter matrix
 *		- int kernel_id: Filter matrix being used
 *		- int Nx: img width
 *		- int Ny: img height
 *		- int Nz: img depth
 *		- int out_ch_offset: Number of output channels (number of filters)
 *		- float conv_bias: Convolution bias
 *		- float mean: Batch normalization mean
 *		- float var: Batch normalization variance
 *		- float weight: Batch normalization weight
 *		- float bias: Batch normalization bias
 *		- float EPSILON: Batch normalization epsilon (added to the variance to avoid dividing by zero)
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE, const int KERNEL_SIZE> __global__ void conv3D_normReLu(float *imgf, float *img, float *kernel, int kernel_id, unsigned Nx, unsigned Ny, unsigned Nz, int out_ch_offset, float conv_bias, float mean, float var, float weight, float bias, float EPSILON){

	// Block index
  	int bx = blockIdx.x;

  	// Thread index
	int tx = threadIdx.x;

	// Lock depth data into the L1 cache
  	volatile __shared__ float sdata[BLOCK_SIZE];

        unsigned center = (KERNEL_SIZE - 1)/2;
        unsigned xz_img_off = Nx*Nz;
        unsigned xz_ker_off = KERNEL_SIZE*Nz;

        unsigned xz_in_img_off = bx*Nx*out_ch_offset;

        // Across the horizontal 
        for (int cnt = 0; cnt<Nx; cnt++){

		// Reset shared data
		sdata[tx] = 0;

		__syncthreads();

		int idx = xz_in_img_off + cnt*out_ch_offset + kernel_id;
		int ii, jj;

		for (int kj = 0; kj<KERNEL_SIZE; kj++){
	                jj = kj + (bx - center);

	                if((jj >= 0) && (jj <= Ny-1)){

		        for (int ki = 0; ki<KERNEL_SIZE; ki++){
		           ii = ki + (cnt - center);

		              if ((ii >= 0) && (ii <= Nx-1)){
		                    // Across the depth 
		                    for (int kk = tx; kk<Nz; kk+= BLOCK_SIZE)                    
		                       sdata[tx] += img[jj*xz_img_off + ii*Nz + kk] * kernel[kj*xz_ker_off + ki*Nz + kk];    	                    
		               }

		          }

		     }

		 }


		  __syncthreads();


		  // Unroll the reduction method  
		  if (BLOCK_SIZE >= 1024){ 
		  	if(tx < 512)
		  		sdata[tx] += sdata[tx + 512]; 

			__syncthreads(); 

		  }
		  if (BLOCK_SIZE >= 512){ 
			  if(tx < 256)
			  	sdata[tx] += sdata[tx + 256]; 

			__syncthreads(); 

		  }
		  if (BLOCK_SIZE >= 256){ 
			  if(tx < 128)
			  	sdata[tx] += sdata[tx + 128]; 

			__syncthreads(); 

		  }
		  if (BLOCK_SIZE >= 128){ 
			  if(tx < 64)
			  	sdata[tx] += sdata[tx + 64]; 

			__syncthreads(); 

		  }

		  // No need of sync when a single warp is involved
		  if (tx < 32){
		  	if (BLOCK_SIZE >= 64) 
		 		sdata[tx] += sdata[tx + 32];

		  	if (BLOCK_SIZE >= 32) 
		 		sdata[tx] += sdata[tx + 16];

		 	if (BLOCK_SIZE >= 16) 
		 		sdata[tx] += sdata[tx + 8];

		  	if (BLOCK_SIZE >= 8) 
		 		sdata[tx] += sdata[tx + 4];

		  	if (BLOCK_SIZE >= 4) 
		 		sdata[tx] += sdata[tx + 2];

		  	if (BLOCK_SIZE >= 2) 
		  		sdata[tx] += sdata[tx + 1];

		  } 

	   	 if(tx == 0){
			 // Add element of the final image matrix in the device
			 imgf[idx] = (sdata[tx] + conv_bias);
			 // Batch normalization
		 	 imgf[idx] = ((imgf[idx] - mean)/sqrtf(var + EPSILON)) * (weight) + bias;
		  	 // ReLu (max(0,x))
		 	 imgf[idx] = (imgf[idx] < 0) ? 0 : imgf[idx];	        
		 }

       }  


}



/* nnUnpoolingCUDA
 *
 * Description: CUDA kernel for Nearest-Neighbor upsampling a matrix. 
 *		The threads move across the output image. 
 *		The block ID is used for selecting the input matrix row.
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *		- int UP_SAMP_SIZE: Upsampling size  (total size = KERNEL_SIZE*KERNEL_SIZE)
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix of image
 *		- float *img: Input matrix of image
 *		- int Nx: img width
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE, int UP_SAMP_SIZE> __global__ void nnUnpoolingCUDA(float *imgf, float *img, int Nx){

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


 /* get_smid
 *
 * Description: CUDA function that returns the SM ID on which the block is being executed
 *
 * Parameter:   None
 *
 * Returns:     Nothing
 *
 * */
__device__ __inline__ uint get_smid(void){
	uint sm_id;
	asm volatile ("mov.u32 %0, %smid;" : "=r"(sm_id));
	
	return sm_id;

}

 /* persistent_reset_data
 *
 * Description: CUDA kernel for reseting the values of a vector. 
 *		Like "reset_data" but with the possibility of running
 *		persistently.
 *
 * Parameter:   
 *		- int* data_streaming: Vector to reset
 *		- int const* lut_persistent: Vector used to reset data_streaming with
 *		- size_t data_streaming_size: Size of data_streaming
 *		- size_t lut_persistent_size: Size of lut_persistent
  *		- unsigned* active: Flag signaling whether to keep executing or not 
 *
 * Returns:     Nothing
 *
 * */
__global__ void persistent_reset_data(int* data_streaming, int const* lut_persistent, size_t data_streaming_size, size_t lut_persistent_size, unsigned* active){
    size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;
    
    /*
    if(threadIdx.x == 0)
    	printf("persistent_reset_data SM ID: %d \n ", get_smid());
    */

    while(*active){
    	for(size_t i = idx; i < data_streaming_size; i += stride)
    		data_streaming[i] = lut_persistent[i % lut_persistent_size];
    }

}



