/*--------------------------- gpu_cache_locking.cuh ---------------------
|  File gpu_cache_locking.cuh
|
|  Description: Declarations for the GPU cache locking set up  
|
|  Version: 1.1
*-----------------------------------------------------------------------*/

 
/* print_stream_attr_prop
 *
 * Description: Displays the information regarding the L2 cache locking configuration
 *
 * Parameter:   
 * 		- cudaStreamAttrValue stream_attribute: Attributes of the stream
 *
 * Returns:     Nothing
 *
 * */
void print_stream_attr_prop(cudaStreamAttrValue stream_attribute){
  printf("Persistent data address: %p \n", stream_attribute.accessPolicyWindow.base_ptr);
  printf("Persistent data size: %lu \n", stream_attribute.accessPolicyWindow.num_bytes);
  printf("Hit ratio: %f \n", stream_attribute.accessPolicyWindow.hitRatio);
  printf("Hit prop: %d \n", stream_attribute.accessPolicyWindow.hitProp);
  printf("Miss prop: %d \n", stream_attribute.accessPolicyWindow.missProp);

}


/* get_max_l2_cache_lock_size
 *
 * Description: Retrieves the maximum space that the L2 cache can lock
 *
 * Parameter:	 None
 *
 * Returns:     The maximum configurable lock size for the current device in bytes
 *
 * */
float get_max_l2_cache_lock_size(){
    cudaDeviceProp device_prop;
    int current_device = 0;
    checkCudaErrors(cudaGetDevice(&current_device));
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, current_device));
    
    return device_prop.persistingL2CacheMaxSize;
}

/* set_l2_locking_attr
 *
 * Description: Configures the L2 related attributes of a stream
 *
 * Parameter:   
 * 		- cudaStreamAttrValue stream_attribute: Attributes of the stream
 *		- float persistent_cache_spc_mb: Max amount of megabytes to reserve on the cache
 *		- void* persistent_data_ptr: Base address of the data to lock 
 *		- float persistent_data_size: Amount of megabytes to lock in cache
 *
 * Returns:     Nothing
 *
 * */
void set_l2_locking_attr(cudaStreamAttrValue* stream_attribute, float persistent_cache_spc_mb, void* persistent_data_ptr, float persistent_data_size){

  unsigned const MB2B = 1048576;
  
  // Calculate the amount of data to lock
  unsigned locked_data_size = std::min(persistent_data_size, std::min(persistent_cache_spc_mb * MB2B, get_max_l2_cache_lock_size()));
  
  // Set performance hint for the persisting L2 cache
  checkCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, locked_data_size));

  stream_attribute->accessPolicyWindow.base_ptr = persistent_data_ptr;
  stream_attribute->accessPolicyWindow.num_bytes = locked_data_size;
  stream_attribute->accessPolicyWindow.hitRatio = 1.0;
  stream_attribute->accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attribute->accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

  #ifdef PRINT_ALL
	  print_stream_attr_prop(*stream_attribute);
  #endif
}


/* reset_l2_locking_attr
 *
 * Description: Resets the L2 persistent attributes for a given stream
 *
 * Parameter:   
 * 		- cudaStreamAttrValue stream_attribute: Attributes of the stream
 *		- cudaStream_t stream: Stream used for updating the L2 persistent attribute   
 *
 * Returns:     Nothing
 *
 * */
void reset_l2_locking_attr(cudaStreamAttrValue* stream_attribute, cudaStream_t stream){

  // Clean L2 cache locking
  stream_attribute->accessPolicyWindow.num_bytes = 0;                                          
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, stream_attribute);   
  cudaCtxResetPersistingL2Cache();                                                           

  #ifdef PRINT_ALL
	  print_stream_attr_prop(*stream_attribute);
  #endif
}



