#ifndef __CUDA_GLOBALS__ 
#define __CUDA_GLOBALS__ 

#define deviceId 0 
#define maxMem (unsigned long) 12079136768 

cudaDeviceProp prop ; 

/* #define N_THREADS 512 // 512 (default), 256 threads per block for high occupancy  */
#define N_THREADS 1024 

#define IF_CHUNCKS 1 

#define N_NEURONS_PER_CHUNCK (unsigned long) 1000 
#define N_CHUNCKS N_NEURONS / N_NEURONS_PER_CHUNCK // otherwise memory problem if N_NEURONS > 20000 because of array[CHUNCK_SIZE] too large
#define CHUNCK_SIZE N_NEURONS * N_NEURONS_PER_CHUNCK 

__device__ curandState dev_states[N_NEURONS_PER_CHUNCK] ;
__device__ curandState dev_states_randi[N_NEURONS_PER_CHUNCK] ;

__device__ int dev_i_chunck ; 
__device__ unsigned long long dev_total_n_post=0 ; 

#define ThreadsPerBlock N_THREADS 
#define BlocksPerGrid (unsigned long) (N_NEURONS_PER_CHUNCK + ThreadsPerBlock - 1) / ThreadsPerBlock 

__device__ int dev_pre_pop, dev_post_pop ; 

__device__ unsigned long dev_n_per_pop[n_pop] ; 
__device__ unsigned long dev_cum_n_per_pop[n_pop+1] ; 
__device__ int dev_which_pop[N_NEURONS] ; 
__device__ float dev_K_over_Na[n_pop] ; 

__device__ int DEV_IF_STRUCTURE=0 ; 

int con_vec_chunck[CHUNCK_SIZE] ;
int *con_vec ; 

__device__ int dev_con_vec_chunck[CHUNCK_SIZE] ; 
__device__ float dev_con_prob_chunck[CHUNCK_SIZE] ; 
__device__ int dev_n_post[N_NEURONS] ; 

__device__ float prefactor[N_NEURONS], dev_theta[N_NEURONS] ; 

__device__ float dev_ksi[DUM], dev_ksi_1[DUM] ; 

__global__ void init_dev_globals() { 
  unsigned long id = threadIdx.x + blockIdx.x * blockDim.x ; 
  unsigned long i_neuron = id + dev_i_chunck * N_NEURONS_PER_CHUNCK ; 
  
  if(id < N_NEURONS_PER_CHUNCK && i_neuron < N_NEURONS) 
    for(unsigned long i=0; i<N_NEURONS; i++) { 
      /* dev_con_prob_chunck[id + i * N_NEURONS_PER_CHUNCK] = 0 ;  */
      /* dev_con_vec_chunck[id + i * N_NEURONS_PER_CHUNCK] = 0 ;  */
      dev_con_prob_chunck[i + id * N_NEURONS] = 0 ; 
      dev_con_vec_chunck[i + id * N_NEURONS] = 0 ; 
    } 
} 

__host__ void init_cuda_globals() { 

  printf("###########################################################\n") ;
  printf("N_NEURONS %lu K %.0f sqrt(K) %.2f \n", N_NEURONS, K, sqrt_K) ; 
  printf("N_CHUNCKS %lu N_NEURONS_PER_CHUNCK %lu CHUNCK_SIZE %lu \n", N_CHUNCKS, N_NEURONS_PER_CHUNCK, CHUNCK_SIZE) ; 
  printf("BlocksPerGrid %lu ThreadsPerBlock %d \n", BlocksPerGrid, ThreadsPerBlock) ; 
  printf("###########################################################\n") ; 
  
  cudaCheck(cudaGetDeviceProperties(&prop, deviceId)) ; 
  con_vec = (int*) malloc( (unsigned long) N_NEURONS * N_NEURONS * sizeof(int) ) ; 
  
  for(i=0; i<N_NEURONS*N_NEURONS; i++) 
    con_vec[i] = 0 ; 
  
  for(i=0; i<CHUNCK_SIZE; i++) 
    con_vec_chunck[i] = 0 ; 
} 

__host__ void copyHostGlobalsToDev() {
  
  cudaCheck(cudaMemcpyToSymbol( dev_n_per_pop, &n_per_pop, n_pop * sizeof(unsigned long ) ) ) ; 
  cudaCheck(cudaMemcpyToSymbol( dev_cum_n_per_pop, &cum_n_per_pop, (n_pop+1) * sizeof(unsigned long ) ) ) ; 
  cudaCheck(cudaMemcpyToSymbol( dev_which_pop, &which_pop, N_NEURONS * sizeof(int) ) ) ; 
  cudaCheck(cudaMemcpyToSymbol( dev_K_over_Na, &K_over_Na, n_pop * sizeof(float) ) ) ; 
  
}

__global__ void print_dev_globals() {

  unsigned long id = (unsigned long) threadIdx.x + blockIdx.x * blockDim.x ; 
  unsigned int i=0 ;

  if(id<1) { 
    cuPrintf("dev_n_per_pop: ") ; 
    for(i=0; i<n_pop;i++) 
      cuPrintf("%lu ", dev_n_per_pop[i] ) ; 
    cuPrintf("\n") ; 
    
    cuPrintf("dev_cum_n_per_pop: ") ; 
    for(i=0; i<n_pop+1;i++) 
      cuPrintf("%lu ", dev_cum_n_per_pop[i] ) ; 
    
    cuPrintf("dev_K_over_Na: ") ; 
    for(i=0; i<n_pop;i++) 
      cuPrintf("%.2f ", dev_K_over_Na[i] ) ; 
    cuPrintf("\n") ; 
  } 
}

#endif
