#include "librairies.h" 
#include "CudaFunc.cu" 
#include "cuPrintf.cu"

#define N_NEURONS 10000UL 
#define N_THREADS 512 
#define BlocksPerGrid (N_NEURONS + N_THREADS - 1) / N_THREADS 

#include "rand_kernel.h" 
#include "dum.h" 

int main(int argc, char *argv[]) {

  cudaPrintfInit() ; 
  
  printf("BlocksPerGrid %lu ThreadsPerBlock %d \n", BlocksPerGrid, N_THREADS) ;

  cudaCheck( cudaMalloc( (void **) &dev_states,  N_NEURONS * sizeof(curandState) ) ) ; 
  
  printf("Setup kernel ... ") ; 
  setup_kernel <<< BlocksPerGrid, N_THREADS>>> ( dev_states ) ; 
  cudaCheckLastError("\n Failed to setup kernel \n") ; 
  printf("\u2713 \n") ; 
  
  init_dum() ; 
  
  d_dum_func <<< BlocksPerGrid, N_THREADS >>> (dev_states) ;   
  print_d_dum<<< BlocksPerGrid, N_THREADS >>> () ; 
  
  cudaCheck(cudaMemcpyToSymbol(d_dum_array, &dum_array, N_NEURONS*sizeof(double) ) ) ; 

  print_d_dum<<< BlocksPerGrid, N_THREADS >>> () ; 
    
  d_dum_func<<< BlocksPerGrid, N_THREADS >>> (dev_states) ; 
  print_d_dum<<< BlocksPerGrid, N_THREADS >>> () ; 
  
  cudaCheck(cudaMemcpyFromSymbol(&dum_array, d_dum_array, N_NEURONS*sizeof(double) ) ) ; 
  
  printf("%.2f %.2f %.2f\n", dum_array[0], dum_array[1], dum_array[2]) ;

  cudaPrintfDisplay(stdout, true) ;
  
  cudaPrintfEnd() ;   
  
}

