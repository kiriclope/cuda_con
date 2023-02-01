#include "librairies.h"
#include "cuPrintf.cu"
#include "CudaFunc.cu"

#include "globals.h"

#include "cuda_globals.h"

#include "utils.h"
#include "con_LR_utils.h"
#include "cuda_utils.h"

///////////////////////////////////////////////////////////////////  

int main(int argc, char *argv[]) {
  cudaPrintfInit();
  
  IF_STRUCTURE = IF_RING || IF_SPEC ; 
  
  init_globals() ;  
  init_cuda_globals() ; 
  copyHostGlobalsToDev() ;

  if(IF_LOW_RANK)
    copy_ksi_to_dev() ;
  
  setup_kernel <<<BlocksPerGrid, ThreadsPerBlock>>> () ; 
  
  for(int i_chunck=0; i_chunck<N_CHUNCKS; i_chunck++) { 
    
    cudaCheck( cudaMemcpyToSymbol(dev_i_chunck, &i_chunck, sizeof(int) ) ) ; 
    
    init_dev_globals <<<BlocksPerGrid, ThreadsPerBlock>>> () ; 
      
    kernel_gen_con_prob <<<BlocksPerGrid, ThreadsPerBlock>>> () ; 
    kernel_gen_con_vec <<<BlocksPerGrid, ThreadsPerBlock>>> () ; 
    
    if(IF_CHUNCKS) { 
      cudaCheck( cudaMemcpyFromSymbol(&con_vec_chunck, dev_con_vec_chunck, CHUNCK_SIZE * sizeof(int) ) ) ; 
      
      for(j=0;j<CHUNCK_SIZE;j++) { 
	con_vec[j + i_chunck * CHUNCK_SIZE] =  con_vec_chunck[j] ; 
	// con_vec[i_chunck + j * N_CHUNCKS] =  con_vec_chunck[j] ; 
	con_vec_chunck[j] = 0 ; 
      }
    }
	
    else 
      cudaCheck( cudaMemcpyFromSymbol(&con_vec, dev_con_vec_chunck, N_NEURONS * N_NEURONS * sizeof(int) ) ) ; 
  }
  
  cudaPrintfDisplay(stdout, true) ; 
  cudaPrintfEnd(); 

  gen_con_sparse_rep() ; 
  
  create_con_dir() ; 
  save_to_con_file() ; 
  delete_globals() ; 
  
  return 0 ; 
  
}