#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mycurand.h"
#include "librairies.h"
#include "cuPrintf.cu"
#include "devFunctionProtos.h"
#include "devHostConstants.h"
#include "CudaFunc.cu"
#include "Matrix_Utils.cu"
#include "GenConProbDistDepMat.cu"

///////////////////////////////////////////////////////////////////  

int main(int argc, char *argv[]) {
  
  char* AtoB ;
  AtoB =  (char *) malloc( strlen("EE") ) ;

  if(IF_LARGE)
    AtoB = argv[1] ;
  
  unsigned long int *nbN, *Cpt ;
  nbNeurons(nbN) ;
  CptNeurons(nbN, Cpt) ;
  
  // ///////////////////////////////////////////////////////////////////    
  
  unsigned long nChunks = 1, deviceId = 0 ;
  unsigned long maxNeurons = N_NEURONS ;

  ///////////////////////////////////////////////////////////////////

  cudaDeviceProp prop;
  unsigned long maxMem = 12079136768;

  cudaCheck(cudaGetDeviceProperties(&prop, deviceId));
  printf("Global Mem = %ld, ", prop.totalGlobalMem);
  maxMem = prop.totalGlobalMem;

  if( maxMem < (unsigned long) (N_NEURONS * N_NEURONS * 4 + N_NEURONS * 4) ) {
    while( maxMem < (unsigned long) ( (N_NEURONS / nChunks) * N_NEURONS * 4   + N_NEURONS * 5 ) ) 
      nChunks += 1 ;
    
    if( nChunks % 2 !=0 )
      nChunks += 1 ;
  }
  
  maxNeurons = (unsigned long) N_NEURONS / nChunks ; // PreS

  if(IF_CHUNKS) {
    nChunks = NCHUNKS ;
    maxNeurons = MAXNEURONS ;
  }

  printf(" maxNeurons = %lu, nChunks = %lu\n", maxNeurons, nChunks);

  ///////////////////////////////////////////////////////////////////

  /* choose 256 threads per block for high occupancy */
  int ThreadsPerBlock = N_THREADS ;
  int BlocksPerGrid = ( N_NEURONS + ThreadsPerBlock-1 ) / ThreadsPerBlock;
  
  if(BlocksPerGrid > 65536) {
    printf("BlocksPerGrid exceds valid number of allowed blocks of 65536");
    exit(-1);
  }

  curandState *devStates;
  cudaCheck(cudaMalloc((void **)&devStates,  N_NEURONS * sizeof(curandState)));
  
  printf("Setup kernel ... \n");
  setup_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, time(NULL));
  cudaCheckLastError("setup_kernel failed\n");

  ///////////////////////////////////////////////////////////////////

  unsigned long long chunckSize = (unsigned long long) ( N_NEURONS / nChunks * N_NEURONS ) ;
  printf("chunckSize = %llu, ", chunckSize);

  BlocksPerGrid = (maxNeurons + ThreadsPerBlock - 1) / ThreadsPerBlock;
  printf("Threads per block : %d, Blocks per grid : %d \n", ThreadsPerBlock, BlocksPerGrid);

  // ///////////////////////////////////////////////////////////////////    

  float *dev_conVecPtr, *dev_preFactor ; //*preFactor = NULL ; // Short range connections 
  float *dev_conVecPtrLR, *dev_preFactorLR ; // Long Range connections 

  float *fullConVec = NULL, *conVec = NULL ;
  unsigned long *IdPost ;
  int *nbPost ;

  ///////////////////////////////////////////////////////////////////
  
  fullConVec = (float *) malloc((unsigned long long) N_NEURONS * N_NEURONS * sizeof(float)) ;

  IdPost = (unsigned long *) malloc( (unsigned long long) N_NEURONS * ( 2ULL + (unsigned long long) K + N_NEURONS ) * sizeof( unsigned long ) ) ; 
  nbPost = (int *) malloc( (unsigned long long) N_NEURONS * sizeof(int));

  int **nbPreSab = (int **)malloc(nbpop * sizeof(int *) );
  for(int i=0; i<nbpop; i++)
    nbPreSab[i] = (int *) malloc(nbpop * sizeof(int) );

  for(int i=0; i<nbpop; i++)
    for(int j=0;j<nbpop;j++)
      nbPreSab[i][j] = 0 ;
  
  ////////////////////////////////////////////////////////////////////    

  cudaCheck(cudaMallocHost((void **)&conVec, (unsigned long long) chunckSize * sizeof(float)));  
  cudaCheck(cudaMalloc((void **)&dev_conVecPtr, (unsigned long long) chunckSize * sizeof(float)));
  if(IF_SPACE)
    cudaCheck(cudaMalloc((void **)&dev_preFactor, (unsigned long long) nbpop * N_NEURONS * sizeof(float)));

  // cudaCheck(cudaMallocHost((void **)&preFactor, 2 * N_NEURONS * sizeof(float)));

  ///////////////////////////////////////////////////////////////////

  enum ConMat_type {
    random,distDependent
  };

  ConMat_type conMatType = random ; 
  if(IF_SPACE) {
    printf("Generating Spatial Matrix ... \n") ; 
    conMatType = distDependent ;
  }
  else 
    if(IF_RING) 
      printf("Generating Ring ... \n") ; 
    else 
      printf("Generating Random Matrix ... \n") ; 
  
  if(IF_SPEC) 
    printf("with specific connections ... \n") ; 
    
  ///////////////////////////////////////////////////////////////////
  
  double *host_Sigma ; 
  if(IF_SPACE || IF_RING)
    cudaCheck(cudaMallocHost((void **)&host_Sigma,  nbpop * sizeof(double))); 

  if(IF_SPACE || IF_RING)
    for(int j=0;j<nbpop;j++) 
      host_Sigma[j] = Sigma[j] ;

  if(IF_RING & IF_SPEC)
    for(int j=0;j<nbpop;j++) 
      host_Sigma[j] = host_Sigma[j]/sqrt(K) ;

  double *host_Dij ;
  cudaCheck(cudaMallocHost((void **)&host_Dij,  nbpop * sizeof(double))) ; 

  if(IF_SPACE || IF_RING) {
    for(int j=0;j<nbpop*nbpop;j++)
      host_Dij[j] = Dij[j] ; 
    
    printf("Sigma ") ;
    for(int j=0;j<nbpop;j++) 
      printf("%.4f ",Sigma[j]) ;  
    printf("\n") ;
    
    if(IF_LONG_RANGE) {
      printf("Sigma long range ") ;
      for(int j=0;j<nbpop;j++) 
	printf("%.4f ",SigmaLR[j]) ;  
      printf("\n") ;      
    }
  }

  cudaPrintfInit();

  switch(conMatType) {
    
  case random:
    
    for(unsigned long i = 0; i < nChunks; i++) { 

      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, i, maxNeurons);
      
      printf("Generating chunk %lu ... \n", i) ; fflush(stdout) ;  
      printf(" Generating Binary Matrix ...\n") ;

      if(IF_RING) {
	KernelGenConRing<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN, Cpt, host_Sigma, host_Dij) ; 
	KernelGenDistDepConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons) ; 
      }
      else 
	kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN) ; 

      cudaPrintfDisplay(stdout, true) ; 
      
      printf("  Copy dev to Host ... \n") ;
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (unsigned long long) chunckSize * sizeof(float), cudaMemcpyDeviceToHost)) ;
      
      for(unsigned long j=0;j<chunckSize ; j++) { 
	fullConVec[j + chunckSize * i] = (float) conVec[j] ; 
	// printf("# %llu Con %f fullConVec %f \n", j + chunckSize * i, conVec[j], fullConVec[j + chunckSize * i]) ;
	conVec[j] = 0 ;
      }
    }
    
    cudaPrintfEnd();
    
    break;
    
  case distDependent:
    initPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_preFactor);

    for(unsigned long i = 0; i < nChunks; i++) { 

      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, i, maxNeurons) ;

      printf("Generating chunk %lu ... \n", i); fflush(stdout);
	
      printf(" Generating Probabilty Matrix ...\n");
      if(DIMENSION==1) 
	KernelGenConProbMat<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr,i,maxNeurons,nbN,Cpt,host_Sigma,host_Dij) ; 
      else
	KernelGenConProbMat2D<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr,i,maxNeurons,nbN,Cpt,host_Sigma) ; 
      
      printf("  Generating preFactor ...\n");
      KernelConProbPreFactor<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, dev_preFactor, i, maxNeurons) ;
      
    }
        
    for(unsigned long i = 0; i < nChunks; i++) { 

      printf("Generating chunk %lu ... \n", i); fflush(stdout);
      initConVec<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, i, maxNeurons) ;

      printf(" Generating Probabilty Matrix ...\n");
      if(DIMENSION==1)
	KernelGenConProbMat<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr,i,maxNeurons,nbN,Cpt,host_Sigma, host_Dij) ; 
      else
	KernelGenConProbMat2D<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr,i,maxNeurons,nbN,Cpt,host_Sigma) ; 
      
      printf("  Generating Normalized Matrix ...\n") ;
      KernelConProbNorm<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, dev_preFactor, i, maxNeurons) ; 

      printf("   Generating Binary Matrix ...\n") ; 
      KernelGenDistDepConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons) ; 
      
      cudaCheck(cudaMemcpy(conVec, dev_conVecPtr, (unsigned long long) ( N_NEURONS/ nChunks ) * N_NEURONS * sizeof(float), cudaMemcpyDeviceToHost)) ; 
      
      for(unsigned long j = 0; j < chunckSize ; j++) { 

	// if(normConVec[j]!=N_NEURONS/nbpop) { 
	//   printf("\n ERRROR Chunk %llu normConVec[%llu] = %.0f \n", i, j, conVec[j] ) ; 
	//   exit(-1) ; 
	// }

	fullConVec[j + chunckSize * i] = conVec[j] ; 
	conVec[j] = 0 ; 
      }
      
    }
    
    break ; 
    
  default:
    for(unsigned long i = 0; i < nChunks; i++) 
      kernelGenConMat<<<BlocksPerGrid, ThreadsPerBlock>>>(devStates, dev_conVecPtr, i, maxNeurons, nbN);
  }
  
  printf("Free devPtr ... ");

  cudaFree(dev_conVecPtr); 

  if(IF_SPACE || IF_RING) {
    cudaFree(dev_preFactor); 
    cudaFreeHost(host_Sigma); 
  }

  cudaFreeHost(conVec); 

  printf("Done\n") ; 
  
  /////////////////////////////////////////////////////////////////// 

  // ///////////////////////////////////////////////////////////////////    
  // // On CPU 
  // ///////////////////////////////////////////////////////////////////    

  ////////////////////////////////////////////////////////////////////    
  
  unsigned long *idxPost = (unsigned long *) malloc( (unsigned long long) N_NEURONS * sizeof(unsigned long) ) ; // idx of the post neurons 
  idxPost[0] = 0 ;
  
  char *path ; // = '\0';
  CreatePath(path) ;

  if(IF_SPARSEVEC) {
    
    unsigned long counter = 0 ;
    
    if(IF_PRES) { 
      printf("Generating vectors nbPreS & IdPreS ... ") ;
      for(int i=0;i<nbpop;i++) 
	for(unsigned long l=Cpt[i];l<Cpt[i+1];l++) //Postsynaptic neurons
	  for(int j=0;j<nbpop;j++) 
	    for(unsigned long k=Cpt[j];k<Cpt[j+1];k++) { //Presynaptic neurons
	      if(fullConVec[l + N_NEURONS * k]) { // k-->l column to row 
		IdPost[counter] = k ; // ID preS of l 
		nbPost[l]++ ; // nb preS of l
		nbPreSab[i][j]++ ; 
		counter+=1 ; 
	      } 
	    } 
    } 
    else { 
      printf("Generating vectors nbPost & IdPost ... ") ; 
      for(int i=0;i<nbpop;i++) 
	for(unsigned long k=Cpt[i];k<Cpt[i+1];k++) //Presynaptic neurons 
	  for(int j=0;j<nbpop;j++) 
	    for(unsigned long l=Cpt[j];l<Cpt[j+1];l++) { //Postsynaptic neurons 
	      if(fullConVec[k + N_NEURONS * l]) { // PreS k --> Post l (column to row)
		IdPost[counter] = l ; // ID post of k 
		nbPost[k]++ ; // nb post of k 
		nbPreSab[j][i]++ ; 
		counter+=1 ; 
	      } 
	    }
    }
    
    cudaFreeHost(nbN); 
    cudaFreeHost(Cpt); 
   
    free(fullConVec) ; 
    
    // if(~IF_MATRIX) free(fullConVec) ; 
    
    // unsigned long *dev_IdPost ; 
    // int *dev_nbPost ; 
    
    // unsigned long *tempIdPost ;
    // tempIdPost = (unsigned long *) malloc( (unsigned long long) N_NEURONS * N_NEURONS * sizeof( unsigned long ) ) ; 

    // cudaCheck(cudaMalloc((void **)&dev_IdPost, (unsigned long long) N_NEURONS * N_NEURONS * sizeof( unsigned long ) ) ) ; 
    // cudaCheck(cudaMalloc((void **)&dev_nbPost, (unsigned long long) N_NEURONS * sizeof(int))) ; 

    // kernelGenSparseRep<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_conVecPtr, dev_IdPost, dev_nbPost, 0, N_NEURONS) ; 
    // cudaFree(dev_conVecPtr) ; 
 
    // cudaCheck(cudaMemcpy(tempIdPost, dev_IdPost, (unsigned long long) N_NEURONS * N_NEURONS * sizeof( unsigned long ), cudaMemcpyDeviceToHost)) ; 
    // cudaCheck(cudaMemcpy(nbPost, dev_nbPost, (unsigned long long) N_NEURONS * sizeof(int), cudaMemcpyDeviceToHost)) ; 
    // cudaFree(dev_IdPost) ; 
    // cudaFree(dev_nbPost) ; 
    
    // unsigned long l=0 ; 
    // for(unsigned long i=0;i<N_NEURONS*N_NEURONS;i++) 
    //   if(tempIdPost!=0) { 
    // 	IdPost[l] = tempIdPost[i] ; 
    // 	l++ ; 
    //   } 
    // free(tempIdPost) ; 
    
    ///////////////////////////////////////////////////////////////////  
    // Average number of Presynaptic neurons 
    /////////////////////////////////////////////////////////////////// 
    
    CheckPres(path,Cpt,IdPost,nbPost,nbPreSab) ; 
    free(nbPreSab) ; 
    
    ///////////////////////////////////////////////////////////////////    
    // Writing to File
    ///////////////////////////////////////////////////////////////////
    
    if(IF_LARGE)
      WritetoFileLarge(path,IdPost,nbPost,idxPost,AtoB) ;
    else
      WritetoFile(path,IdPost,nbPost,idxPost) ;
  }

  printf("Free Host ptr ... ") ;
  free(IdPost); 
  free(idxPost); 
  free(nbPost); 
  printf("Done\n") ;
  
  ///////////////////////////////////////////////////////////////////    
  // Writing Complete Matrix
  ///////////////////////////////////////////////////////////////////

  if(IF_MATRIX) {
    if(IF_SPARSEVEC)
      CheckSparseVec(path) ; 
    else {
      WriteMatrix(path,fullConVec) ; 
      free(fullConVec) ; 
    }
  }
  
  return 0 ;
  
}