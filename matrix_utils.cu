#ifndef __MATRIXUTILS__
#define __MATRIXUTILS__

///////////////////////////////////////////////////////////////////    

__global__ void setup_kernel(curandState *state, unsigned long long seed ) {
  unsigned long int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets different seed, a different sequence number, no offset */
  if(id < N_NEURONS) 
    curand_init(seed * (id + 7), id, 0, &state[id]);
}

///////////////////////////////////////////////////////////////////    

__device__ float unif_dist(curandState *state, unsigned long int kNeuron) {
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/
  float randNumber= 0.0 ; 
  if(kNeuron < N_NEURONS) {
    curandState localState = state[kNeuron]; /* state in global memory */
    randNumber = curand_uniform(&localState);
    state[kNeuron] = localState;
  }
  return randNumber;
}

__device__ float normal_dist(curandState *state, unsigned long int kNeuron, float mean, float std) {
  /*RETURNS ONE SAMPLE FROM NORMAL DISTRIBUTION*/
  float randNumber= 0.0 ; 
  if(kNeuron < N_NEURONS) {
    curandState localState = state[kNeuron]; /* state in global memory */
    randNumber = curand_normal(&localState) ;
    state[kNeuron] = localState;
  }
  return randNumber;
}

///////////////////////////////////////////////////////////////////    

__host__ void nbNeurons(unsigned long * &n_per_pop) {
  
  cudaCheck(cudaMallocHost((void **)&n_per_pop, n_pop * sizeof( unsigned long )));
  printf("Number of neurons : ") ;
  unsigned long i = 0; 
  while(i<n_pop) {
    if(i==0) 
      n_per_pop[i] = N_NEURONS*popSize ;
    else
      n_per_pop[i] = (unsigned long) ( N_NEURONS - n_per_pop[0] ) / max( (n_pop-1), 1 ) ;
       
    printf("%lu ", n_per_pop[i]) ;
    ++i ;
  }
  printf("\n") ;
}

///////////////////////////////////////////////////////////////////    
 
__host__ void CptNeurons(unsigned long* n_per_pop, unsigned long* &cum_n_per_pop) {
  cudaCheck(cudaMallocHost((void **)&cum_n_per_pop, n_pop * sizeof( unsigned long int)));
  printf("Counter : ") ;

  unsigned long i,j;
  for(i=0;i<n_pop+1;i++) {
    cum_n_per_pop[i] = 0 ;
    for(j=0;j<i;j++) {
      cum_n_per_pop[i] = cum_n_per_pop[i] + n_per_pop[j] ; 
    } 
    printf("%lu ", cum_n_per_pop[i]) ;
  } 
  printf("\n") ; 
}

///////////////////////////////////////////////////////////////////    

__global__ void initConVec(float *dev_conVec, int lChunck, unsigned long int maxNeurons) {
  unsigned long id = threadIdx.x + blockIdx.x * blockDim.x; 
  unsigned long kNeuron = id + lChunck * maxNeurons;
  unsigned long i;
  if(id < maxNeurons && kNeuron < N_NEURONS) 
    for(i = 0; i < N_NEURONS; i++) 
      dev_conVec[i + id * N_NEURONS] = 0 ; 
      // dev_conVec[id + i * N_NEURONS] = 0 ; 
}

///////////////////////////////////////////////////////////////////    

__host__ __device__ int whichPop(unsigned long int neuronIdx) {
  
  // const double PROPORTIONS[4] = {.75,.25,0,0} ;

  int popIdx = 0 ;
  unsigned long propPop = N_NEURONS*popSize ;
  unsigned long propPop0 = N_NEURONS*popSize ;
  
  while( neuronIdx > propPop-1 ) {
    popIdx++ ;
    // propPop += propPop ;
    propPop += (unsigned long ) ( N_NEURONS - propPop0 ) / max( (n_pop-1), 1 ) ;
    // propPop += int( N_NEURONS * ( 1. - popSize ) ) / max( (n_pop-1), 1 ) ;
  }
  return popIdx ;
}

///////////////////////////////////////////////////////////////////    

__host__ int dirExists(const char *path) {
  struct stat info;

  if(stat( path, &info ) != 0)
    return 0;
  else if(info.st_mode & S_IFDIR)
    return 1;
  else
    return 0;
}

///////////////////////////////////////////////////////////////////    

__host__ void CreatePath(char *&path) {
  
  char* cdum ;
  cdum = (char *) malloc(500 * sizeof(char) ) ;
    
  if(IF_LOW_RANK) 
    sprintf(cdum, "../../connectivity/%dpop/N%d/K%.0f/low_rank/xi_%.2f_mean_%.2f_var/", n_pop, (int) (N_NEURONS/nbPref), K, MEAN_XI, VAR_XI) ;
  elif(IF_SPEC)
    sprintf(cdum, "../../connectivity/%dpop/N%d/K%.0f/spec/kappa_%.2f/", n_pop, (int) (N_NEURONS/nbPref), K, Sigma[0]) ; 
  else 
    sprintf(cdum, "../../connectivity/%dpop/N%d/K%.0f", n_pop, (int) (N_NEURONS/nbPref), K) ; 
  
  path = (char *) malloc( (strlen(cdum) + 500) * sizeof(char) ) ;
  strcpy(path,cdum) ;

  char *mkdirp ;   
  mkdirp = (char *) malloc( (strlen(path) + 500) * sizeof(char) );
  
  strcpy(mkdirp,"mkdir -p ") ; 
  strcat(mkdirp,path) ; 
  printf("%s\n",mkdirp) ;

  const int dir_err = system(mkdirp) ; 
  printf("%d\n",dir_err) ;

  if(-1 == dir_err) {
    printf("error creating directory : ");
  }
  else { 
    printf("Created directory : ") ;
  }
  printf("%s\n",path) ;
}

///////////////////////////////////////////////////////////////////    

__host__ void CheckPres(char *path, unsigned long* cum_n_per_pop, unsigned long *id_post, int *n_post, int **nbPreSab) {

  // for(unsigned long k=0;k<N_NEURONS;k++) 
  //   nbPreSab[ whichPop(k) ][ whichPop( id_post[k] ) ] += n_post[k] ;
  
  printf("\nAverage nbPreS : ");

  const char* str ;
  if(IF_Nk)
    str ="/nbPreSab_Nk.txt" ;
  else
    str ="/nbPreSab.txt" ;

  char *strPreSab ;
  strPreSab =  (char *) malloc( (strlen(path) + strlen(str) + 100 ) * sizeof(char) ) ;

  strcpy(strPreSab,path) ;
  strcat(strPreSab,str) ;

  FILE * pFile ;
  pFile = fopen (strPreSab,"w");

  for(int i=0;i<n_pop;i++) 
    for(int j=0;j<n_pop;j++) { 
      printf("%.3f ", (double) nbPreSab[i][j] / (double) ( cum_n_per_pop[i+1]-cum_n_per_pop[i] ) ) ;
      fprintf(pFile,"%.3f ", nbPreSab[i][j] / (double) (cum_n_per_pop[i+1]-cum_n_per_pop[i]) ) ;
    }
  printf("\n");

  fclose(pFile);
}

///////////////////////////////////////////////////////////////////    

__host__ void WritetoFile(char *path, unsigned long int *id_post, int *n_post, unsigned long int *idx_post) {

  printf(" Writing to Files :\n") ;

  char *nbpath  ;
  char *idxpath ;
  char  *Idpath ;

  const char *strid_post = "/id_post.dat";
  const char *stridx_post = "/idx_post.dat";
  const char *strn_post = "/n_post.dat"; 

  if(IF_Nk) {
    strid_post = "/id_post_Nk.dat";
    stridx_post = "/idx_post_Nk.dat";
    strn_post = "/n_post_Nk.dat"; 
  }

  if(IF_PRES) {
    strid_post = "/id_pre.dat" ; 
    stridx_post = "/idx_pre.dat"; 
    strn_post = "/n_pre.dat"; 
  }
  
  nbpath =  (char *) malloc( (strlen(path)+strlen(strn_post) + 100) * sizeof(char) ) ;
  idxpath = (char *) malloc( (strlen(path)+strlen(stridx_post) + 100) * sizeof(char) ) ;
  Idpath = (char *)  malloc( (strlen(path)+strlen(strid_post) + 100 ) * sizeof(char) ) ;

  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strn_post) ;
  strcat(idxpath,stridx_post) ;
  strcat(Idpath,strid_post) ;

  unsigned long nbCon = 0;
  for(unsigned long i = 0; i < N_NEURONS; i++) 
    nbCon += n_post[i];
  
  FILE *fid_post, *fn_post, *fidx_post ;
  
  printf("sizeof id_post %lu \n",  nbCon) ;

  for(int i=0;i<10;i++)
    printf("%lu ",id_post[i]);
  printf("\n"); 

  fid_post = fopen(Idpath, "wb");
  fwrite(id_post, sizeof(*id_post) , nbCon , fid_post);
  fclose(fid_post);

  printf("%s\n",Idpath) ;

  for(unsigned long i=1;i<N_NEURONS;i++)
    idx_post[i] = idx_post[i-1] + n_post[i-1] ;

  for(int i=0;i<10;i++)
    printf("%lu ",idx_post[i]);
  printf("\n"); 

  fidx_post = fopen(idxpath, "wb") ;
  fwrite(idx_post, sizeof(*idx_post) , N_NEURONS , fidx_post); 
  fclose(fidx_post);

  printf("%s\n",idxpath) ;

  for(int i=0;i<10;i++)
    printf("%d ",n_post[i]);
  printf("\n"); 

  fn_post = fopen(nbpath, "wb") ;
  fwrite(n_post, sizeof(*n_post) , N_NEURONS , fn_post) ;
  fclose(fn_post);

  printf("%s\n",nbpath) ;
  printf("Done\n") ;

}

///////////////////////////////////////////////////////////////////    

__host__ void WriteMatrix(char *path, float *conVec) {

  printf("Writing Cij Matrix to : \n") ;   
  const char* strMatrix = "/Cij_Matrix.dat"; 
  char *pathMatrix ; 
  pathMatrix = (char *) malloc( (strlen(path)+strlen(strMatrix)+100) * sizeof(char) ) ;

  strcpy(pathMatrix,path) ;
  strcat(pathMatrix,strMatrix) ;

  printf("%s\n",pathMatrix);
  
  FILE *Out;
  Out = fopen(pathMatrix,"wb");  
  fwrite(conVec, sizeof(float), (unsigned long) N_NEURONS * N_NEURONS , Out) ;
  fclose(Out) ;
}

__host__ void CheckSparseVec(char * path) {

  char *nbpath  ;
  char *idxpath ;
  char *Idpath ;

  const char *strid_post = "/id_post.dat";
  const char *stridx_post = "/idx_post.dat";
  const char *strn_post = "/n_post.dat"; 

  if(IF_Nk) {
    strid_post = "/id_post_Nk.dat" ; 
    stridx_post = "/idx_post_Nk.dat"; 
    strn_post = "/n_post_Nk.dat";    
  }

  if(IF_PRES) {
    strid_post = "/id_pre.dat" ; 
    stridx_post = "/idx_pre.dat"; 
    strn_post = "/n_pre.dat"; 
  }

  nbpath =  (char *) malloc( (strlen(path)+strlen(strn_post) + 100 ) * sizeof(char) ) ;
  idxpath = (char *) malloc( (strlen(path)+strlen(stridx_post) + 100 ) * sizeof(char) ) ;
  Idpath = (char *)  malloc( (strlen(path)+strlen(strid_post) + 100 ) * sizeof(char) ) ;
  
  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strn_post) ;
  strcat(idxpath,stridx_post) ;
  strcat(Idpath,strid_post) ;

  int *n_post ;
  unsigned long int *idx_post ;
  unsigned long int *id_post ;

  n_post = new int [N_NEURONS] ;
  idx_post = new unsigned long int [N_NEURONS] ;

  FILE *fn_post, *fidx_post, *fid_post ;
  
  int dum ;
  
  fn_post = fopen(nbpath, "rb") ;
  dum = fread(&n_post[0], sizeof n_post[0], N_NEURONS , fn_post);  
  fclose(fn_post);
  
  fidx_post = fopen(idxpath, "rb") ;
  dum = fread(&idx_post[0], sizeof idx_post[0], N_NEURONS , fidx_post);
  fclose(fidx_post);
  
  unsigned long int nbposttot = 0 ;
  for(int j=0 ; j<N_NEURONS; j++)
    nbposttot += n_post[j] ;
  
  id_post = new unsigned long int [nbposttot] ;
  
  fid_post = fopen(Idpath, "rb");
  dum = fread(&id_post[0], sizeof id_post[0], nbposttot , fid_post); 
  fclose(fid_post);
  
  printf("Writing Cij Matrix to : \n") ;
  const char* strMatrix = "/Cij_Matrix.dat";
  char *pathMatrix ;
  pathMatrix = (char *) malloc( (strlen(path)+strlen(strMatrix)+100) * sizeof(char) ) ;

  strcpy(pathMatrix,path) ;
  strcat(pathMatrix,strMatrix) ;

  printf("%s\n",pathMatrix);
  
  FILE *Out;
  Out = fopen(pathMatrix,"wb");
  
  float **M ;
  M = new float*[N_NEURONS] ;
  for(int i=0;i<N_NEURONS;i++) 
    M[i] = new float[N_NEURONS]() ;

  for(int i=1;i<N_NEURONS;i++)
    idx_post[i] = idx_post[i-1] + n_post[i-1] ;
  
  for(int i=0;i<N_NEURONS;i++) 
    for(int l=idx_post[i]; l<idx_post[i]+n_post[i]; l++) 
      M[id_post[l]][i] = 1 ;
  
  for (int i=0; i<N_NEURONS; i++) 
    fwrite(M[i], sizeof(M[i][0]), N_NEURONS, Out) ; 
  
  fclose(Out) ; 
  delete [] M ;

}


///////////////////////////////////////////////////////////////////    

__global__ void kernelGenSparseRep(float *dev_conVec, unsigned long *dev_id_post, int *dev_n_post, int lChunk, unsigned long maxNeurons) {
  
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread 
  unsigned long kNeuron = id + lChunk * maxNeurons; 
  unsigned long i ; 
  int n_post ; 

  if(id < maxNeurons & kNeuron < N_NEURONS) {

    dev_n_post[kNeuron] = 0 ; 
    n_post = 0 ; 
    
    for(i=0;i<N_NEURONS;i++) 
      if(dev_conVec[i + id * N_NEURONS]) {// id-->i column to row
	dev_id_post[i + id * N_NEURONS] = i ; 
	n_post += 1 ; 
      } 
    dev_n_post[kNeuron] = n_post ; 
  }  
}

///////////////////////////////////////////////////////////////////    

__global__ void kernelGenConMat(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int* n_per_pop) { 

  /* indexing of matrix row + clm x N_NEURONS*/ 
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x ; 
  unsigned long kNeuron = id + lChunck * maxNeurons ; 
  unsigned long i ;
  
  if(id < maxNeurons && kNeuron < N_NEURONS) 
    for(i=0; i<N_NEURONS; i++) { // i is row and id is clmn 
      // cuPrintf("id %d i %d \n",id,i) ;     
      if( (float) ( K ) / (float) ( n_per_pop[whichPop(i)] ) >= unif_dist(state, kNeuron) ) // neuron[id] receives input from i 
    	dev_conVec[i + id * N_NEURONS] = 1.0 ; 
      else 
    	dev_conVec[i + id * N_NEURONS] = 0.0 ; 
    } 
} 

///////////////////////////////////////////////////////////////////    

// __host__ void GenSparseVec(float *conVec, unsigned long int *id_post, int *n_post, int *nbPreS, int lChunk, unsigned long int maxNeurons) {
    
//     unsigned long int counter = 0 ;
    
//     for(int i=0;i<n_pop;i++) 
//       for(unsigned long int k=cum_n_per_pop[i];k<cum_n_per_pop[i+1];k++) { //Presynaptic neurons
// 	for(int j=0;j<n_pop;j++) 
// 	  for(unsigned long int l=cum_n_per_pop[j];l<cum_n_per_pop[j+1];l++) //Postsynaptic neurons
// 	    if(fullConVec[k + N_NEURONS * l]) { // k-->l column to row
// 	      id_post[counter] = l ;
// 	      n_post[k]++ ;
// 	      nbPreSab[j][i]++ ;
// 	      counter+=1 ;
// 	    }   
// 	// printf("PresId %d, nPost %d \r",k,n_post[k]);
//       }

// }

#endif
